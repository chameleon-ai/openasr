use crate::{Segment, Transcription, WordTimestamp};

use super::slicing::{AudioSlice, LongFormBenchmarkMetadata};
use super::timeline::TimelineMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentTimeDomain {
    RelativeToSliceContent,
    AbsoluteOriginal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SliceTranscript {
    pub slice: AudioSlice,
    pub text: String,
    pub segments: Vec<Segment>,
    pub time_domain: SegmentTimeDomain,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentMergePolicy {
    pub max_gap_seconds: f32,
    pub redundant_overlap_ratio: f32,
    pub redundant_min_words: usize,
}

impl Default for SegmentMergePolicy {
    fn default() -> Self {
        Self {
            max_gap_seconds: 1.2,
            redundant_overlap_ratio: 0.8,
            redundant_min_words: 4,
        }
    }
}

/// Largest gap (seconds) between the last matched word in the previous segment
/// and the first matched word in the current that the seam-stitch will accept
/// when both segments carry **acoustic** (not interpolated) word timestamps. A
/// true slice-boundary re-read sits inside the inter-slice overlap: for the
/// default 0.5s overlap the two word-instances land within a single word or
/// two, and even a jittery re-read stays under ~2s. Anything wider is a
/// legitimate repeat -- a verse re-sung, a question-and-echo, an ABAB pattern
/// in a song, etc. -- and stitching would consume real words from both
/// segments. Interpolated (`approximate_word_timestamps`) families tile word
/// spans uniformly across the segment, so their regap is a function of the
/// tile rather than acoustic reality; the guard is skipped for them.
const SEGMENT_STITCH_MAX_REGAP_SECONDS: f32 = 2.0;

/// How far past the previous segment's end the current slice's matched seam
/// phrase may still BEGIN and count as a re-read, when both segments carry
/// **acoustic** word timestamps. A true re-read re-emits audio the previous
/// slice already covered, so that audio (and hence the current slice's copy of
/// the phrase) sits inside the inter-slice overlap, i.e. strictly before
/// `previous.end`. A phrase whose first word in the current slice begins
/// after `previous.end` is NOT a re-read: it is NEW audio the previous slice
/// never transcribed. The regap guard above cannot catch this shape, because
/// a deliberate repeat landing right at the cut ("there we go <pause> there
/// we go", a question-and-echo) can have a small regap (under
/// `SEGMENT_STITCH_MAX_REGAP_SECONDS`) while still starting clear after
/// `previous.end`. Stitching that phrase would consume real words from the
/// current segment and lose audible content, so it is refused up to this
/// tolerance (enough for onset-timing jitter on a word straddling the cut).
const SEGMENT_STITCH_REREAD_MAX_PAST_PREV_END_SECONDS: f32 = 0.15;

/// How far inside `previous.end` the earlier word-instance of a
/// single-unit seam may have ended and still count as the word the slice
/// cut landed on (the clamp check in `apply_suffix_prefix_stitch`). A
/// straddling word is clamped at the segment end, or its decode end-estimate
/// lands a few tens of ms short of it. A word that finished clearly before
/// the cut was followed by NEW speech: its same-text successor across the
/// boundary is a genuine back-to-back repeat ("Yeah. Yeah."), not a re-read
/// of the cut word.
const SEGMENT_STITCH_SEAM_CLAMP_TOLERANCE_SECONDS: f32 = 0.1;

/// Confidence ceiling of the new slice's head word for the B-side seam
/// phantom rule (below). A re-read of audio the previous slice already
/// committed is a second, weaker decode of that same audio, so the artifact
/// carries less confidence than the reading that already owns it. Genuine
/// straddling words are decoded with the whole word's audio in view and
/// sit at or above this value in every observed seam pair, except shapes
/// the committed-word floor and the width gate below already keep.
const SEAM_PHANTOM_HEAD_MAX_CONFIDENCE: f32 = 0.5;

/// Confidence floor of the committed word for the B-side seam phantom rule:
/// only a committed word decoded with certainty hands its audio over to be
/// protected. When the committed word is itself uncertain, both readings
/// are live candidates and neither may be eaten, no matter how weak or
/// stretched the re-read looks.
const SEAM_PHANTOM_COMMITTED_MIN_CONFIDENCE: f32 = 0.85;

/// Minimum width of the re-read window for the B-side seam phantom rule.
/// The phantom pads out over the committed tail plus the silence the cut
/// left behind, so its window is far wider than the speech it covers; a
/// genuine straddling word stays close to its spoken duration. Genuine
/// held words (a sung vowel over several seconds of music) are far wider
/// still but carry the confidence of the slice that heard them whole, and
/// the two confidence gates above keep them on the table.
const SEAM_PHANTOM_HEAD_MIN_WIDTH_SECONDS: f32 = 0.9;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LongFormAssembleStats {
    pub skipped_silent_chunks: usize,
    pub duplicate_merge_count: usize,
}

#[derive(Debug)]
pub struct TranscriptAssembler {
    timeline: TimelineMap,
    merge_policy: SegmentMergePolicy,
    segments: Vec<Segment>,
    /// Decode-scope provenance aligned one-for-one with `segments`. This is
    /// intentionally carried beside the public transcript contract: local
    /// speaker counters are meaningful only inside the slice that produced
    /// them, but that implementation detail must not leak into serialized
    /// transcript segments.
    speaker_scope_by_segment: Vec<Option<usize>>,
    stats: LongFormAssembleStats,
    /// End of the audio region (original-timeline seconds) already committed by
    /// prior slices. Consecutive slices overlap at forced/energy cuts, so the
    /// next slice re-decodes that region; anything it emits before this point is
    /// a redundant re-read (or a weak-model hallucination of partial audio) and
    /// is trimmed by time before it can survive into the transcript.
    committed_end_original: Option<f32>,
    approximate_word_timestamps: bool,
}

impl TranscriptAssembler {
    pub fn new(timeline: TimelineMap, merge_policy: SegmentMergePolicy) -> Self {
        Self {
            timeline,
            merge_policy,
            segments: Vec::new(),
            speaker_scope_by_segment: Vec::new(),
            stats: LongFormAssembleStats::default(),
            committed_end_original: None,
            approximate_word_timestamps: false,
        }
    }

    /// Only decoders without native word alignment may regenerate estimates
    /// after stitching. Acoustic/native anchors retain their original times.
    pub(crate) fn with_approximate_word_timestamps(mut self, approximate: bool) -> Self {
        self.approximate_word_timestamps = approximate;
        self
    }

    pub fn push_slice_result(&mut self, transcript: SliceTranscript) {
        self.push_slice_result_with_scope(transcript, None);
    }

    /// Assemble one independently decoded slice while retaining the exact
    /// scope that owns its local speaker labels. The scope travels through the
    /// same overlap trimming and duplicate suppression as the segment itself,
    /// so callers never need to guess provenance back from final timestamps.
    pub(crate) fn push_slice_result_with_speaker_scope(
        &mut self,
        transcript: SliceTranscript,
        speaker_scope: usize,
    ) {
        self.push_slice_result_with_scope(transcript, Some(speaker_scope));
    }

    fn push_slice_result_with_scope(
        &mut self,
        mut transcript: SliceTranscript,
        speaker_scope: Option<usize>,
    ) {
        // The trim boundary is the region committed by *prior* slices; this
        // slice's own span is folded into the boundary afterwards so the next
        // slice trims against it (even if this slice is silent / emits nothing).
        let trim_boundary = self.committed_end_original;
        let slice_committed_end = self.slice_content_end_original(&transcript.slice);
        self.committed_end_original = Some(match self.committed_end_original {
            Some(previous) => previous.max(slice_committed_end),
            None => slice_committed_end,
        });
        transcript.text = transcript.text.trim().to_string();
        if transcript.text.is_empty()
            && transcript
                .segments
                .iter()
                .all(|segment| segment.text.trim().is_empty())
        {
            self.stats.skipped_silent_chunks += 1;
            return;
        }
        if transcript.segments.is_empty() {
            let sample_rate = 16_000.0_f32;
            transcript.segments.push(Segment {
                start: 0.0,
                end: transcript.slice.content_duration_samples() as f32 / sample_rate,
                text: transcript.text.clone(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            });
            transcript.time_domain = SegmentTimeDomain::RelativeToSliceContent;
        }
        let time_domain = transcript.time_domain;
        let slice = transcript.slice.clone();
        // Text stitch is a cross-slice seam repair for wordless seq2seq
        // families. Same-slice adjacent cues (Whisper word timestamps, cue
        // splitter output) stay distinct even when their texts share a tail.
        let mut cross_slice_seam = true;
        for mut segment in transcript.segments {
            segment.text = segment.text.trim().to_string();
            if segment.text.is_empty() {
                continue;
            }
            let mut mapped = self.map_segment_time(&segment, &slice, time_domain);
            // Cross-slice stitch first. DecodeInvariant families (qwen/moss/
            // firered/mimo) force interpolated words for cue splitting; those
            // words make a midpoint trim eat the textual re-read prefix and
            // would skip stitch if we required `words.is_empty()`. Same-slice
            // adjacent cues still skip stitch via `cross_slice_seam`.
            if cross_slice_seam && self.try_stitch_seam_overlap(&mut mapped) {
                self.stats.duplicate_merge_count += 1;
                if mapped.text.trim().is_empty() {
                    continue;
                }
                self.segments.push(mapped);
                self.speaker_scope_by_segment.push(speaker_scope);
                cross_slice_seam = false;
                continue;
            }
            // Time-domain overlap trim: drop any leading words / whole segments
            // whose audio lies in the region a prior slice already committed.
            // Skip when stitch already consumed the re-read; interpolated
            // seq2seq words would otherwise delete the matching prefix.
            if let Some(boundary) = trim_boundary
                && trim_committed_overlap(&mut mapped, boundary)
            {
                self.stats.duplicate_merge_count += 1;
                continue;
            }
            // B-side seam phantom: the new slice re-read the committed tail
            // as a different low-confidence head word. Runs AFTER the midpoint
            // trim, which may have dropped the leading committed fragments
            // that front a stretched phantom: the head inspected is the first
            // word that survives into the transcript. The midpoint trim keeps
            // such a stretched window (its majority sits past the boundary),
            // so only this rule removes it.
            if cross_slice_seam && self.drop_seam_phantom_head(&mut mapped) {
                self.stats.duplicate_merge_count += 1;
                if mapped.text.trim().is_empty() {
                    continue;
                }
            }
            if self.try_drop_redundant_segment(&mapped) {
                self.stats.duplicate_merge_count += 1;
                continue;
            }
            cross_slice_seam = false;
            // Distinct segments are kept distinct: the post-ASR cue
            // re-segmentation pass owns subtitle granularity, so the assembler
            // no longer coalesces adjacent same-speaker segments into paragraph
            // blobs. Only exact / high-overlap duplicates from slice overlap are
            // dropped above.
            self.segments.push(mapped);
            self.speaker_scope_by_segment.push(speaker_scope);
        }
    }

    pub fn into_transcription(self) -> Transcription {
        self.into_parts().0
    }

    pub fn into_parts(self) -> (Transcription, LongFormAssembleStats) {
        let (transcription, stats, _) = self.into_parts_with_speaker_scopes();
        (transcription, stats)
    }

    /// Return the assembled transcript together with exact per-segment decode
    /// scope provenance. The vector is aligned with `transcription.segments`;
    /// `None` denotes a caller that did not opt into local-speaker scopes.
    pub(crate) fn into_parts_with_speaker_scopes(
        self,
    ) -> (Transcription, LongFormAssembleStats, Vec<Option<usize>>) {
        let text = crate::transcript_text::join_segment_texts(
            self.segments.iter().map(|segment| segment.text.as_str()),
        );
        let transcription = Transcription {
            truncated_decodes: Vec::new(),
            unnamed_speakers: Vec::new(),
            text,
            segments: self.segments,
            longform: None,
            language: None,
            ..Default::default()
        };
        debug_assert_eq!(
            transcription.segments.len(),
            self.speaker_scope_by_segment.len(),
            "speaker scope provenance must stay aligned with assembled segments"
        );
        (transcription, self.stats, self.speaker_scope_by_segment)
    }

    pub fn benchmark_metadata(&self) -> LongFormBenchmarkMetadata {
        LongFormBenchmarkMetadata {
            chunk_count: self.segments.len(),
            skipped_silent_chunks: self.stats.skipped_silent_chunks,
            duplicate_merge_count: self.stats.duplicate_merge_count,
            provenance: vec!["core.longform.assembler".to_string()],
        }
    }

    fn map_segment_time(
        &self,
        segment: &Segment,
        slice: &AudioSlice,
        time_domain: SegmentTimeDomain,
    ) -> Segment {
        let sample_rate = 16_000.0_f32;
        let mut start = segment.start.max(0.0);
        let mut end = segment.end.max(start);
        let mut content_offset = 0.0_f32;
        if time_domain == SegmentTimeDomain::RelativeToSliceContent {
            content_offset = slice.content_start_sample as f32 / sample_rate;
            start += content_offset;
            end += content_offset;
        }
        let original_start = self.timeline.map_processed_to_original_seconds(start);
        let original_end = self
            .timeline
            .map_processed_to_original_seconds(end)
            .max(original_start);
        let words = segment
            .words
            .iter()
            .filter_map(|word| {
                map_word_time_to_original(
                    word,
                    content_offset,
                    time_domain,
                    &self.timeline,
                    original_start,
                    original_end,
                )
            })
            .collect();
        Segment {
            start: original_start,
            end: original_end,
            text: segment.text.clone(),
            speaker: segment.speaker.clone(),
            speaker_label: segment.speaker_label.clone(),
            speaker_person_id: None,
            speaker_snapshot_label: None,
            words,
        }
    }

    /// End of this slice's content span in original-timeline seconds. The
    /// `content_end_sample` indexes the same processed/plan audio domain that
    /// [`Self::map_segment_time`] maps from, so mapping it through the timeline
    /// yields the original-time cut point this slice commits up to.
    fn slice_content_end_original(&self, slice: &AudioSlice) -> f32 {
        let sample_rate = 16_000.0_f32;
        let processed_end = slice.content_end_sample as f32 / sample_rate;
        self.timeline
            .map_processed_to_original_seconds(processed_end)
    }

    fn try_drop_redundant_segment(&self, current: &Segment) -> bool {
        let Some(previous) = self.segments.last() else {
            return false;
        };
        if current.start < previous.end {
            return false;
        }
        let gap_seconds = current.start - previous.end;
        if gap_seconds > self.merge_policy.max_gap_seconds {
            return false;
        }
        let previous_words = normalize_words(&previous.text);
        let current_words = normalize_words(&current.text);
        if previous_words.is_empty() || current_words.is_empty() {
            return false;
        }
        if previous_words == current_words {
            return true;
        }
        if current_words.len() >= self.merge_policy.redundant_min_words
            && contains_window(&previous_words, &current_words)
        {
            return true;
        }
        // Fuzzy LCW drop stays ASCII-only. Per-character CJK units make the
        // longform_en_zh fixture (repeated Mandarin blocks plus a new tail)
        // look like an 80% re-read and would delete the continuation.
        if !mostly_ascii_units(&previous_words) || !mostly_ascii_units(&current_words) {
            return false;
        }
        let overlap = longest_common_window_len(&previous_words, &current_words);
        let min_len = previous_words.len().min(current_words.len());
        min_len >= self.merge_policy.redundant_min_words
            && overlap as f32 / min_len as f32 >= self.merge_policy.redundant_overlap_ratio
    }

    /// Stitch a wordless cross-slice re-read by a tail-anchored suffix/prefix
    /// match. Seq2seq families such as Qwen emit one segment spanning the
    /// whole slice and no word timestamps, so the time-domain trim above cannot
    /// see the 0.5–2s energy overlap.
    ///
    /// Thresholds: n ≥ 2 when the windows time-overlap, n ≥ 3 on a mere
    /// abutment. A one-unit match is allowed only for a single non-numeral
    /// CJK char that is not a doubled sentence-initial (「谢谢」), and only
    /// when the segment time-overlap covers that char's estimated duration
    /// -- so 「我 我」 / 「吃饭」+「饭，」 still stitch, while 「八」/「谢」
    /// and English head-words do not (interpolated tile times stay refused).
    /// A one-unit non-CJK match is a further candidate when both segments
    /// carry acoustic word timestamps: the straddling-word shape (the cut
    /// lands mid-word and both slices decode it). It stands only when the
    /// full acoustic vet passes (regap, past-previous-end, seam clamp). The
    /// same candidate also applies when the match sits one or two current
    /// units past leading re-read fragments; then the vet must also place
    /// every fragment word in the committed region.
    /// Matches are always a suffix of `previous`; an interior window is
    /// never searched.
    ///
    /// The stitch rewrites text only. `previous.end` is left alone so a
    /// later slice cannot swallow the earlier segment's boundary.
    fn try_stitch_seam_overlap(&mut self, current: &mut Segment) -> bool {
        let Some(previous) = self.segments.last_mut() else {
            return false;
        };
        let time_overlap = previous.end - current.start;
        let time_overlaps = time_overlap > 1.0e-3;
        let gap_seconds = current.start - previous.end;
        if !time_overlaps && gap_seconds > self.merge_policy.max_gap_seconds {
            return false;
        }
        let min_units = if time_overlaps { 2 } else { 3 };
        let stitched = apply_suffix_prefix_stitch(
            previous,
            current,
            min_units,
            time_overlap,
            SEGMENT_STITCH_MAX_REGAP_SECONDS,
            self.approximate_word_timestamps,
        );
        if stitched && self.approximate_word_timestamps {
            // A prior seam may already have advanced these estimates beyond
            // the original audio window. Preserve that committed boundary
            // when this same segment is stitched to yet another slice.
            let previous_start = previous
                .words
                .first()
                .map_or(previous.start, |word| word.start.max(previous.start))
                .min(previous.end);
            previous.words = crate::subtitle::cues::interpolate_word_timestamps(
                &previous.text,
                previous_start,
                previous.end,
            );
            // Keep both original segment windows for subsequent alignment.
            // Only the approximate remainder starts after committed speech.
            let remainder_start = current.start.max(previous.end).min(current.end);
            current.words = crate::subtitle::cues::interpolate_word_timestamps(
                &current.text,
                remainder_start,
                current.end,
            );
        }
        stitched
    }

    /// Drop the new slice's head word when it is a B-side seam phantom: a
    /// decode artifact re-reading audio the previous slice already committed
    /// as a *different* token. The suffix-prefix stitch cannot remove it
    /// (the texts share no overlap) and the midpoint trim cannot remove it
    /// (the stretched window's majority sits past the boundary), so without
    /// this pass both the committed word and the phantom survive.
    ///
    /// Three signals must agree, or the word stays. The committed word is
    /// the one the cut landed on -- clamped at the previous segment end: a
    /// word that finished clearly before the cut made room for new speech.
    /// Both words carry confidence, with the committed word decoded with
    /// certainty and the re-read weakly. And the re-read window is stretched
    /// wide, padding over the silence the cut left behind. Same-token heads
    /// belong to the suffix-prefix stitch family and the midpoint trim, and
    /// non-Latin heads keep the unit rules, so both stay out of scope.
    /// Returns `true` when the phantom head word was dropped (the caller
    /// counts the merge and drops the segment when it was left empty).
    fn drop_seam_phantom_head(&self, current: &mut Segment) -> bool {
        if self.approximate_word_timestamps {
            // Interpolated tiles are not acoustic: width is a function of
            // the tile, so the stretch test has no meaning for them.
            return false;
        }
        let Some(previous) = self.segments.last() else {
            return false;
        };
        let (Some(committed), Some(phantom)) = (previous.words.last(), current.words.first())
        else {
            return false;
        };
        // The re-read must reach back over the committed word's audio.
        if phantom.start >= committed.end {
            return false;
        }
        if previous.end - committed.end > SEGMENT_STITCH_SEAM_CLAMP_TOLERANCE_SECONDS {
            return false;
        }
        if normalize_words(&committed.word) == normalize_words(&phantom.word)
            || !phantom.word.chars().any(|ch| ch.is_ascii_alphabetic())
        {
            return false;
        }
        let (Some(committed_confidence), Some(phantom_confidence)) =
            (committed.confidence, phantom.confidence)
        else {
            return false;
        };
        if committed_confidence < SEAM_PHANTOM_COMMITTED_MIN_CONFIDENCE
            || phantom_confidence >= SEAM_PHANTOM_HEAD_MAX_CONFIDENCE
        {
            return false;
        }
        if phantom.end - phantom.start < SEAM_PHANTOM_HEAD_MIN_WIDTH_SECONDS {
            return false;
        }
        let chars: Vec<char> = current.text.chars().collect();
        let new_text = match leading_word_char_offset(&chars, &current.words, 1) {
            Some(offset) => chars[offset..]
                .iter()
                .collect::<String>()
                .trim()
                .to_string(),
            // Words did not align to the text (unexpected): rebuild from the
            // kept tokens rather than mis-slice the string.
            None => crate::transcript_text::join_segment_texts(
                current.words[1..].iter().map(|word| word.word.as_str()),
            ),
        };
        current.words.drain(0..1);
        if let Some(first) = current.words.first() {
            current.start = first.start;
        }
        current.text = new_text;
        // A phantom head was dropped. The remainder (if any) is genuine
        // continuation that still runs the redundancy check before it is
        // pushed.
        true
    }
}

fn speakers_conflict(previous: &Segment, current: &Segment) -> bool {
    match (&previous.speaker, &current.speaker) {
        (Some(left), Some(right)) => left != right,
        _ => false,
    }
}

/// Rewrite seam text when a tail-anchored suffix/prefix overlap exists.
/// Returns `true` when `current` was trimmed (caller drops it if empty).
/// Never extends `previous.end`.
fn apply_suffix_prefix_stitch(
    previous: &mut Segment,
    current: &mut Segment,
    min_units: usize,
    time_overlap_seconds: f32,
    max_seam_regap_seconds: f32,
    approximate_word_timestamps: bool,
) -> bool {
    let Some(overlap) = suffix_prefix_overlap(
        &previous.text,
        &current.text,
        min_units,
        time_overlap_seconds,
        !approximate_word_timestamps,
    ) else {
        return false;
    };
    let consume_end = extend_consumed_current_end(&current.text, overlap.curr_end);
    let prev_prefix = char_prefix(&previous.text, overlap.prev_start);
    let consumed = current
        .text
        .chars()
        .take(consume_end)
        .skip(overlap.curr_start)
        .collect::<String>();
    let consumed = consumed.trim();
    let remainder = char_suffix(&current.text, consume_end).trim().to_string();
    let previous_words = split_words_at_char(&previous.text, &previous.words, overlap.prev_start);
    let current_words = split_words_at_char(&current.text, &current.words, consume_end);
    // Reject the seam when both segments carry acoustic (non-interpolated) word
    // timestamps and the matched word-instances sit several seconds apart in
    // the original audio. A true slice-boundary re-read lands within the
    // inter-slice overlap (~0.5s at `SlicingOptions::default`) so the two
    // word-instances are acoustically adjacent. A legitimate repeat (a verse
    // re-sung seconds later, an ABAB pattern in a song, etc.) sits far beyond
    // that gap; stitching would consume real words from both segments and lose
    // audible content. Interpolated families carry synthetic tile times, so
    // their regap is not acoustically meaningful and the guard is skipped.
    //
    // A single-unit seam additionally needs the whole vet: one shared word is
    // textually unredundant, so it is admitted only because BOTH word-instances
    // are placed on the original timeline, the earlier one is the word the cut
    // landed on (clamped at `previous.end` or within its end-estimate
    // jitter), and the later one begins where the straddling audio restarts
    // (at/before `previous.end` within onset jitter). A word that finished
    // well before the cut, followed by a same-text word after it, is a
    // genuine back-to-back repeat and is refused by the clamp check; a word
    // with no usable times of its own is refused outright.
    //
    // A seam that skipped leading re-read fragments additionally needs every
    // fragment word placed in the committed region (midpoint strictly before
    // `previous.end`: exactly the words the midpoint trim drops). A fragment
    // whose audio lies at/past the boundary is new speech -- a fresh phrase
    // that merely happens to be fronted by the previous tail word -- and must
    // survive.
    let single_unit = overlap.units == 1;
    let unit_token = current
        .text
        .chars()
        .skip(overlap.curr_start)
        .take(overlap.curr_end - overlap.curr_start)
        .collect::<String>();
    let non_cjk_single_unit = single_unit && !is_single_cjk_char(&unit_token);
    let mut acoustic_vet_passed = true;
    if !approximate_word_timestamps {
        acoustic_vet_passed = match (
            previous_words
                .as_ref()
                .and_then(|(_, matched)| matched.last()),
            current_words
                .as_ref()
                .and_then(|(matched, _)| matched.get(overlap.skip_curr_units)),
        ) {
            (Some(prev_match_last), Some(curr_match_first)) => {
                let regap_seconds = curr_match_first.start - prev_match_last.end;
                if regap_seconds > max_seam_regap_seconds {
                    return false;
                }
                // A re-read is audio the previous slice already covered, so the
                // current slice's copy of the phrase must sit at/before
                // `previous.end` (inside the inter-slice overlap). A phrase
                // that BEGINS after the previous segment's end is new audio the
                // previous slice never transcribed -- a deliberate repeat/echo
                // landing at the cut -- even when its regap is small enough to
                // pass the check above.
                let past_prev_end = curr_match_first.start - previous.end;
                if past_prev_end > SEGMENT_STITCH_REREAD_MAX_PAST_PREV_END_SECONDS {
                    return false;
                }
                if single_unit
                    && previous.end - prev_match_last.end
                        > SEGMENT_STITCH_SEAM_CLAMP_TOLERANCE_SECONDS
                {
                    return false;
                }
                if overlap.skip_curr_units > 0 {
                    let fragments_committed = current_words
                        .as_ref()
                        .map(|(matched, _)| {
                            !matched.is_empty()
                                && matched
                                    .iter()
                                    .take(overlap.skip_curr_units)
                                    .all(|fragment| {
                                        0.5 * (fragment.start + fragment.end) < previous.end
                                    })
                        })
                        .unwrap_or(false);
                    if !fragments_committed {
                        return false;
                    }
                }
                true
            }
            _ => false,
        };
    }
    if non_cjk_single_unit && !acoustic_vet_passed {
        return false;
    }
    let leftover_words = current_words
        .as_ref()
        .map(|(_, leftover)| leftover.clone())
        .unwrap_or_default();
    if speakers_conflict(previous, current) {
        current.text = remainder;
        current.words = leftover_words;
        return true;
    }
    // The overlap phrase is re-homed onto `previous` below: its text becomes
    // `prev_prefix + consumed`, so `previous` keeps every word it already had
    // -- the words before the overlap AND the overlap's own words. Truncating
    // `previous.words` to the pre-overlap prefix (the historical behavior)
    // deleted the overlap's words here while `current.words = leftover` dropped
    // them on the other side, so a phrase whose speech straddled the slice
    // boundary (the overlap re-read) survived in text but lost every word
    // window. Keeping the earlier segment's native acoustic words leaves exactly
    // one timed copy of the phrase in the timeline.
    let mut completed = format!("{prev_prefix}{consumed}");
    // A period sitting on `previous` after the overlap is a truncated-slice
    // hallucination when current continues with content (no punct right after
    // the overlap). Keep it only when the remainder is empty or current
    // already carried immediately-following seam punct. Conversely, the
    // earlier copy's trailing punct is the same artifact once the re-homed
    // text carries the current slice's seam punct ("too." + "too," ->
    // "too,", never "too,.").
    let consumed_ends_punct = consumed.chars().last().is_some_and(is_seam_punctuation);
    let strip_trailing_punct =
        (consume_end == overlap.curr_end && !remainder.is_empty()) || consumed_ends_punct;
    if !strip_trailing_punct {
        let trailing = previous_trailing_punct(&previous.text, overlap.prev_end);
        if !trailing.is_empty() && !completed.ends_with(&trailing) {
            completed.push_str(&trailing);
        }
    }
    previous.text = completed;
    current.text = remainder;
    current.words = leftover_words;
    true
}

fn map_word_time_to_original(
    word: &WordTimestamp,
    content_offset: f32,
    time_domain: SegmentTimeDomain,
    timeline: &TimelineMap,
    segment_start: f32,
    segment_end: f32,
) -> Option<WordTimestamp> {
    let text = word.word.trim();
    if text.is_empty() || !word.start.is_finite() || !word.end.is_finite() {
        return None;
    }
    let mut start = word.start.max(0.0);
    let mut end = word.end.max(start);
    if time_domain == SegmentTimeDomain::RelativeToSliceContent {
        start += content_offset;
        end += content_offset;
    }
    let original_start = timeline
        .map_processed_to_original_seconds(start)
        .clamp(segment_start, segment_end);
    let original_end = timeline
        .map_processed_to_original_seconds(end)
        .max(original_start)
        .clamp(original_start, segment_end);
    Some(WordTimestamp {
        word: text.to_string(),
        start: original_start,
        end: original_end,
        confidence: word.confidence,
    })
}

/// Trim the part of a mapped segment that lies in the audio region a prior slice
/// already committed (`[.., boundary)` in original-timeline seconds). Returns
/// `true` when the whole segment falls inside that region and should be dropped.
///
/// A word is assigned to whichever side of `boundary` holds the majority of it
/// (midpoint rule), so a word straddling the cut is kept in exactly one slice.
/// Leading committed words are dropped and the segment text is reconstructed
/// from the surviving word span (exact substring of the original text, so CJK
/// and glued punctuation stay intact); a segment left empty is dropped.
fn trim_committed_overlap(segment: &mut Segment, boundary: f32) -> bool {
    // Whole segment already behind the committed frontier: drop it outright.
    // (This is the standalone-orphan shape, e.g. a hallucinated 1-word cue.)
    if segment.end <= boundary {
        return true;
    }
    if segment.words.is_empty() {
        // A wordless slice-spanning segment's midpoint almost never sits inside
        // the 0.5–2s overlap, so a midpoint rule would keep the whole re-read.
        // Leave the text for suffix-prefix stitch instead of dropping or keeping
        // the segment as a unit.
        return false;
    }
    let first_keep = segment
        .words
        .iter()
        .position(|word| 0.5 * (word.start + word.end) >= boundary);
    let Some(first_keep) = first_keep else {
        // Every word's majority sits in the committed region.
        return true;
    };
    if first_keep == 0 {
        return false;
    }
    let chars: Vec<char> = segment.text.chars().collect();
    let new_text = match leading_word_char_offset(&chars, &segment.words, first_keep) {
        Some(offset) => chars[offset..]
            .iter()
            .collect::<String>()
            .trim()
            .to_string(),
        // Words did not align to the text (unexpected): rebuild from the kept
        // word tokens rather than mis-slice the string.
        None => crate::transcript_text::join_segment_texts(
            segment.words[first_keep..]
                .iter()
                .map(|word| word.word.as_str()),
        ),
    };
    segment.words.drain(0..first_keep);
    if let Some(first) = segment.words.first() {
        segment.start = first.start;
    }
    segment.text = new_text;
    segment.text.trim().is_empty()
}

/// Char offset at which `words[first_keep]` begins within `chars`, found by the
/// same greedy whitespace-delimited match the cue splitter uses. Returns `None`
/// if a leading word does not align to the text.
fn leading_word_char_offset(
    chars: &[char],
    words: &[WordTimestamp],
    first_keep: usize,
) -> Option<usize> {
    let mut idx = 0usize;
    for word in &words[..first_keep] {
        while idx < chars.len() && chars[idx].is_whitespace() {
            idx += 1;
        }
        let token: Vec<char> = word.word.trim().chars().collect();
        if token.is_empty() {
            continue;
        }
        if idx + token.len() > chars.len() || chars[idx..idx + token.len()] != token[..] {
            return None;
        }
        idx += token.len();
    }
    while idx < chars.len() && chars[idx].is_whitespace() {
        idx += 1;
    }
    Some(idx)
}

/// Split `words` at the first token whose aligned start is `>= char_index`.
/// `None` when the greedy word/text walk does not align (leave words alone).
fn split_words_at_char(
    text: &str,
    words: &[WordTimestamp],
    char_index: usize,
) -> Option<(Vec<WordTimestamp>, Vec<WordTimestamp>)> {
    if words.is_empty() {
        return Some((Vec::new(), Vec::new()));
    }
    let chars: Vec<char> = text.chars().collect();
    let mut split_at = words.len();
    for index in 0..=words.len() {
        let offset = leading_word_char_offset(&chars, words, index)?;
        if offset >= char_index {
            split_at = index;
            break;
        }
    }
    Some((words[..split_at].to_vec(), words[split_at..].to_vec()))
}

/// Word list for [`TranscriptAssembler::try_drop_redundant_segment`].
/// ASCII keeps the historical fold (`don't` → `dont`, `twenty-one` →
/// `twentyone`, `café` → `caf`). CJK has no ASCII letters, so it falls
/// through to one unit per non-punctuation character.
fn normalize_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .flat_map(|word| {
            let ascii: String = word
                .chars()
                .filter(|ch| ch.is_ascii_alphanumeric())
                .collect::<String>()
                .to_ascii_lowercase();
            if !ascii.is_empty() {
                return vec![ascii];
            }
            word.chars()
                .filter(|ch| !is_seam_punctuation(*ch))
                .map(|ch| ch.to_string())
                .collect::<Vec<_>>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OverlapUnit {
    token: String,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SuffixPrefixOverlap {
    prev_start: usize,
    prev_end: usize,
    curr_start: usize,
    curr_end: usize,
    /// Number of matched overlap units. A one-unit seam is textually
    /// unredundant, so its callers require the acoustic vet to have actually
    /// placed both word-instances on the timeline before it may stand.
    units: usize,
    /// Leading `current` units consumed beyond the match. A straddling word
    /// is sometimes fronted in the re-decode by a fragment of the earlier
    /// word(s) ("...blacked out." / "I black out, but..."); the fragments are
    /// re-read audio the previous slice already committed, so the one-word
    /// match sits one or two units in. The vet must prove every fragment word
    /// is committed-region audio before the skip may stand.
    skip_curr_units: usize,
}

fn is_seam_punctuation(ch: char) -> bool {
    matches!(
        ch,
        ',' | '.' | '!' | '?' | ';' | ':' | '，' | '。' | '！' | '？' | '；' | '：' | '、'
    )
}

fn is_ascii_word_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '\'' || ch == '-'
}

fn is_cjk_numeral(ch: char) -> bool {
    matches!(
        ch,
        '零' | '一'
            | '二'
            | '三'
            | '四'
            | '五'
            | '六'
            | '七'
            | '八'
            | '九'
            | '十'
            | '百'
            | '千'
            | '万'
    )
}

fn is_numeric_unit(token: &str) -> bool {
    !token.is_empty()
        && token
            .chars()
            .all(|ch| ch.is_ascii_digit() || is_cjk_numeral(ch))
}

fn is_single_cjk_char(token: &str) -> bool {
    let mut chars = token.chars();
    match (chars.next(), chars.next()) {
        (Some(ch), None) => !ch.is_ascii() && !ch.is_ascii_whitespace(),
        _ => false,
    }
}

fn overlap_units(text: &str) -> Vec<OverlapUnit> {
    let chars: Vec<char> = text.chars().collect();
    let mut units = Vec::new();
    let mut index = 0usize;
    while index < chars.len() {
        let ch = chars[index];
        if ch.is_whitespace() || is_seam_punctuation(ch) {
            index += 1;
            continue;
        }
        if is_ascii_word_char(ch) {
            let start = index;
            let mut token = String::new();
            while index < chars.len() && is_ascii_word_char(chars[index]) {
                token.push(chars[index].to_ascii_lowercase());
                index += 1;
            }
            units.push(OverlapUnit {
                token,
                start,
                end: index,
            });
            continue;
        }
        units.push(OverlapUnit {
            token: ch.to_string(),
            start: index,
            end: index + 1,
        });
        index += 1;
    }
    units
}

fn char_prefix(text: &str, char_count: usize) -> String {
    text.chars().take(char_count).collect()
}

fn char_suffix(text: &str, char_skip: usize) -> String {
    text.chars().skip(char_skip).collect()
}

const CJK_CHAR_SECONDS: f32 = 0.18;

/// Longest *suffix* of `previous` that is a prefix of `current`. Interior
/// matches are not considered: a re-read can only replay the previous tail.
/// A single non-CJK unit is only a *candidate* here; on acoustic families
/// the caller must still pass its full acoustic vet, and on interpolated
/// tile times it is refused outright.
///
/// When no head-anchored match stands on acoustic families, one further
/// candidate is admitted: the previous tail word equal to the current's
/// second or third unit, i.e. the match shifted one or two units past
/// leading re-read fragments ("...blacked out." / "I black out, but...").
/// The fragments are partial re-decodes of the last word(s) inside the
/// inter-slice overlap and would otherwise block the one-word seam outright.
/// Like the head-anchored candidate it stays a candidate:
/// `apply_suffix_prefix_stitch` must place the matched word on the timeline,
/// pass the regap / past / seam-clamp vet, AND prove every skipped fragment
/// word is committed-region audio (a word the midpoint trim drops) before the
/// seam may stand.
fn suffix_prefix_overlap(
    previous: &str,
    current: &str,
    min_units: usize,
    time_overlap_seconds: f32,
    acoustic: bool,
) -> Option<SuffixPrefixOverlap> {
    let previous_units = overlap_units(previous);
    let current_units = overlap_units(current);
    if previous_units.is_empty() || current_units.is_empty() {
        return None;
    }
    let max_n = previous_units.len().min(current_units.len());
    if max_n == 0 {
        return None;
    }
    for n in (1..=max_n).rev() {
        let previous_suffix = &previous_units[previous_units.len() - n..];
        let current_prefix = &current_units[..n];
        if !previous_suffix
            .iter()
            .zip(current_prefix)
            .all(|(left, right)| left.token == right.token)
        {
            continue;
        }
        if !accept_overlap_n(
            n,
            min_units,
            previous_suffix,
            &current_units,
            time_overlap_seconds,
            acoustic,
        ) {
            continue;
        }
        return Some(SuffixPrefixOverlap {
            prev_start: previous_suffix[0].start,
            prev_end: previous_suffix[n - 1].end,
            curr_start: current_prefix[0].start,
            curr_end: current_prefix[n - 1].end,
            units: n,
            skip_curr_units: 0,
        });
    }
    if acoustic && current_units.len() >= 2 {
        let previous_tail = &previous_units[previous_units.len() - 1];
        let mut skip = 0usize;
        for (index, shifted) in current_units.iter().take(3).enumerate() {
            if index >= 1 && previous_tail.token == shifted.token {
                skip = index;
                break;
            }
        }
        if skip > 0
            && !is_numeric_unit(&previous_tail.token)
            && !is_single_cjk_char(&previous_tail.token)
        {
            let shifted = &current_units[skip];
            return Some(SuffixPrefixOverlap {
                prev_start: previous_tail.start,
                prev_end: previous_tail.end,
                curr_start: shifted.start,
                curr_end: shifted.end,
                units: 1,
                skip_curr_units: skip,
            });
        }
    }
    None
}

fn accept_overlap_n(
    n: usize,
    min_units: usize,
    overlap: &[OverlapUnit],
    current_units: &[OverlapUnit],
    time_overlap_seconds: f32,
    acoustic: bool,
) -> bool {
    if overlap.iter().all(|unit| is_numeric_unit(&unit.token)) {
        return false;
    }
    if n >= min_units {
        return true;
    }
    // Single-char CJK re-read (「我 我」, 「吃饭」/「饭，」): require the
    // windows to actually overlap by at least that char's spoken duration.
    // A doubled sentence-initial (「谢谢」) is a new phrase, not a re-read.
    if n == 1
        && time_overlap_seconds > 1.0e-3
        && time_overlap_seconds + 1.0e-3 >= CJK_CHAR_SECONDS
        && overlap.len() == 1
        && is_single_cjk_char(&overlap[0].token)
        && !(current_units.len() >= 2 && current_units[0].token == current_units[1].token)
    {
        return true;
    }
    // A single non-CJK word is admitted as a re-read *candidate* only for
    // families whose word timestamps are acoustic: the straddling-word shape
    // (the cut lands mid-word, both slices decode it, the earlier copy is
    // clamped at the boundary and the later copy restarts within onset
    // jitter). One shared word carries no textual redundancy, so on
    // interpolated tile times it stays refused (the historical English
    // head-word false positive); on acoustic families
    // `apply_suffix_prefix_stitch` then requires the full acoustic vet
    // (regap, past-previous-end, and the seam clamp) to pass, or the seam
    // is dropped there. The CJK single-char rule above is untouched.
    acoustic && n == 1 && !is_single_cjk_char(&overlap[0].token)
}

fn extend_consumed_current_end(current: &str, overlap_end: usize) -> usize {
    let chars: Vec<char> = current.chars().collect();
    if overlap_end >= chars.len() {
        return chars.len();
    }
    let mut end = overlap_end;
    while end < chars.len() && chars[end].is_whitespace() {
        end += 1;
    }
    while end < chars.len() && is_seam_punctuation(chars[end]) {
        end += 1;
    }
    end
}

/// Punctuation that already sat on `previous` after the overlap units.
fn previous_trailing_punct(previous: &str, prev_end: usize) -> String {
    let chars: Vec<char> = previous.chars().collect();
    let mut index = prev_end;
    while index < chars.len() && chars[index].is_whitespace() {
        index += 1;
    }
    let mut trailing = String::new();
    while index < chars.len() && is_seam_punctuation(chars[index]) {
        trailing.push(chars[index]);
        index += 1;
    }
    trailing
}

fn mostly_ascii_units(words: &[String]) -> bool {
    if words.is_empty() {
        return false;
    }
    let ascii = words.iter().filter(|word| word.is_ascii()).count();
    ascii * 2 >= words.len()
}

fn contains_window(haystack: &[String], needle: &[String]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return false;
    }
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

/// Longest common *contiguous* window. DP is O(n·m) / O(min(n, m)) space.
fn longest_common_window_len(left: &[String], right: &[String]) -> usize {
    if left.is_empty() || right.is_empty() {
        return 0;
    }
    let (short, long) = if left.len() <= right.len() {
        (left, right)
    } else {
        (right, left)
    };
    let mut prev = vec![0usize; short.len() + 1];
    let mut best = 0usize;
    for long_token in long {
        let mut curr = vec![0usize; short.len() + 1];
        for (index, short_token) in short.iter().enumerate() {
            if short_token == long_token {
                curr[index + 1] = prev[index] + 1;
                best = best.max(curr[index + 1]);
            }
        }
        prev = curr;
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::longform::{AudioSlice, AudioSliceKind, TimelineAnchor};

    fn slice(start: usize, end: usize) -> AudioSlice {
        AudioSlice {
            index: 0,
            kind: AudioSliceKind::Fixed,
            start_sample: start,
            end_sample: end,
            content_start_sample: start,
            content_end_sample: end,
        }
    }

    #[test]
    fn assembler_maps_relative_segment_times() {
        let timeline = TimelineMap::from_anchors(vec![
            TimelineAnchor {
                processed_seconds: 0.0,
                original_seconds: 0.0,
            },
            TimelineAnchor {
                processed_seconds: 10.0,
                original_seconds: 10.0,
            },
        ]);
        let mut assembler = TranscriptAssembler::new(timeline, SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000, 32_000),
            text: "hello".to_string(),
            segments: vec![Segment {
                start: 0.0,
                end: 0.5,
                text: "hello".to_string(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: vec![WordTimestamp {
                    word: "hello".to_string(),
                    start: 0.1,
                    end: 0.4,
                    confidence: None,
                }],
            }],
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 1);
        assert!(transcription.segments[0].start >= 1.0);
        assert_eq!(transcription.segments[0].words.len(), 1);
        assert_eq!(transcription.segments[0].words[0].word, "hello");
        assert!(transcription.segments[0].words[0].start >= 1.1);
        assert!(transcription.segments[0].words[0].end <= 1.4);
    }

    #[test]
    fn assembler_keeps_exact_scope_provenance_across_slice_overlap() {
        fn labeled_segment(start: f32, end: f32, text: &str) -> Segment {
            Segment {
                start,
                end,
                text: text.to_string(),
                speaker: Some("SPEAKER_01".to_string()),
                speaker_label: Some("SPEAKER_01".to_string()),
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            }
        }

        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result_with_speaker_scope(
            SliceTranscript {
                slice: slice(0, 480_000),
                text: "first owner".to_string(),
                segments: vec![labeled_segment(29.6, 29.9, "first owner")],
                time_domain: SegmentTimeDomain::RelativeToSliceContent,
            },
            0,
        );
        assembler.push_slice_result_with_speaker_scope(
            SliceTranscript {
                slice: slice(472_000, 496_000),
                text: "second owner".to_string(),
                segments: vec![labeled_segment(0.6, 0.9, "second owner")],
                time_domain: SegmentTimeDomain::RelativeToSliceContent,
            },
            1,
        );

        let (transcription, _, scopes) = assembler.into_parts_with_speaker_scopes();
        assert_eq!(transcription.segments.len(), 2);
        assert!((transcription.segments[0].start - 29.6).abs() < 1e-4);
        assert!((transcription.segments[1].start - 30.1).abs() < 1e-4);
        assert_eq!(scopes, vec![Some(0), Some(1)]);
    }

    #[test]
    fn assembler_drops_redundant_overlap() {
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "hello world from openasr".to_string(),
            segments: vec![Segment {
                start: 0.0,
                end: 1.0,
                text: "hello world from openasr".to_string(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            }],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(15_000, 31_000),
            text: "hello world from openasr".to_string(),
            segments: vec![Segment {
                start: 1.05,
                end: 2.0,
                text: "hello world from openasr".to_string(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            }],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 1);
    }

    #[test]
    fn assembler_keeps_adjacent_segments_distinct() {
        // Adjacent, non-overlapping segments are no longer coalesced into a
        // paragraph blob: the post-ASR cue re-segmentation pass owns subtitle
        // granularity. Each segment survives with its own words and timing, and
        // the joined transcript text still reads as one paragraph.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 32_000),
            text: "hello world".to_string(),
            segments: vec![
                Segment {
                    start: 0.0,
                    end: 1.0,
                    text: "hello world".to_string(),
                    speaker: None,
                    speaker_label: None,
                    speaker_person_id: None,
                    speaker_snapshot_label: None,
                    words: vec![
                        WordTimestamp {
                            word: "hello".to_string(),
                            start: 0.1,
                            end: 0.4,
                            confidence: None,
                        },
                        WordTimestamp {
                            word: "world".to_string(),
                            start: 0.5,
                            end: 0.9,
                            confidence: None,
                        },
                    ],
                },
                Segment {
                    start: 1.0,
                    end: 2.0,
                    text: "from openasr".to_string(),
                    speaker: None,
                    speaker_label: None,
                    speaker_person_id: None,
                    speaker_snapshot_label: None,
                    words: vec![
                        WordTimestamp {
                            word: "from".to_string(),
                            start: 1.1,
                            end: 1.4,
                            confidence: None,
                        },
                        WordTimestamp {
                            word: "openasr".to_string(),
                            start: 1.5,
                            end: 1.9,
                            confidence: None,
                        },
                    ],
                },
            ],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(
            transcription.segments.len(),
            2,
            "adjacent segments must stay distinct"
        );
        assert_eq!(transcription.segments[0].text, "hello world");
        assert_eq!(transcription.segments[1].text, "from openasr");
        assert_eq!(transcription.text, "hello world from openasr");
    }

    #[test]
    fn assembler_preserves_slice_boundaries_for_synthesized_slice_segments() {
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "first chunk".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000, 32_000),
            text: "second chunk".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.segments[0].start, 0.0);
        assert_eq!(transcription.segments[0].end, 1.0);
        assert_eq!(transcription.segments[1].start, 1.0);
        assert_eq!(transcription.segments[1].end, 2.0);
    }

    fn word(text: &str, start: f32, end: f32) -> WordTimestamp {
        WordTimestamp {
            word: text.to_string(),
            start,
            end,
            confidence: None,
        }
    }

    fn word_conf(text: &str, start: f32, end: f32, confidence: f32) -> WordTimestamp {
        WordTimestamp {
            word: text.to_string(),
            start,
            end,
            confidence: Some(confidence),
        }
    }

    fn absolute_segment(text: &str, start: f32, end: f32, words: Vec<WordTimestamp>) -> Segment {
        Segment {
            start,
            end,
            text: text.to_string(),
            speaker: None,
            speaker_label: None,
            speaker_person_id: None,
            speaker_snapshot_label: None,
            words,
        }
    }

    /// DecodeInvariant interpolation: Han+trailing punct (「我。」) and ASCII
    /// runs with attached punct, linearly spaced — same grouping as
    /// `seq2seq_word_timestamps_from_generated_tokens`.
    fn interpolated_words(text: &str, start: f32, end: f32) -> Vec<WordTimestamp> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        for ch in text.chars() {
            if ch.is_whitespace() {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                continue;
            }
            let starts_han = !ch.is_ascii() && !is_seam_punctuation(ch);
            let last_is_han = current
                .chars()
                .next_back()
                .is_some_and(|last| !last.is_ascii() && !is_seam_punctuation(last));
            if (starts_han && !current.is_empty()) || (ch.is_ascii_alphanumeric() && last_is_han) {
                tokens.push(std::mem::take(&mut current));
            }
            current.push(ch);
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        if tokens.is_empty() {
            return Vec::new();
        }
        let step = (end - start).max(1.0e-3) / tokens.len() as f32;
        tokens
            .iter()
            .enumerate()
            .map(|(index, token)| {
                let word_start = start + step * index as f32;
                word(token, word_start, word_start + step)
            })
            .collect()
    }

    #[test]
    fn assembler_time_trims_hallucinated_leading_overlap_word() {
        // Field defect shape: a forced cut at 1.0s widens the overlap so the next
        // slice re-reads the straddling audio. A weak model hallucinates the head
        // of its monolithic segment ("If,") from that already-committed region;
        // its text does not match anything in slice 1, so text-equality dedup is
        // blind to it. The time trim drops it by timestamp and keeps the rest.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "hello world".to_string(),
            segments: vec![absolute_segment(
                "hello world",
                0.1,
                0.9,
                vec![word("hello", 0.1, 0.4), word("world", 0.5, 0.9)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(12_000, 32_000),
            text: "If, mad indeed would".to_string(),
            segments: vec![absolute_segment(
                "If, mad indeed would",
                0.80,
                1.90,
                vec![
                    word("If,", 0.80, 0.95),
                    word("mad", 1.10, 1.30),
                    word("indeed", 1.35, 1.60),
                    word("would", 1.65, 1.90),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.segments[1].text, "mad indeed would");
        assert_eq!(transcription.segments[1].words.len(), 3);
        assert_eq!(transcription.segments[1].words[0].word, "mad");
        assert!((transcription.segments[1].start - 1.10).abs() < 1e-4);
        assert_eq!(transcription.text, "hello world mad indeed would");
        // The trimmed word is not a dropped segment, so no whole-segment drop.
        assert_eq!(stats.duplicate_merge_count, 0);
    }

    #[test]
    fn assembler_drops_standalone_orphan_inside_committed_span() {
        // The "If," rendered as its own leading cue: the whole segment sits behind
        // the committed frontier and is dropped outright.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "hello world".to_string(),
            segments: vec![absolute_segment(
                "hello world",
                0.1,
                0.9,
                vec![word("hello", 0.1, 0.4), word("world", 0.5, 0.9)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(12_000, 32_000),
            text: "If,".to_string(),
            segments: vec![
                absolute_segment("If,", 0.80, 0.95, vec![word("If,", 0.80, 0.95)]),
                absolute_segment(
                    "mad indeed",
                    1.10,
                    1.60,
                    vec![word("mad", 1.10, 1.30), word("indeed", 1.35, 1.60)],
                ),
            ],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.segments[0].text, "hello world");
        assert_eq!(transcription.segments[1].text, "mad indeed");
        assert_eq!(transcription.text, "hello world mad indeed");
        assert_eq!(stats.duplicate_merge_count, 1);
    }

    #[test]
    fn assembler_keeps_straddling_word_with_majority_after_boundary() {
        // A word straddling the 1.0s cut whose midpoint (1.025s) is past the
        // boundary belongs to the new slice and is kept whole.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "hello world".to_string(),
            segments: vec![absolute_segment(
                "hello world",
                0.1,
                0.9,
                vec![word("hello", 0.1, 0.4), word("world", 0.5, 0.9)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(12_000, 32_000),
            text: "straddle tail".to_string(),
            segments: vec![absolute_segment(
                "straddle tail",
                0.85,
                1.60,
                vec![word("straddle", 0.85, 1.20), word("tail", 1.30, 1.60)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.segments[1].text, "straddle tail");
        assert_eq!(transcription.segments[1].words.len(), 2);
    }

    #[test]
    fn assembler_trims_straddling_word_with_majority_before_boundary() {
        // Same cut, but the leading word's midpoint (0.925s) is before the
        // boundary, so the word belongs to the prior slice and is trimmed.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "hello world".to_string(),
            segments: vec![absolute_segment(
                "hello world",
                0.1,
                0.9,
                vec![word("hello", 0.1, 0.4), word("world", 0.5, 0.9)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(12_000, 32_000),
            text: "straddle tail".to_string(),
            segments: vec![absolute_segment(
                "straddle tail",
                0.75,
                1.60,
                vec![word("straddle", 0.75, 1.10), word("tail", 1.30, 1.60)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.segments[1].text, "tail");
        assert_eq!(transcription.segments[1].words.len(), 1);
        assert_eq!(transcription.segments[1].words[0].word, "tail");
    }

    #[test]
    fn assembler_does_not_trim_without_overlap() {
        // Abutting, non-overlapping slices: the second slice's words all sit past
        // the committed frontier, so nothing is trimmed.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000),
            text: "hello world".to_string(),
            segments: vec![absolute_segment(
                "hello world",
                0.1,
                0.9,
                vec![word("hello", 0.1, 0.4), word("world", 0.5, 0.9)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000, 32_000),
            text: "next words here".to_string(),
            segments: vec![absolute_segment(
                "next words here",
                1.20,
                1.90,
                vec![
                    word("next", 1.20, 1.40),
                    word("words", 1.50, 1.70),
                    word("here", 1.75, 1.90),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.segments[1].text, "next words here");
        assert_eq!(transcription.segments[1].words.len(), 3);
        assert_eq!(stats.duplicate_merge_count, 0);
    }

    #[test]
    fn assembler_maps_packed_slice_segments_back_to_original_timeline() {
        let timeline = TimelineMap::from_anchors(vec![
            TimelineAnchor {
                processed_seconds: 0.0,
                original_seconds: 0.0,
            },
            TimelineAnchor {
                processed_seconds: 1.0,
                original_seconds: 1.0,
            },
            TimelineAnchor {
                processed_seconds: 1.2,
                original_seconds: 12.0,
            },
            TimelineAnchor {
                processed_seconds: 2.2,
                original_seconds: 13.0,
            },
        ]);
        let mut assembler = TranscriptAssembler::new(timeline, SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: AudioSlice {
                index: 0,
                kind: AudioSliceKind::Energy,
                start_sample: 0,
                end_sample: 35_200,
                content_start_sample: 0,
                content_end_sample: 35_200,
            },
            text: "first second".to_string(),
            segments: vec![
                Segment {
                    start: 0.1,
                    end: 0.9,
                    text: "first".to_string(),
                    speaker: None,
                    speaker_label: None,
                    speaker_person_id: None,
                    speaker_snapshot_label: None,
                    words: Vec::new(),
                },
                Segment {
                    start: 1.3,
                    end: 2.0,
                    text: "second".to_string(),
                    speaker: None,
                    speaker_label: None,
                    speaker_person_id: None,
                    speaker_snapshot_label: None,
                    words: Vec::new(),
                },
            ],
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 2);
        assert!(transcription.segments[0].end <= 1.0);
        assert!(transcription.segments[1].start >= 12.0);
        assert!(transcription.segments[1].end <= 13.0);
    }

    fn energy_slice(index: usize, start: usize, end: usize) -> AudioSlice {
        AudioSlice {
            index,
            kind: AudioSliceKind::Energy,
            start_sample: start,
            end_sample: end,
            content_start_sample: start,
            content_end_sample: end,
        }
    }

    fn assemble_three_wordless(slices: [&str; 3]) -> String {
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        let bounds = [
            (0, 16_000 * 25 + 8_000),
            (16_000 * 25, 16_000 * 50 + 8_000),
            (16_000 * 50, 16_000 * 69 + 9_600),
        ];
        for (index, (text, (start, end))) in slices.iter().zip(bounds).enumerate() {
            assembler.push_slice_result(SliceTranscript {
                slice: energy_slice(index, start, end),
                text: (*text).to_string(),
                segments: Vec::new(),
                time_domain: SegmentTimeDomain::RelativeToSliceContent,
            });
        }
        assembler.into_transcription().text
    }

    fn assemble_wordless_overlap(previous: &str, current: &str) -> Transcription {
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 0, 16_000 * 25 + 8_000),
            text: previous.to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(1, 16_000 * 25, 16_000 * 50 + 8_000),
            text: current.to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.into_transcription()
    }

    #[test]
    fn assembler_stitches_qwen_e2e_abutting_wordless_slice_segments() {
        let previous = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country. 今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去一家新开的川菜馆吃饭。听说那里的麻婆豆腐特别正宗。周末的时候，我。";
        let current = "的时候，我通常会读书或者看一部电影，放松一下。今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去一家新开的川菜馆吃饭。听说那里的麻婆豆腐特别正宗。吃饭。听说那里的麻婆豆腐特别正宗。周末的时候，我通常会读书或者看一部电影，放松一下。And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";
        let (transcription, stats) = {
            let mut assembler =
                TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
            assembler.push_slice_result(SliceTranscript {
                slice: energy_slice(0, 0, 16_000 * 25 + 8_000),
                text: previous.to_string(),
                segments: Vec::new(),
                time_domain: SegmentTimeDomain::RelativeToSliceContent,
            });
            assembler.push_slice_result(SliceTranscript {
                slice: energy_slice(1, 16_000 * 25, 16_000 * 69 + 9_600),
                text: current.to_string(),
                segments: Vec::new(),
                time_domain: SegmentTimeDomain::RelativeToSliceContent,
            });
            assembler.into_parts()
        };
        assert!(
            stats.duplicate_merge_count >= 1,
            "abutting 的时候我 (n=4) must stitch, got {stats:?} text={:?}",
            transcription.text
        );
        assert!(
            transcription.segments[0].text.ends_with("周末的时候，我"),
            "previous must keep its own window text, got {:#?}",
            transcription.segments[0]
        );
        assert!(
            !transcription.segments[0].text.contains("通常会"),
            "current body must not move onto previous, got {:#?}",
            transcription.segments[0]
        );
        assert!(
            transcription.segments[1].text.starts_with("通常会"),
            "remainder must start after the overlap units, got {:#?}",
            transcription.segments[1]
        );
        assert!(
            !transcription.segments[1].text.starts_with("的时候"),
            "re-read prefix must not survive on the later slice, got {:#?}",
            transcription.segments
        );
        assert!(
            (transcription.segments[0].end - 25.5).abs() < 1e-3,
            "previous.end must stay on the first slice window, got {:#?}",
            transcription.segments
        );
        assert!(
            (transcription.segments[1].start - 25.0).abs() < 1e-3,
            "remainder must keep the later slice window, got {:#?}",
            transcription.segments
        );
    }

    #[test]
    fn stitched_estimates_fill_the_original_windows_without_retiming_native_words() {
        let previous = "周末的时候，我。";
        let current = "的时候，我通常会读书。";
        for approximate in [false, true] {
            let mut assembler =
                TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default())
                    .with_approximate_word_timestamps(approximate);
            assembler.push_slice_result(SliceTranscript {
                slice: energy_slice(0, 0, 16_000 * 25 + 8_000),
                text: previous.into(),
                segments: vec![absolute_segment(
                    previous,
                    0.0,
                    25.5,
                    interpolated_words(previous, 0.0, 25.5),
                )],
                time_domain: SegmentTimeDomain::RelativeToSliceContent,
            });
            let original_words = interpolated_words(current, 0.0, 20.0);
            assembler.push_slice_result(SliceTranscript {
                slice: energy_slice(1, 16_000 * 25, 16_000 * 45),
                text: current.into(),
                segments: vec![absolute_segment(current, 0.0, 20.0, original_words.clone())],
                time_domain: SegmentTimeDomain::RelativeToSliceContent,
            });
            let out = assembler.into_transcription();
            assert_eq!(out.segments.len(), 2);
            let left = &out.segments[0];
            let right = &out.segments[1];
            assert_eq!(left.end, 25.5);
            assert_eq!(right.start, 25.0);
            assert_eq!(right.end, 45.0);
            assert_eq!(right.text, "通常会读书。");
            if approximate {
                assert_eq!(left.words.last().unwrap().end, left.end);
                assert_eq!(right.words.first().unwrap().start, left.end);
                assert_eq!(right.words.last().unwrap().end, right.end);
                assert_eq!(
                    left.words
                        .iter()
                        .map(|word| word.word.as_str())
                        .collect::<String>(),
                    left.text
                );
                assert!(right.words.iter().all(|word| word.confidence.is_none()));
                // A later stitch must not rewind a start already repaired at
                // the preceding seam, even when the original windows overlap.
                let mut next = absolute_segment("会读书。然后散步。", 44.5, 60.0, Vec::new());
                let mut resumed = TranscriptAssembler::new(
                    TimelineMap::identity(),
                    SegmentMergePolicy::default(),
                )
                .with_approximate_word_timestamps(true);
                resumed.segments = out.segments.clone();
                assert!(resumed.try_stitch_seam_overlap(&mut next));
                assert_eq!(resumed.segments[1].words[0].start, 25.5);
            } else {
                let native_first = original_words
                    .iter()
                    .find(|word| word.word == "通")
                    .unwrap();
                assert_eq!(
                    right.words.first().unwrap().start,
                    25.0 + native_first.start
                );
            }
        }
    }

    #[test]
    fn assembler_stitches_decode_invariant_worded_qwen_slice_seam() {
        // Production CLI path: DecodeInvariant words are forced on for cue
        // splitting, so qwen emits one slice-spanning segment with interpolated
        // timestamps. Midpoint trim would eat "的时候" and skip stitch.
        let previous = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country. 今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去一家新开的川菜馆吃饭。听说那里的麻婆豆腐特别正宗。周末的时候，我。";
        let current = "的时候，我通常会读书或者看一部电影，放松一下。今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去一家新开的川菜馆吃饭。听说那里的麻婆豆腐特别正宗。吃饭。听说那里的麻婆豆腐特别正宗。周末的时候，我通常会读书或者看一部电影，放松一下。And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 0, 16_000 * 25 + 8_000),
            text: previous.to_string(),
            segments: vec![absolute_segment(
                previous,
                0.0,
                25.5,
                interpolated_words(previous, 0.0, 25.5),
            )],
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(1, 16_000 * 25, 16_000 * 69 + 9_600),
            text: current.to_string(),
            segments: vec![absolute_segment(
                current,
                0.0,
                44.1,
                interpolated_words(current, 0.0, 44.1),
            )],
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let (transcription, stats) = assembler.into_parts();
        assert!(
            stats.duplicate_merge_count >= 1,
            "worded DecodeInvariant seam must stitch, got {stats:?} text={:?}",
            transcription.text
        );
        assert!(
            transcription.segments[0].text.ends_with("周末的时候，我"),
            "previous must keep its own window text, got {:#?}",
            transcription.segments[0]
        );
        assert!(
            !transcription.segments[0].text.contains("通常会"),
            "current body must not move onto previous, got {:#?}",
            transcription.segments[0]
        );
        assert!(
            transcription.segments.len() >= 2,
            "remainder must stay after the overlap units, got {:#?}",
            transcription.segments
        );
        assert!(
            transcription.segments[1].text.starts_with("通常会"),
            "remainder must start after the overlap units, got {:#?}",
            transcription.segments[1]
        );
        assert!(
            (transcription.segments[0].end - 25.5).abs() < 1e-3,
            "previous.end must stay on the first slice window, got {:#?}",
            transcription.segments
        );
        assert!(
            (transcription.segments[1].start - 25.0).abs() < 1e-3,
            "remainder must keep the later slice window, got {:#?}",
            transcription.segments
        );
    }

    #[test]
    fn assembler_keeps_repeated_phrase_far_from_seam_with_acoustic_words() {
        // Whisper (acoustic word timestamps) on a music clip: the previous slice
        // ends with "She came from Planet Claire" at 233-236s absolute, and the
        // next slice legitimately re-sings the same phrase 9s later at 245-249s
        // absolute. The texts share a suffix/prefix seam, but the matched
        // word-instances sit ~9s apart in the original audio -- far outside the
        // ~0.5s slice overlap. The seam-stitch must NOT consume the phrase
        // out of both segments. This is the false-negative the regap guard
        // exists to prevent; the worded Decode Invariant family
        // (approximate_word_timestamps=true) is exempt because its tile times
        // make the regap meaningless.
        //
        // Whisper word times are slice-relative, so both segments use
        // RelativeToSliceContent; the slice offsets land prev's clause at
        // 212.5-239.0s and curr's at 238.5-265.0s in the original axis (the
        // 0.5s abut matches the default slice overlap).
        let prev_text = "Some say she's from Mars She came from Planet Claire";
        let cur_text = "She came from Planet Claire all the trees are red";
        // slice 1 covers 212..239 original; slice 2 covers 238.5..265 original.
        // Both use content==slice so `RelativeToSliceContent` places them at the
        // right spot on the original timeline via content_offset.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 16_000 * 212, 16_000 * 239),
            text: prev_text.to_string(),
            segments: vec![absolute_segment(
                prev_text,
                0.0,
                27.0,
                vec![
                    word("Some", 0.5, 0.9),
                    word("say", 0.9, 1.2),
                    word("she's", 1.2, 1.7),
                    word("from", 1.7, 2.0),
                    word("Mars", 2.0, 2.8),
                    word("She", 20.5, 20.9),
                    word("came", 20.9, 21.3),
                    word("from", 21.3, 21.7),
                    word("Planet", 21.7, 22.4),
                    word("Claire", 22.4, 24.1),
                ],
            )],
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(1, 16_000 * 238 + 8_000, 16_000 * 265),
            text: cur_text.to_string(),
            segments: vec![absolute_segment(
                cur_text,
                0.0,
                27.0,
                vec![
                    word("She", 2.0, 2.4),
                    word("came", 2.4, 2.8),
                    word("from", 2.8, 3.2),
                    word("Planet", 3.2, 3.9),
                    word("Claire", 3.9, 5.6),
                    word("all", 6.5, 6.8),
                    word("the", 6.8, 7.1),
                    word("trees", 7.1, 7.6),
                    word("are", 7.6, 7.9),
                    word("red", 7.9, 8.5),
                ],
            )],
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(
            transcription
                .text
                .matches("She came from Planet Claire")
                .count(),
            2,
            "a repeated phrase 8s apart must survive both segments, got {:?}",
            transcription.text
        );
        assert_eq!(
            transcription.segments.len(),
            2,
            "the second verse must not be dropped as a seam, got {:#?}",
            transcription.segments
        );
    }

    #[test]
    fn assembler_keeps_overlap_words_when_seam_phrase_is_previous_tail() {
        // Whisper (acoustic) genuine re-read at the 292.0s slice cut: the same
        // "There we go" audio (290.4-292.0) sits in the inter-slice overlap, so
        // BOTH slices read it and each emits it. In the previous slice it is the
        // segment TAIL ("...wee bit. There we go."); in the next it is the
        // segment HEAD ("There we go. All right."). The current slice's copy
        // BEGINS inside the overlap, before previous.end (292.0), so it is a
        // true re-read: the stitch dedupes to one phrase. The deduped phrase's
        // word windows must survive on the earlier (committed) segment, not be
        // orphaned from both sides.
        let prev_text = "wee bit. There we go.";
        let cur_text = "There we go. All right.";
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 16_000 * 265, 16_000 * 292),
            text: prev_text.to_string(),
            segments: vec![absolute_segment(
                prev_text,
                265.0,
                292.0,
                vec![
                    word("wee", 268.0, 269.0),
                    word("bit.", 289.6, 290.2),
                    word("There", 290.40, 291.10),
                    word("we", 291.10, 291.55),
                    word("go.", 291.55, 292.00),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(1, 16_000 * 291, 16_000 * 319),
            text: cur_text.to_string(),
            segments: vec![absolute_segment(
                cur_text,
                291.5,
                318.5,
                vec![
                    word("There", 291.55, 291.65),
                    word("we", 291.65, 291.90),
                    word("go.", 291.90, 292.00),
                    word("All", 294.03, 294.98),
                    word("right.", 294.78, 296.48),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let transcription = assembler.into_transcription();

        // The seam must have deduped (one "There we go.", remainder after it).
        assert_eq!(
            transcription.text.matches("There we go").count(),
            1,
            "a seam re-read must collapse to one phrase, got {:?}",
            transcription.text
        );

        // The phrase's words must survive in the word timeline exactly once:
        // on the earlier (committed) segment, not orphaned from both.
        let prev_words: Vec<String> = transcription.segments[0]
            .words
            .iter()
            .map(|w| w.word.trim().to_string())
            .collect();
        assert!(
            prev_words.iter().any(|w| w.eq_ignore_ascii_case("There")),
            "previous tail 'There' lost its word window (orphaned phrase), got {prev_words:?}"
        );
        assert!(
            prev_words.iter().any(|w| w.eq_ignore_ascii_case("go.")),
            "previous tail 'go.' lost its word window (orphaned phrase), got {prev_words:?}"
        );
        let cur_words: Vec<String> = transcription.segments[1]
            .words
            .iter()
            .map(|w| w.word.trim().to_string())
            .collect();
        assert!(
            !cur_words.iter().any(|w| w.eq_ignore_ascii_case("There")),
            "current seam copy must not double the phrase, got {cur_words:?}"
        );
        assert!(
            cur_words.iter().any(|w| w.eq_ignore_ascii_case("All")),
            "the post-seam remainder must survive, got {cur_words:?}"
        );
    }

    #[test]
    fn assembler_keeps_distinct_repeated_phrase_at_the_cut() {
        // Whisper (acoustic) on bonnie's real 292.0s cut: "there we go" is said
        // TWICE, with a silent gap across the cut -- utterance 1 in the previous
        // slice's tail (ending at 292.0), then silence, then utterance 2 in the
        // next slice's HEAD (292.44-294.23, fully AFTER previous.end). This is a
        // legitimate echo, not a slice-overlap re-read: the current slice's copy
        // is NEW audio the previous slice never transcribed. The regap between
        // the two word-instances is only ~0.5s (inside the 2s regap guard), so
        // the old path stitched it and DROPPED utterance 2. The reread
        // past-previous-end guard must refuse the stitch, so both "There we go"
        // survive, each timed.
        let prev_text = "wee bit. There we go.";
        let cur_text = "There we go. All right.";
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 16_000 * 265, 16_000 * 292),
            text: prev_text.to_string(),
            segments: vec![absolute_segment(
                prev_text,
                265.0,
                292.0,
                vec![
                    word("wee", 268.0, 269.0),
                    word("bit.", 290.0, 290.6),
                    word("There", 290.40, 291.30),
                    word("we", 291.30, 291.70),
                    word("go.", 291.70, 292.00),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(1, 16_000 * 291, 16_000 * 319),
            text: cur_text.to_string(),
            segments: vec![absolute_segment(
                cur_text,
                291.5,
                318.5,
                vec![
                    word("There", 292.44, 292.74),
                    word("we", 292.54, 293.32),
                    word("go.", 293.12, 293.60),
                    word("All", 294.03, 294.98),
                    word("right.", 294.78, 296.48),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let transcription = assembler.into_transcription();

        // Both utterances must survive (the echo must not be collapsed to one).
        assert_eq!(
            transcription.text.matches("There we go").count(),
            2,
            "a repeated phrase said twice across the cut must not be deduped, got {:?}",
            transcription.text
        );

        // The second utterance's words must survive on the later segment.
        let cur_words: Vec<String> = transcription
            .segments
            .last()
            .map(|segment| {
                segment
                    .words
                    .iter()
                    .map(|w| w.word.trim().to_string())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        assert!(
            cur_words.iter().any(|w| w.eq_ignore_ascii_case("There")),
            "the SECOND 'there we go' lost its word window, got {cur_words:?}"
        );
        assert!(
            cur_words.iter().any(|w| w.eq_ignore_ascii_case("All")),
            "the post-echo continuation must survive, got {cur_words:?}"
        );
    }

    #[test]
    fn assembler_stitches_single_acoustic_word_straddling_the_cut() {
        // The ducks shape: the cut lands mid-("too") at 72.55s. The earlier
        // copy is clamped at the boundary, the later slice re-decodes the
        // same word 0.08s past it. The transcript must carry exactly one
        // "too", the re-homed text must take the current slice's seam punct
        // (no "too,."), and the later segment must resume at "man."
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000 * 72 + 8_800),
            text: "that's my philosophy too.".to_string(),
            segments: vec![absolute_segment(
                "that's my philosophy too.",
                70.0,
                72.55,
                vec![
                    word("that's", 70.5, 70.9),
                    word("my", 71.0, 71.3),
                    word("philosophy", 71.4, 72.1),
                    word("too.", 72.26, 72.55),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000 * 72 + 8_800, 16_000 * 75),
            text: "too, man. would i like to".to_string(),
            segments: vec![absolute_segment(
                "too, man. would i like to",
                72.55,
                75.0,
                vec![
                    word("too,", 72.63, 73.30),
                    word("man.", 73.30, 73.60),
                    word("would", 73.80, 74.20),
                    word("i", 74.30, 74.50),
                    word("like", 74.55, 74.90),
                    word("to", 74.95, 75.00),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(
            transcription.text,
            "that's my philosophy too, man. would i like to"
        );
        assert_eq!(transcription.segments.len(), 2);
        let too_words = transcription
            .segments
            .iter()
            .flat_map(|segment| segment.words.iter())
            .filter(|w| w.word.trim_matches(['.', ',']).eq_ignore_ascii_case("too"))
            .count();
        assert_eq!(
            too_words, 1,
            "the straddling word must survive exactly once, got {:#?}",
            transcription
        );
        assert_eq!(transcription.segments[1].words[0].word, "man.");
        assert_eq!(stats.duplicate_merge_count, 1);
    }

    #[test]
    fn vy_out_straddle_repro() {
        // Byte-exact vy/whisper-large-v3-turbo segments 6/7 around the
        // 504.0s cut: "out." clamped at the previous end (504.0), "out,"
        // restarting 34ms past it on the next slice.
        let previous_text = "a really tough day today and I want you guys to know that I got a lot of things done. I got so much done today in like the span of like... not even... it wasn't even that busy and then towards the end it got busy and I got so much done and like the feeling of being able to be like, \"Oh, it's almost three o'clock. I can go play video games with the hotties.\" I was so relieved. You guys don't understand. The moment that I hit go live, I blacked out.";
        let current_text = "out, but also my body starts relaxing. So thank you. This is a really special place. I don't know how many times I've said this, but this is a really special place. And I'm really happy to see you guys. Hi. I worked really hard today. I'm sure you did too, hottie. Right? You worked hard today? Good job. Good job. What's up? I'm starting to understand that";
        let previous_words = vec![
            word("a", 477.19, 477.79),
            word("really", 477.59, 478.181),
            word("tough", 477.981, 478.507),
            word("day", 478.307, 478.776),
            word("today", 478.576, 479.2),
            word("and", 479.0, 479.575),
            word("I", 479.375, 479.702),
            word("want", 479.502, 479.862),
            word("you", 479.662, 480.031),
            word("guys", 479.831, 480.184),
            word("to", 479.984, 480.286),
            word("know", 480.086, 480.411),
            word("that", 480.211, 480.546),
            word("I", 480.346, 480.689),
            word("got", 480.489, 481.044),
            word("a", 480.844, 481.456),
            word("lot", 481.256, 481.7),
            word("of", 481.5, 481.918),
            word("things", 481.718, 482.2075),
            word("done.", 482.0075, 482.4495),
            word("I", 482.2495, 482.609),
            word("got", 482.409, 482.928),
            word("so", 482.728, 483.377),
            word("much", 483.177, 483.792),
            word("done", 483.592, 484.179),
            word("today", 483.979, 484.698),
            word("in", 484.498, 485.122),
            word("like", 484.922, 485.282),
            word("the", 485.082, 485.577),
            word("span", 485.377, 485.956),
            word("of", 485.756, 486.461),
            word("like...", 486.261, 487.142),
            word("not", 486.942, 487.585),
            word("even...", 487.385, 487.849),
            word("it", 487.649, 488.1635),
            word("wasn't", 487.9635, 488.4945),
            word("even", 488.2945, 488.682),
            word("that", 488.482, 488.905),
            word("busy", 488.705, 489.187),
            word("and", 488.987, 489.393),
            word("then", 489.193, 489.578),
            word("towards", 489.378, 489.8),
            word("the", 489.6, 489.955),
            word("end", 489.755, 490.073),
            word("it", 489.873, 490.204),
            word("got", 490.004, 490.405),
            word("busy", 490.205, 490.795),
            word("and", 490.595, 491.115),
            word("I", 490.915, 491.242),
            word("got", 491.042, 491.447),
            word("so", 491.247, 491.68),
            word("much", 491.48, 491.925),
            word("done", 491.725, 492.171),
            word("and", 491.971, 492.351),
            word("like", 492.151, 492.513),
            word("the", 492.313, 492.761),
            word("feeling", 492.561, 493.078),
            word("of", 492.878, 493.29102),
            word("being", 493.091, 493.489),
            word("able", 493.289, 493.7),
            word("to", 493.5, 493.873),
            word("be", 493.673, 494.06702),
            word("like,", 493.867, 494.282),
            word("\"Oh,", 494.082, 494.433),
            word("it's", 494.233, 494.636),
            word("almost", 494.436, 494.862),
            word("three", 494.662, 495.07376),
            word("o'clock.", 494.87375, 495.31726),
            word("I", 495.11725, 495.511),
            word("can", 495.311, 495.727),
            word("go", 495.527, 496.122),
            word("play", 495.922, 496.52),
            word("video", 496.32, 496.756),
            word("games", 496.556, 496.964),
            word("with", 496.764, 497.066),
            word("the", 496.866, 497.227),
            word("hotties.\"", 497.027, 497.487),
            word("I", 497.287, 497.738),
            word("was", 497.538, 498.077),
            word("so", 497.877, 498.6675),
            word("relieved.", 498.4675, 499.1475),
            word("You", 498.9475, 499.293),
            word("guys", 499.093, 499.49152),
            word("don't", 499.2915, 500.0945),
            word("understand.", 499.8945, 501.037),
            word("The", 500.837, 501.654),
            word("moment", 501.454, 501.938),
            word("that", 501.738, 502.106),
            word("I", 501.906, 502.249),
            word("hit", 502.049, 502.451),
            word("go", 502.251, 502.937),
            word("live,", 502.737, 503.581),
            word("I", 503.381, 503.898),
            word("blacked", 503.698, 504.0),
            word("out.", 503.83, 504.0),
        ];
        let current_words = vec![
            word("out,", 504.0345, 504.5455),
            word("but", 504.3455, 504.803),
            word("also", 504.603, 505.08),
            word("my", 504.88, 505.361),
            word("body", 505.161, 505.696),
            word("starts", 505.496, 506.453),
            word("relaxing.", 506.253, 507.953),
            word("So", 509.28, 509.648),
        ];
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000 * 504),
            text: previous_text.to_string(),
            segments: vec![absolute_segment(
                previous_text,
                477.5,
                504.0,
                previous_words,
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000 * 504 + 552, 16_000 * 531),
            text: current_text.to_string(),
            segments: vec![absolute_segment(
                current_text,
                504.0345,
                530.5,
                current_words,
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(
            stats.duplicate_merge_count, 1,
            "the 34ms-gap straddle must stitch, got {:#?}",
            transcription
        );
        let out_words = transcription
            .segments
            .iter()
            .flat_map(|segment| segment.words.iter())
            .filter(|w| w.word.trim_matches(['.', ',']).eq_ignore_ascii_case("out"))
            .count();
        assert_eq!(
            out_words, 1,
            "the straddling 'out' must survive exactly once, got {:#?}",
            transcription
        );
        assert!(
            transcription.segments[0].text.ends_with("blacked out,"),
            "re-homed seam punct must win over the clamped copy's period, got {:#?}",
            transcription
        );
        assert!(
            transcription.segments[1]
                .text
                .starts_with("but also my body"),
            "the remainder must resume after the re-read, got {:#?}",
            transcription
        );
    }

    #[test]
    fn assembler_stitches_acoustic_seam_fronted_by_committed_fragments() {
        // The real vy shape: the 504.0s cut splits "...I blacked out." The
        // next slice re-decodes the overlap tail as fragments plus the
        // straddling word ("I black out, but also..."). The fragment fronting
        // blocks a head-anchored prefix match, so without the
        // committed-fragment skip the seam would never be seen and both "out"
        // copies survive. Every fragment word sits in the committed region
        // (midpoint before the boundary: a word the midpoint trim drops), so
        // the skip stands: one "out" survives, the re-homed seam punct wins,
        // the remainder resumes at "but".
        let previous_text = "a really rough day today and I want you to know that I got a lot done. The moment that I hit go live, I blacked out.";
        let current_text = "I black out, but also my body starts relaxing. So thank you. This is a really special place.";
        let previous_words = vec![
            word("a", 477.19, 477.79),
            word("really", 477.59, 478.181),
            word("rough", 478.307, 478.776),
            word("day", 478.9, 479.4),
            word("today", 479.5, 480.1),
            word("and", 480.3, 480.8),
            word("I", 481.0, 481.4),
            word("want", 481.6, 482.1),
            word("you", 482.3, 482.8),
            word("to", 483.0, 483.3),
            word("know", 483.4, 483.9),
            word("that", 484.1, 484.6),
            word("I", 484.8, 485.2),
            word("got", 485.4, 485.9),
            word("a", 486.1, 486.4),
            word("lot", 486.5, 486.9),
            word("done.", 487.0, 487.6),
            word("The", 500.837, 501.654),
            word("moment", 501.454, 501.938),
            word("that", 501.738, 502.106),
            word("I", 501.906, 502.249),
            word("hit", 502.049, 502.451),
            word("go", 502.251, 502.937),
            word("live,", 502.737, 503.581),
            word("I", 503.381, 503.898),
            word("blacked", 503.698, 504.0),
            word("out.", 503.83, 504.0),
        ];
        let current_words = vec![
            word("I", 503.5, 503.72),
            word("black", 503.72, 504.0),
            word("out,", 504.0345, 504.5455),
            word("but", 504.3455, 504.803),
            word("also", 504.603, 505.08),
            word("my", 504.88, 505.361),
            word("body", 505.161, 505.696),
            word("starts", 505.496, 506.453),
            word("relaxing.", 506.253, 507.953),
            word("So", 509.28, 509.648),
        ];
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000 * 477, 16_000 * 504),
            text: previous_text.to_string(),
            segments: vec![absolute_segment(
                previous_text,
                477.0,
                504.0,
                previous_words,
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000 * 503 + 8_000, 16_000 * 531),
            text: current_text.to_string(),
            segments: vec![absolute_segment(current_text, 503.5, 530.5, current_words)],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(
            stats.duplicate_merge_count, 1,
            "the fragment-fronted straddle must stitch, got {:#?}",
            transcription
        );
        let out_words = transcription
            .segments
            .iter()
            .flat_map(|segment| segment.words.iter())
            .filter(|w| w.word.trim_matches(['.', ',']).eq_ignore_ascii_case("out"))
            .count();
        assert_eq!(
            out_words, 1,
            "the straddling 'out' must survive exactly once, got {:#?}",
            transcription
        );
        assert!(
            transcription.segments[0].text.ends_with("blacked out,"),
            "the previous tail keeps its own words and takes the re-homed seam punct, got {:#?}",
            transcription
        );
        assert!(
            !transcription.segments[0]
                .text
                .ends_with("blacked I black out"),
            "the fragment text must not graft onto the previous tail, got {:#?}",
            transcription
        );
        assert!(
            transcription.segments[1]
                .text
                .starts_with("but also my body"),
            "the fragment and the re-read must both leave the remainder, got {:#?}",
            transcription
        );
        assert_eq!(
            transcription.segments[1]
                .words
                .first()
                .map(|w| w.word.as_str()),
            Some("but"),
            "the fragment's and re-read's word windows must not survive, got {:#?}",
            transcription
        );
    }

    #[test]
    fn assembler_keeps_seam_fronted_by_new_word() {
        // Negative of the fragment skip: same geometry, but the fronting word
        // is NEW audio (its midpoint lies past the previous end), a fresh
        // phrase that merely happens to be lead by the previous tail word.
        // The committed-region gate refuses the skip; both copies survive.
        let previous_text = "we shut it out.";
        let current_text = "well out, but the feed held.";
        let previous_words = vec![
            word("we", 500.1, 500.5),
            word("shut", 500.7, 501.1),
            word("it", 501.2, 501.5),
            word("out.", 503.83, 504.0),
        ];
        let current_words = vec![
            word("well", 504.1, 504.45),
            word("out,", 504.05, 504.4),
            word("but", 504.45, 504.8),
            word("the", 504.9, 505.1),
            word("feed", 505.2, 505.6),
            word("held.", 505.7, 506.1),
        ];
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000 * 477, 16_000 * 504),
            text: previous_text.to_string(),
            segments: vec![absolute_segment(
                previous_text,
                477.0,
                504.0,
                previous_words,
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000 * 503 + 8_000, 16_000 * 531),
            text: current_text.to_string(),
            segments: vec![absolute_segment(current_text, 503.5, 530.5, current_words)],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(
            stats.duplicate_merge_count, 0,
            "a new-word-fronted seam must not stitch, got {:#?}",
            transcription
        );
        let out_words = transcription
            .segments
            .iter()
            .flat_map(|segment| segment.words.iter())
            .filter(|w| w.word.trim_matches(['.', ',']).eq_ignore_ascii_case("out"))
            .count();
        assert_eq!(
            out_words, 2,
            "the new phrase must keep its own lead word, got {:#?}",
            transcription
        );
        assert!(
            transcription.segments[1].text.starts_with("well out,"),
            "the fresh phrase must survive whole, got {:#?}",
            transcription
        );
    }

    #[test]
    fn assembler_keeps_single_word_said_naturally_before_the_cut() {
        // The plomet shape: the earlier "yeah." finished 0.4s BEFORE the
        // 451.0s cut (not clipped by it), and the later "yeah." is a genuine
        // back-to-back repeat. The seam-clamp check refuses the stitch; both
        // copies survive with their word windows.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000 * 451),
            text: "mhm. yeah.".to_string(),
            segments: vec![absolute_segment(
                "mhm. yeah.",
                448.0,
                451.0,
                vec![word("mhm.", 448.5, 449.2), word("yeah.", 448.9, 450.6)],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000 * 451, 16_000 * 477),
            text: "yeah. i've been there".to_string(),
            segments: vec![absolute_segment(
                "yeah. i've been there",
                450.74,
                455.0,
                vec![
                    word("yeah.", 450.74, 452.38),
                    word("i've", 452.18, 453.88),
                    word("been", 454.05, 454.50),
                    word("there", 454.50, 455.00),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(
            transcription
                .text
                .split_whitespace()
                .filter(|w| w.trim_matches(['.', ',']).eq_ignore_ascii_case("yeah"))
                .count(),
            2,
            "a genuine back-to-back repeat must not be deduped, got {:#?}",
            transcription
        );
        assert_eq!(stats.duplicate_merge_count, 0);
    }

    #[test]
    fn assembler_keeps_single_word_echo_beginning_clear_after_cut() {
        // The bonnie shape: "america!" clamped at the 371.5s cut, a clear
        // pause, then the re-announcement 0.82s past the boundary. New audio,
        // not a re-read: the past-previous-end guard refuses and both copies
        // survive.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000 * 371 + 8_000),
            text: "happy independence day america!".to_string(),
            segments: vec![absolute_segment(
                "happy independence day america!",
                369.0,
                371.5,
                vec![
                    word("happy", 369.2, 369.8),
                    word("day", 369.75, 370.9),
                    word("america!", 370.7, 371.5),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000 * 371 + 8_000, 16_000 * 398),
            text: "america! america! hello!".to_string(),
            segments: vec![absolute_segment(
                "america! america! hello!",
                371.5,
                375.0,
                vec![
                    word("america!", 372.32, 372.68),
                    word("america!", 372.48, 373.76),
                    word("hello!", 373.56, 374.97),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(
            transcription
                .text
                .split_whitespace()
                .filter(|w| w.trim_matches(['!', '.']).eq_ignore_ascii_case("america"))
                .count(),
            3,
            "the echo must survive both segments, got {:#?}",
            transcription
        );
        assert_eq!(stats.duplicate_merge_count, 0);
    }

    #[test]
    fn assembler_keeps_single_word_reread_and_genuine_second_copy() {
        // The ali shape: the speech is "good. good." -- the first "good" sits
        // ON the 159.5s cut (clamped) and the second is a genuine immediate
        // repeat. The later slice re-decoded the straddling first copy too,
        // so the seam carries it twice. The stitch consumes only the
        // re-read instance (the one at the cut), leaving the genuine pair.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000 * 159 + 8_000),
            text: "there we go. good.".to_string(),
            segments: vec![absolute_segment(
                "there we go. good.",
                157.0,
                159.5,
                vec![
                    word("there", 157.5, 157.8),
                    word("we", 157.8, 158.2),
                    word("go.", 157.8, 158.9),
                    word("good.", 159.14, 159.50),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000 * 159 + 8_000, 16_000 * 186),
            text: "good. good. that's it.".to_string(),
            segments: vec![absolute_segment(
                "good. good. that's it.",
                159.44,
                162.0,
                vec![
                    word("good.", 159.44, 160.07),
                    word("good.", 159.87, 160.90),
                    word("that's", 160.70, 161.30),
                    word("it.", 161.50, 162.00),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(
            transcription
                .text
                .split_whitespace()
                .filter(|w| w.trim_matches(['.', ',']).eq_ignore_ascii_case("good"))
                .count(),
            2,
            "only the re-read copy may be consumed, got {:#?}",
            transcription
        );
        assert_eq!(
            transcription.segments[0].text, "there we go. good.",
            "the earlier copy keeps its clamped window text, got {:#?}",
            transcription.segments[0]
        );
        assert!(
            transcription.segments[1].text.starts_with("good."),
            "the genuine second copy must lead the remainder, got {:#?}",
            transcription.segments[1]
        );
        assert_eq!(stats.duplicate_merge_count, 1);
    }

    #[test]
    fn assembler_keeps_single_word_seam_without_acoustic_vet() {
        // Wordless segments: a one-unit English seam has no word times for
        // the vet to run on, so it is refused (the historical head-word
        // protection) and both copies survive.
        let transcription =
            assemble_wordless_overlap("that's my philosophy too.", "too, man. would i like to");
        assert_eq!(transcription.segments.len(), 2);
        assert!(
            transcription.text.contains("too.") && transcription.text.contains("too,"),
            "both wordless copies must survive, got {:?}",
            transcription.text
        );
    }

    #[test]
    fn assembler_keeps_single_word_seam_for_approximate_families() {
        // Interpolated (DecodeInvariant) families: the acoustic vet cannot
        // run on synthetic tile times, so a one-unit English seam stays
        // refused even when the segments carry words.
        let previous = "that's my philosophy too.";
        let current = "too, man. would i like to";
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default())
                .with_approximate_word_timestamps(true);
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000 * 72 + 8_800),
            text: previous.to_string(),
            segments: vec![absolute_segment(
                previous,
                70.0,
                72.55,
                interpolated_words(previous, 70.0, 72.55),
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000 * 72 + 8_800, 16_000 * 75),
            text: current.to_string(),
            segments: vec![absolute_segment(
                current,
                72.55,
                75.0,
                interpolated_words(current, 72.55, 75.0),
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(stats.duplicate_merge_count, 0);
    }

    #[test]
    fn assembler_drops_b_side_seam_phantom_at_min_width() {
        // B-side seam phantom at the narrowest observed re-read width: the
        // cut lands on the committed "know." (decoded with certainty), and
        // the new slice re-reads its tail as the different token "And" -
        // 34% confident, stretched over the committed tail plus the pause
        // after the cut. The re-read is a decode artifact, not speech, so it
        // must not survive; the committed word is untouched.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000 * 179 + 8_000),
            text: "we know.".to_string(),
            segments: vec![absolute_segment(
                "we know.",
                178.9,
                179.5,
                vec![
                    word_conf("we", 179.0, 179.3, 0.97),
                    word_conf("know.", 179.33, 179.5, 0.982),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000 * 179, 16_000 * 183),
            text: "And I know what you mean.".to_string(),
            segments: vec![absolute_segment(
                "And I know what you mean.",
                179.32,
                182.4,
                vec![
                    word_conf("And", 179.32, 180.28, 0.338),
                    word("I", 180.08, 180.54),
                    word("know", 180.54, 181.0),
                    word("what", 181.0, 181.45),
                    word("you", 181.45, 181.8),
                    word("mean.", 181.8, 182.4),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(
            stats.duplicate_merge_count, 1,
            "the phantom head must be dropped, got {:#?}",
            transcription
        );
        assert!(
            !transcription.text.contains("And"),
            "the low-confidence re-read must not survive, got {:?}",
            transcription.text
        );
        assert_eq!(
            transcription.segments[0].text, "we know.",
            "the committed segment must stay untouched, got {:#?}",
            transcription.segments
        );
        assert_eq!(
            transcription.segments[1].text, "I know what you mean.",
            "the remainder must resume after the phantom, got {:?}",
            transcription.segments[1].text
        );
        assert_eq!(
            transcription.segments[1]
                .words
                .first()
                .map(|word| word.word.as_str()),
            Some("I"),
            "the phantom word window must not survive, got {:#?}",
            transcription
        );
    }

    #[test]
    fn assembler_drops_stretched_b_side_seam_phantom() {
        // Wide phantom: the cut lands on the committed "tunnel." and the new
        // slice emits one 45%-confident word whose window stretches 6s over
        // the committed tail and the following silence (the decode padded
        // silence with a made-up word). The whole window is artifact.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000 * 182 + 1_760),
            text: "into the tunnel.".to_string(),
            segments: vec![absolute_segment(
                "into the tunnel.",
                180.5,
                182.11,
                vec![
                    word_conf("into", 180.8, 181.1, 0.96),
                    word_conf("the", 181.1, 181.3, 0.95),
                    word_conf("tunnel.", 181.52, 182.11, 0.919),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000 * 181 + 8_000, 16_000 * 190),
            text: "normal. But yeah, it happened.".to_string(),
            segments: vec![absolute_segment(
                "normal. But yeah, it happened.",
                181.77,
                189.95,
                vec![
                    word_conf("normal.", 181.77, 187.92, 0.455),
                    word("But", 188.12, 188.5),
                    word("yeah,", 188.6, 189.0),
                    word("it", 189.0, 189.3),
                    word("happened.", 189.3, 189.95),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(
            stats.duplicate_merge_count, 1,
            "the stretched phantom head must be dropped, got {:#?}",
            transcription
        );
        assert!(
            !transcription.text.contains("normal"),
            "the stretched re-read must not survive, got {:?}",
            transcription.text
        );
        assert!(
            transcription.segments[0].text.ends_with("into the tunnel."),
            "the committed tail must keep its own word, got {:?}",
            transcription.segments[0].text
        );
        assert!(
            transcription.segments[1].text.starts_with("But yeah"),
            "the remainder must resume at the first real word, got {:?}",
            transcription.segments[1].text
        );
        assert_eq!(
            transcription.segments[1]
                .words
                .first()
                .map(|word| word.word.as_str()),
            Some("But"),
            "the phantom window must not survive, got {:#?}",
            transcription
        );
    }

    #[test]
    fn assembler_drops_seam_phantom_fronted_by_committed_fragments() {
        // The real decode shape of a stretched phantom: the slice re-read
        // emits a run of low-confidence fragments compressed into the
        // committed region, then one stretched word padded over the
        // following silence. The midpoint trim clears the fragments on its
        // own; the phantom rule must then see the stretched word - now the
        // surviving head - and drop it.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000 * 182 + 1_760),
            text: "into the tunnel.".to_string(),
            segments: vec![absolute_segment(
                "into the tunnel.",
                180.5,
                182.11,
                vec![
                    word_conf("into", 180.8, 181.1, 0.96),
                    word_conf("the", 181.1, 181.3, 0.95),
                    word_conf("tunnel.", 181.52, 182.11, 0.919),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000 * 181 + 8_000, 16_000 * 190),
            text: "So, I'm going back to normal. But yeah.".to_string(),
            segments: vec![absolute_segment(
                "So, I'm going back to normal. But yeah.",
                181.61,
                189.0,
                vec![
                    word_conf("So,", 181.61, 181.82, 0.052),
                    word_conf("I'm", 181.62, 181.88, 0.149),
                    word_conf("going", 181.68, 181.88, 0.088),
                    word_conf("back", 181.68, 181.88, 0.453),
                    word_conf("to", 181.68, 181.97, 0.763),
                    word_conf("normal.", 181.77, 187.92, 0.455),
                    word("But", 188.12, 188.5),
                    word("yeah.", 188.6, 188.95),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(
            stats.duplicate_merge_count, 1,
            "the stretched head behind the fragments must be dropped, got {:#?}",
            transcription
        );
        assert!(
            !transcription.text.contains("normal"),
            "the stretched re-read must not survive, got {:?}",
            transcription.text
        );
        assert!(
            !transcription.text.contains("So,"),
            "the committed-region fragments must not survive, got {:?}",
            transcription.text
        );
        assert!(
            transcription.segments[0].text.ends_with("into the tunnel."),
            "the committed tail must keep its own word, got {:?}",
            transcription.segments[0].text
        );
        assert_eq!(
            transcription.segments[1].text, "But yeah.",
            "the remainder must resume at the first new word, got {:?}",
            transcription.segments[1].text
        );
        assert_eq!(
            transcription.segments[1]
                .words
                .first()
                .map(|word| word.word.as_str()),
            Some("But"),
            "only the new words may survive, got {:#?}",
            transcription
        );
    }

    #[test]
    fn assembler_keeps_b_side_head_when_committed_word_uncertain() {
        // The committed word is decoded at 82% confidence (under the 85%
        // floor) even though the re-read is a different token at 41%
        // confidence stretched a full second wide: both readings are live
        // candidates and neither may be eaten. Both survive.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000 * 451),
            text: "i said it?".to_string(),
            segments: vec![absolute_segment(
                "i said it?",
                450.2,
                451.0,
                vec![
                    word_conf("i", 450.3, 450.5, 0.9),
                    word_conf("said", 450.5, 450.75, 0.94),
                    word_conf("it?", 450.8, 451.0, 0.82),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000 * 450 + 12_000, 16_000 * 454),
            text: "if? then we left.".to_string(),
            segments: vec![absolute_segment(
                "if? then we left.",
                450.91,
                452.95,
                vec![
                    word_conf("if?", 450.91, 451.91, 0.41),
                    word("then", 451.95, 452.3),
                    word("we", 452.3, 452.6),
                    word("left.", 452.6, 452.95),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(
            stats.duplicate_merge_count, 0,
            "an uncertain committed word must not hand its audio away, got {:#?}",
            transcription
        );
        assert_eq!(transcription.segments[0].text, "i said it?");
        assert_eq!(
            transcription.segments[1].text, "if? then we left.",
            "the straddling head must lead the remainder, got {:?}",
            transcription.segments[1].text
        );
    }

    #[test]
    fn assembler_keeps_tight_b_side_head_over_confident_committed_word() {
        // Width gate: the committed word decodes at 99.8% and the re-read at
        // 23.5%, but its window is tight (0.37s, close to its spoken
        // duration) - a genuine straddling word, not a stretch over silence.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000 * 64 + 4_800),
            text: "working on your".to_string(),
            segments: vec![absolute_segment(
                "working on your",
                63.3,
                64.3,
                vec![
                    word_conf("working", 63.45, 63.87, 0.99),
                    word_conf("on", 63.67, 64.08, 0.994),
                    word_conf("your", 64.05, 64.3, 0.998),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000 * 64, 16_000 * 67),
            text: "free time is no longer".to_string(),
            segments: vec![absolute_segment(
                "free time is no longer",
                64.05,
                65.7,
                vec![
                    word_conf("free", 64.26, 64.626, 0.235),
                    word("time", 64.43, 64.93),
                    word("is", 64.73, 65.1),
                    word("no", 64.9, 65.31),
                    word("longer", 65.11, 65.7),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(
            stats.duplicate_merge_count, 0,
            "a tight window is a word, not a phantom, got {:#?}",
            transcription
        );
        assert_eq!(
            transcription.segments[1].text, "free time is no longer",
            "the genuine straddle must survive whole, got {:?}",
            transcription.segments[1].text
        );
    }

    #[test]
    fn assembler_keeps_b_side_head_without_confidence() {
        // No confidence is no evidence. Some families emit word timestamps
        // without per-word confidences; the rule stays inert rather than
        // guessing.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000 * 318 + 15_040),
            text: "i work.".to_string(),
            segments: vec![absolute_segment(
                "i work.",
                318.0,
                318.94,
                vec![
                    word_conf("i", 318.1, 318.3, 0.96),
                    word_conf("work.", 318.44, 318.94, 0.958),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000 * 318 + 8_000, 16_000 * 321),
            text: "Right again.".to_string(),
            segments: vec![absolute_segment(
                "Right again.",
                318.93,
                319.95,
                vec![
                    word("Right", 318.93, 319.61),
                    word("again.", 319.61, 319.95),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(
            stats.duplicate_merge_count, 0,
            "the rule must be inert without confidences, got {:#?}",
            transcription
        );
        assert_eq!(
            transcription.segments[1].text, "Right again.",
            "the head word must survive, got {:?}",
            transcription.segments[1].text
        );
    }

    #[test]
    fn assembler_keeps_b_side_phantom_shape_for_approximate_families() {
        // Interpolated (non-acoustic) tile times: width is a function of the
        // tile, not of speech, so the stretch test is meaningless and the
        // rule stays off even for a shape that would be a phantom on an
        // acoustic family.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default())
                .with_approximate_word_timestamps(true);
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000 * 179 + 8_000),
            text: "we know.".to_string(),
            segments: vec![absolute_segment(
                "we know.",
                178.9,
                179.5,
                vec![
                    word_conf("we", 179.0, 179.3, 0.97),
                    word_conf("know.", 179.33, 179.5, 0.982),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000 * 179, 16_000 * 183),
            text: "And I know what you mean.".to_string(),
            segments: vec![absolute_segment(
                "And I know what you mean.",
                179.32,
                182.4,
                vec![
                    word_conf("And", 179.32, 180.28, 0.338),
                    word("I", 180.08, 180.54),
                    word("know", 180.54, 181.0),
                    word("what", 181.0, 181.45),
                    word("you", 181.45, 181.8),
                    word("mean.", 181.8, 182.4),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(stats.duplicate_merge_count, 0);
        assert!(
            transcription.text.contains("And"),
            "the approximate family must keep its head word, got {:?}",
            transcription.text
        );
    }

    #[test]
    fn assembler_stitches_same_token_seam_instead_of_phantom_drop() {
        // Same-token ownership: the committed "out." clamps the cut and the
        // re-read is a low-confidence, stretched "out," - every phantom gate
        // except the token-inequality one would fire. Same-token seams are
        // the suffix-prefix stitch's territory (and the midpoint trim's):
        // the stitch consumes the re-read and the committed word stands.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: slice(0, 16_000 * 504),
            text: "i blacked out.".to_string(),
            segments: vec![absolute_segment(
                "i blacked out.",
                503.2,
                504.0,
                vec![
                    word_conf("i", 503.3, 503.5, 0.95),
                    word_conf("blacked", 503.5, 503.83, 0.93),
                    word_conf("out.", 503.83, 504.0, 0.98),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: slice(16_000 * 503 + 12_000, 16_000 * 508),
            text: "out, rest.".to_string(),
            segments: vec![absolute_segment(
                "out, rest.",
                503.85,
                507.2,
                vec![
                    word_conf("out,", 503.85, 506.85, 0.3),
                    word("rest.", 506.9, 507.2),
                ],
            )],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(stats.duplicate_merge_count, 1);
        let out_words = transcription
            .segments
            .iter()
            .flat_map(|segment| segment.words.iter())
            .filter(|word| {
                word.word
                    .trim_matches(['.', ','])
                    .eq_ignore_ascii_case("out")
            })
            .count();
        assert_eq!(
            out_words, 1,
            "the straddling 'out' must survive exactly once, got {:#?}",
            transcription
        );
        assert!(
            transcription.segments[0].text.ends_with("blacked out,"),
            "the re-homed seam punct must win, got {:?}",
            transcription.segments[0].text
        );
        assert_eq!(
            transcription.segments[1].text, "rest.",
            "the re-read word must leave the remainder, got {:?}",
            transcription.segments[1].text
        );
    }

    #[test]
    fn assembler_stitches_wordless_cjk_overlap_re_read() {
        // Field defect from qwen3-asr-0.6b on fixtures/longform_en_zh.wav: the
        // energy cut at 25.5s re-reads ~0.5s of audio, the model emits one
        // wordless segment per slice, and the overlap decoded as "的时候，"
        // after a truncated "周末的时候，我。".
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 0, 16_000 * 25 + 8_000),
            text: "周末的时候，我。".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(1, 16_000 * 25, 16_000 * 50 + 8_000),
            text: "的时候，我通常会读书或者看一部电影，放松一下。".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let (transcription, stats) = assembler.into_parts();
        assert_eq!(
            transcription.text.replace(' ', ""),
            "周末的时候，我通常会读书或者看一部电影，放松一下。"
        );
        assert_eq!(transcription.segments[0].text, "周末的时候，我");
        assert!(
            transcription.segments[1].text.starts_with("通常会"),
            "remainder must start after the overlap units, got {:#?}",
            transcription.segments
        );
        assert!(
            !transcription
                .segments
                .iter()
                .any(|segment| segment.text.contains("的时候，") && segment.start + 1.0e-3 >= 25.0),
            "re-read prefix must not survive on the later slice, got {:#?}",
            transcription.segments
        );
        assert!(
            (transcription.segments[0].end - 25.5).abs() < 1e-3,
            "stitch must not swallow previous.end, got {:#?}",
            transcription.segments
        );
        assert!(
            stats.duplicate_merge_count >= 1,
            "overlap re-read must count as a seam stitch, got {stats:?}"
        );
    }

    #[test]
    fn assembler_strips_truncated_period_when_remainder_continues() {
        // Qwen seam: the model plants `。` at the slice cut. Current continues
        // with content after the overlap, so that period is not real.
        let previous = "周末的时候，我。";
        let current = "的时候，我通常会读书或者看一部电影，放松一下。";
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 0, 16_000 * 25 + 8_000),
            text: previous.to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(1, 16_000 * 25, 16_000 * 50 + 8_000),
            text: current.to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let transcription = assembler.into_transcription();
        assert!(
            transcription.segments[0].text.ends_with('我'),
            "truncated period must be stripped, got {:#?}",
            transcription.segments[0]
        );
        assert!(
            !transcription.segments[0].text.ends_with('。'),
            "previous must not keep the fake period, got {:#?}",
            transcription.segments[0]
        );
        assert!(
            (transcription.segments[0].end - 25.5).abs() < 1e-3,
            "stripping the period must not move previous.end, got {:#?}",
            transcription.segments
        );
    }

    #[test]
    fn assembler_does_not_move_current_sentence_body_onto_previous() {
        // Text may cross the slice window; timestamps must stay with the
        // audio that produced them. Consumed = overlap units + following
        // punct only.
        let previous = "周末的时候，我。";
        let current = "的时候，我通常会读书或者看一部电影，放松一下。今天天气非常好，";
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 0, 16_000 * 25 + 8_000),
            text: previous.to_string(),
            segments: vec![absolute_segment(previous, 0.0, 25.5, Vec::new())],
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(1, 16_000 * 25, 16_000 * 50 + 8_000),
            text: current.to_string(),
            segments: vec![absolute_segment(current, 0.0, 25.5, Vec::new())],
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments[0].text, "周末的时候，我");
        assert!(
            !transcription.segments[0].text.contains("通常会"),
            "previous must not absorb speech after the slice boundary, got {:#?}",
            transcription.segments[0]
        );
        assert!(
            transcription.segments[1]
                .text
                .contains("通常会读书或者看一部电影，放松一下。"),
            "sentence body must stay on the later slice, got {:#?}",
            transcription.segments[1]
        );
        assert!(
            (transcription.segments[0].end - 25.5).abs() < 1e-3,
            "previous.end must stay 25.5, got {:#?}",
            transcription.segments
        );
        assert!(
            (transcription.segments[1].start - 25.0).abs() < 1e-3,
            "remainder must keep the later slice start, got {:#?}",
            transcription.segments
        );
    }

    #[test]
    fn assembler_joins_origin_longform_golden_slice_texts() {
        let mimo = assemble_three_wordless([
            "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country. 今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去一家新开的川菜馆吃饭，听说那里的麻婆豆腐特别正宗。周末的时候，我",
            "我通常会读书或者看一部电影放松一下。And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去一家新开的川菜馆吃饭，听说那里的麻婆豆腐特别正宗。",
            "周末的时候，我通常会读书或者看一部电影放松一下。And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.",
        ]);
        let firered_aed = assemble_three_wordless([
            "AND SO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU ASK WHAT YOU CAN DO FOR YOUR COUNTRY今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的川菜馆吃饭听说那里的麻婆豆腐特别正宗周末的时候我",
            "我通常会读书或者看一部电影放松一下 AND SO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU ASK WHAT YOU CAN DO FOR YOUR COUNTRY今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的川菜馆吃饭听说那里的麻婆豆腐特别正宗",
            "周末的时候我通常会读书或者看一部电影放松一下 ANDSO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU ASK WHAT YOU CAN DO FOR YOUR COUNTRY",
        ]);
        let firered_llm = assemble_three_wordless([
            "and so my fellow americans ask not what your country can do for you ask what you can do for your country 今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的川菜馆吃饭听说那里的麻婆豆腐特别正宗周末的时候我",
            "我通常会读书或者看一部电影放松一下 and so my fellow americans ask not what your country can do for you ask what you can do for your country 今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的川菜馆吃饭听说那里的麻婆豆腐特别正宗中",
            "周末的时候我通常会读书或者看一部电影放松一下 and so my fellow americans ask not what your country can do for you ask what you can do for your country",
        ]);
        assert_eq!(
            mimo,
            "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country. 今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去一家新开的川菜馆吃饭，听说那里的麻婆豆腐特别正宗。周末的时候，我通常会读书或者看一部电影放松一下。And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.今天天气非常好，我打算和朋友们一起去公园散步。晚上我们还计划去一家新开的川菜馆吃饭，听说那里的麻婆豆腐特别正宗。周末的时候，我通常会读书或者看一部电影放松一下。And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country."
        );
        assert_eq!(
            firered_aed,
            "AND SO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU ASK WHAT YOU CAN DO FOR YOUR COUNTRY今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的川菜馆吃饭听说那里的麻婆豆腐特别正宗周末的时候我通常会读书或者看一部电影放松一下 AND SO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU ASK WHAT YOU CAN DO FOR YOUR COUNTRY今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的川菜馆吃饭听说那里的麻婆豆腐特别正宗周末的时候我通常会读书或者看一部电影放松一下 ANDSO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU ASK WHAT YOU CAN DO FOR YOUR COUNTRY"
        );
        assert_eq!(
            firered_llm,
            "and so my fellow americans ask not what your country can do for you ask what you can do for your country 今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的川菜馆吃饭听说那里的麻婆豆腐特别正宗周末的时候我通常会读书或者看一部电影放松一下 and so my fellow americans ask not what your country can do for you ask what you can do for your country 今天天气非常好我打算和朋友们一起去公园散步晚上我们还计划去一家新开的川菜馆吃饭听说那里的麻婆豆腐特别正宗中周末的时候我通常会读书或者看一部电影放松一下 and so my fellow americans ask not what your country can do for you ask what you can do for your country"
        );
    }

    #[test]
    fn assembler_stitches_single_cjk_char_when_windows_time_overlap() {
        // firered / mimo goldens on the same fixture: "...我" + "我通常会..."
        let transcription =
            assemble_wordless_overlap("周末的时候，我", "我通常会读书或者看一部电影放松一下。");
        assert_eq!(
            transcription.text,
            "周末的时候，我通常会读书或者看一部电影放松一下。"
        );
    }

    #[test]
    fn assembler_keeps_short_abutting_repetition_without_time_overlap() {
        // Two-unit prefix shared by abutting complete slices must not stitch.
        // Exact duplicates still drop via text-equality; this pair differs.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 0, 16_000),
            text: "今天好".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(1, 16_000, 32_000),
            text: "今天坏".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.text, "今天好今天坏");
    }

    #[test]
    fn assembler_drops_leading_orphan_contained_in_previous_suffix() {
        // 50.5s qwen shape: next slice re-reads a tail-anchored prefix
        // "吃饭。听说…正宗。" (15 units ≥ 2), then continues.
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 0, 16_000 * 50 + 8_000),
            text: "晚上我们还计划去一家新开的川菜馆吃饭。听说那里的麻婆豆腐特别正宗。".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(1, 16_000 * 50, 16_000 * 69),
            text: "吃饭。听说那里的麻婆豆腐特别正宗。周末的时候，我通常会读书或者看一部电影，放松一下。".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let (transcription, stats) = assembler.into_parts();
        assert!(
            !transcription.text.contains("吃饭。吃饭")
                && transcription
                    .text
                    .matches("听说那里的麻婆豆腐特别正宗")
                    .count()
                    == 1,
            "leading 吃饭。 plus the re-read sentence must be dropped, got {:?}",
            transcription.text
        );
        assert!(
            transcription
                .text
                .contains("周末的时候，我通常会读书或者看一部电影，放松一下。"),
            "continuation after the re-read must stay, got {:?}",
            transcription.text
        );
        assert!(
            !transcription
                .segments
                .iter()
                .any(|segment| segment.text.trim() == "吃饭。"),
            "isolated 吃饭。 cue must not survive, got {:#?}",
            transcription.segments
        );
        assert!(
            stats.duplicate_merge_count >= 1,
            "leading re-read trim must count, got {stats:?}"
        );
    }

    #[test]
    fn assembler_stitches_partial_suffix_char_re_read() {
        // moss shape at the same seam: "...吃饭" + "饭，听说..."
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 0, 16_000 * 50 + 8_000),
            text: "晚上我们还计划去一家新开的川菜馆吃饭".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(1, 16_000 * 50, 16_000 * 69),
            text: "饭，听说那里的麻婆豆腐特别正宗。".to_string(),
            segments: Vec::new(),
            time_domain: SegmentTimeDomain::RelativeToSliceContent,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(
            transcription.text.replace(' ', ""),
            "晚上我们还计划去一家新开的川菜馆吃饭，听说那里的麻婆豆腐特别正宗。"
        );
    }

    #[test]
    fn assembler_keeps_wo_when_new_sentence_starts_with_wo() {
        let transcription = assemble_wordless_overlap(
            "今天天气非常好，我打算和朋友们一起去公园散步。",
            "我觉得晚上会下雨。",
        );
        assert!(
            transcription.text.contains("我觉得晚上会下雨"),
            "new-sentence 我 must stay, got {:?}",
            transcription.text
        );
        assert!(!transcription.text.contains("散步。觉得"));
    }

    #[test]
    fn assembler_keeps_jintian_when_it_is_not_the_previous_tail() {
        let transcription = assemble_wordless_overlap("我们今天去公园散步。", "今天晚上吃什么？");
        assert!(
            transcription.text.contains("今天晚上吃什么"),
            "new-sentence 今天 must stay, got {:?}",
            transcription.text
        );
    }

    #[test]
    fn assembler_keeps_english_head_word_that_is_not_the_previous_tail() {
        let transcription = assemble_wordless_overlap(
            "and so my fellow americans ask not what your country can do for you",
            "and then we all went home",
        );
        assert!(
            transcription.text.contains("and then we all went home"),
            "English head-word must stay, got {:?}",
            transcription.text
        );
        assert!(!transcription.text.contains("for you then"));
    }

    #[test]
    fn assembler_does_not_glue_digit_strings_at_a_shared_numeral() {
        let transcription = assemble_wordless_overlap("电话是一三八", "八零零一二三四");
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.text, "电话是一三八八零零一二三四");
    }

    #[test]
    fn assembler_does_not_eat_reduplicated_thanks() {
        let transcription = assemble_wordless_overlap("非常感谢", "谢谢大家");
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(transcription.text, "非常感谢谢谢大家");
    }

    #[test]
    fn assembler_keeps_worded_adjacent_segments_in_the_same_slice() {
        let mut assembler =
            TranscriptAssembler::new(TimelineMap::identity(), SegmentMergePolicy::default());
        assembler.push_slice_result(SliceTranscript {
            slice: energy_slice(0, 0, 16_000 * 6),
            text: "that is how it ended in the end in the end we all agreed".to_string(),
            segments: vec![
                absolute_segment(
                    "that is how it ended in the end",
                    0.0,
                    3.0,
                    vec![
                        word("that", 0.0, 0.3),
                        word("is", 0.3, 0.5),
                        word("how", 0.5, 0.8),
                        word("it", 0.8, 1.0),
                        word("ended", 1.0, 1.5),
                        word("in", 1.6, 1.8),
                        word("the", 1.8, 2.1),
                        word("end", 2.1, 3.0),
                    ],
                ),
                absolute_segment(
                    "in the end we all agreed",
                    3.0,
                    6.0,
                    vec![
                        word("in", 3.0, 3.2),
                        word("the", 3.2, 3.4),
                        word("end", 3.4, 3.8),
                        word("we", 3.8, 4.1),
                        word("all", 4.1, 4.5),
                        word("agreed", 4.5, 6.0),
                    ],
                ),
            ],
            time_domain: SegmentTimeDomain::AbsoluteOriginal,
        });
        let transcription = assembler.into_transcription();
        assert_eq!(transcription.segments.len(), 2);
        assert_eq!(
            transcription.segments[0].text,
            "that is how it ended in the end"
        );
        assert_eq!(transcription.segments[1].text, "in the end we all agreed");
        assert_eq!(transcription.segments[0].words.len(), 8);
        assert_eq!(transcription.segments[1].words.len(), 6);
    }

    #[test]
    fn normalize_words_keeps_ascii_fold_and_splits_cjk() {
        assert_eq!(normalize_words("don't"), vec!["dont".to_string()]);
        assert_eq!(normalize_words("twenty-one"), vec!["twentyone".to_string()]);
        assert_eq!(normalize_words("café"), vec!["caf".to_string()]);
        assert_eq!(
            normalize_words("今天好"),
            vec!["今".to_string(), "天".to_string(), "好".to_string()]
        );
    }

    #[test]
    fn longest_common_window_is_quadratic_for_200_units() {
        let left: Vec<String> = (0..200).map(|index| format!("t{index}")).collect();
        let mut right = left.clone();
        right.rotate_left(40);
        let started = std::time::Instant::now();
        let overlap = longest_common_window_len(&left, &right);
        let elapsed = started.elapsed();
        assert_eq!(overlap, 160);
        assert!(
            elapsed.as_millis() < 200,
            "LCW of 200 units must stay well under 200ms, took {elapsed:?}"
        );
    }
}
