//! Host side of the nemotron CTC word-timestamp aligner.
//!
//! Direct port of the NeMo-Speech.cpp reference `timestamp_alignment.{h,cpp}`
//! (itself from NeMo's `MultiSpeakerSOTWordTimestampAligner`). All functions
//! are pure over host buffers. The float operations reproduce NumPy / PyTorch
//! / SciPy / OpenBLAS semantics; every fused operation uses `f32::mul_add` so
//! the port is bit-comparable to the reference built with `-ffp-contract=off`.

use std::collections::BTreeMap;

use super::ctc_spm::CtcSpm;
use super::ctc_unit_tables;

// ---------------------------------------------------------------------------
// UTF-8 / Unicode helpers
// ---------------------------------------------------------------------------

/// Decode the code point at byte `i` and its byte length. Invalid bytes decode
/// to U+FFFD with length 1 (matching the reference's `code_point_at`).
fn code_point_at(bytes: &[u8], i: usize) -> (u32, usize) {
    let b0 = bytes[i];
    if b0 < 0x80 {
        return (u32::from(b0), 1);
    }
    let (n, mut cp) = if (b0 & 0xe0) == 0xc0 && b0 >= 0xc2 {
        (2usize, u32::from(b0 & 0x1f))
    } else if (b0 & 0xf0) == 0xe0 {
        (3usize, u32::from(b0 & 0x0f))
    } else if (b0 & 0xf8) == 0xf0 && b0 <= 0xf4 {
        (4usize, u32::from(b0 & 0x07))
    } else {
        return (0xfffd, 1);
    };
    if i + n > bytes.len() {
        return (0xfffd, 1);
    }
    for &b in &bytes[i + 1..i + n] {
        if b & 0xc0 != 0x80 {
            return (0xfffd, 1);
        }
        cp = (cp << 6) | u32::from(b & 0x3f);
    }
    if (n == 3 && (cp < 0x800 || (0xd800..=0xdfff).contains(&cp)))
        || (n == 4 && !(0x10000..=0x10ffff).contains(&cp))
    {
        return (0xfffd, 1);
    }
    (cp, n)
}

/// Python `str.isspace()`.
fn python_space(cp: u32) -> bool {
    (0x09..=0x0d).contains(&cp)
        || (0x1c..=0x20).contains(&cp)
        || cp == 0x85
        || cp == 0xa0
        || cp == 0x1680
        || (0x2000..=0x200a).contains(&cp)
        || cp == 0x2028
        || cp == 0x2029
        || cp == 0x202f
        || cp == 0x205f
        || cp == 0x3000
}

const DIGIT_ZEROS: &[u32] = &[
    0x30, 0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6, 0xd66,
    0xe50, 0xed0, 0xf20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0, 0x1a80, 0x1a90, 0x1b50,
    0x1bb0, 0x1c40, 0x1c50, 0xa620, 0xa8d0, 0xa900, 0xa9d0, 0xa9f0, 0xaa50, 0xabf0, 0xff10,
    0x104a0, 0x10d30, 0x11066, 0x110f0, 0x11136, 0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650,
    0x116c0, 0x11730, 0x118e0, 0x11950, 0x11c50, 0x11d50, 0x11da0, 0x11f50, 0x16a60, 0x16ac0,
    0x16b50, 0x1d7ce, 0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e950,
    0x1fbf0,
];

fn digit_value(cp: u32) -> i32 {
    let idx = DIGIT_ZEROS.partition_point(|&z| z <= cp);
    if idx == 0 {
        return -1;
    }
    let zero = DIGIT_ZEROS[idx - 1];
    if cp - zero < 10 {
        (cp - zero) as i32
    } else {
        -1
    }
}

/// `<spk:(\d+)>` case-insensitively (with Python's s/k Unicode folds) at
/// `bytes[i]`; returns (end byte, saturated tag number).
fn speaker_tag_at(bytes: &[u8], i: usize) -> Option<(usize, i32)> {
    if bytes[i] != b'<' {
        return None;
    }
    let mut j = i + 1;
    let next = |j: &mut usize| -> u32 {
        if *j >= bytes.len() {
            return 0;
        }
        let (cp, len) = code_point_at(bytes, *j);
        *j += len;
        cp
    };
    let cp = next(&mut j);
    if cp != b's' as u32 && cp != b'S' as u32 && cp != 0x17f {
        return None;
    }
    let cp = next(&mut j);
    if cp != b'p' as u32 && cp != b'P' as u32 {
        return None;
    }
    let cp = next(&mut j);
    if cp != b'k' as u32 && cp != b'K' as u32 && cp != 0x212a {
        return None;
    }
    if next(&mut j) != b':' as u32 {
        return None;
    }
    let mut value: i64 = 0;
    let mut digits = 0;
    loop {
        let at = j;
        let cp = next(&mut j);
        let d = digit_value(cp);
        if d < 0 {
            j = at;
            break;
        }
        value = (value * 10 + i64::from(d)).min(i64::from(i32::MAX));
        digits += 1;
    }
    if digits == 0 || j >= bytes.len() || bytes[j] != b'>' {
        return None;
    }
    Some((j + 1, value as i32))
}

fn speaker_tag_token(token: &str) -> bool {
    let bytes = token.as_bytes();
    speaker_tag_at(bytes, 0).is_some_and(|(end, _)| end == bytes.len())
}

/// `<(?:ev|sty):[^<>]*>` over the whole token.
fn event_or_style_token(token: &str) -> bool {
    let prefix = if token.starts_with("<ev:") {
        4
    } else if token.starts_with("<sty:") {
        5
    } else {
        0
    };
    if prefix == 0 || token.len() <= prefix || !token.ends_with('>') {
        return false;
    }
    token[prefix..].find(['<', '>']) == Some(token.len() - 1 - prefix)
}

fn join_words(words: &[String]) -> String {
    words.join(" ")
}

/// Python `str.split()` over Unicode whitespace.
pub(crate) fn split_whitespace(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut words = Vec::new();
    let mut current = String::new();
    let mut i = 0;
    while i < bytes.len() {
        let (cp, len) = code_point_at(bytes, i);
        if python_space(cp) {
            if !current.is_empty() {
                words.push(std::mem::take(&mut current));
            }
        } else {
            current.push_str(std::str::from_utf8(&bytes[i..i + len]).unwrap_or("\u{fffd}"));
        }
        i += len;
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

pub(crate) fn clean_sot_transcript(transcript: &str, special_tokens: &[String]) -> String {
    let mut text = transcript.to_string();
    for token in special_tokens {
        if token.is_empty() || speaker_tag_token(token) || event_or_style_token(token) {
            continue;
        }
        let mut replaced = String::new();
        let mut cursor = 0;
        while let Some(rel) = text[cursor..].find(token) {
            let at = cursor + rel;
            replaced.push_str(&text[cursor..at]);
            replaced.push(' ');
            cursor = at + token.len();
        }
        replaced.push_str(&text[cursor..]);
        text = replaced;
    }
    join_words(&split_whitespace(&text))
}

// ---------------------------------------------------------------------------
// Transcript units
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UnitKind {
    Word,
    Character,
    Cluster,
}

#[derive(Clone, Debug)]
pub(crate) struct TranscriptWord {
    pub(crate) word: String,
    pub(crate) tag: i32,
    pub(crate) index: usize,
    pub(crate) kind: UnitKind,
    pub(crate) display: String,
    #[allow(dead_code)] // reference unit shape; consumed by the plugin transport
    pub(crate) joins_previous: bool,
}

fn parse_sot_words(transcript: &str) -> Vec<TranscriptWord> {
    // Em dashes (U+2014) separate words.
    let mut text = String::new();
    let mut cursor = 0;
    while let Some(rel) = transcript[cursor..].find('\u{2014}') {
        let at = cursor + rel;
        text.push_str(&transcript[cursor..at]);
        text.push(' ');
        cursor = at + '\u{2014}'.len_utf8();
    }
    text.push_str(&transcript[cursor..]);

    // re.split: parts between tags and the tags themselves.
    let bytes = text.as_bytes();
    let mut parts: Vec<(String, i32)> = Vec::new(); // text, tag in force after it (-2 = none)
    let mut part_start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if let Some((end, tag)) = speaker_tag_at(bytes, i) {
            parts.push((text[part_start..i].to_string(), tag));
            part_start = end;
            i = end;
        } else {
            let (_, len) = code_point_at(bytes, i);
            i += len;
        }
    }
    parts.push((text[part_start..].to_string(), -2));

    let mut tag = if parts.len() == 1 { 0 } else { -1 };
    let mut words = Vec::new();
    for (part, next) in parts {
        for text_word in split_whitespace(&part) {
            words.push(TranscriptWord {
                word: text_word,
                tag,
                index: words.len(),
                kind: UnitKind::Word,
                display: String::new(),
                joins_previous: false,
            });
        }
        tag = next;
    }
    words
}

#[allow(dead_code)] // reference API; kept for parity with the aligner result plumbing
fn speaker_label(tag: i32) -> String {
    if tag < 0 {
        "unknown".to_string()
    } else {
        format!("spk:{tag}")
    }
}

fn in_ranges(table: &[(u32, u32)], cp: u32) -> bool {
    let idx = table.partition_point(|r| r.0 <= cp);
    idx > 0 && cp <= table[idx - 1].1
}

fn code_points(text: &str) -> Vec<u32> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let (cp, len) = code_point_at(bytes, i);
        out.push(cp);
        i += len;
    }
    out
}

fn append_utf8(out: &mut String, cp: u32) {
    if let Some(ch) = char::from_u32(cp) {
        out.push(ch);
    } else {
        out.push('\u{fffd}');
    }
}

/// NeMo's event tag: `<[^<>]+>` over the whole word.
fn event_tag(word: &str) -> bool {
    let bytes = word.as_bytes();
    word.len() >= 3
        && bytes[0] == b'<'
        && bytes[word.len() - 1] == b'>'
        && word[1..].find(['<', '>']) == Some(word.len() - 1 - 1)
}

fn joined(parts: &[String]) -> String {
    parts.concat()
}

fn lexical_characters(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for cp in code_points(text) {
        if in_ranges(ctc_unit_tables::LEXICAL, cp) {
            let mut s = String::new();
            append_utf8(&mut s, cp);
            out.push(s);
        }
    }
    out
}

fn timestamp_script(words: &[TranscriptWord]) -> String {
    let (mut ko, mut ja, mut zh, mut th) = (false, false, false, false);
    for w in words {
        if event_tag(&w.word) {
            continue;
        }
        for cp in code_points(&w.word) {
            ko |= in_ranges(ctc_unit_tables::HANGUL, cp);
            ja |= in_ranges(ctc_unit_tables::KANA, cp);
            zh |= in_ranges(ctc_unit_tables::HAN, cp);
            th |= in_ranges(ctc_unit_tables::THAI, cp);
        }
    }
    if ko {
        "ko"
    } else if ja {
        "ja"
    } else if zh {
        "zh"
    } else if th {
        "th"
    } else {
        ""
    }
    .to_string()
}

fn timestamp_units(words: Vec<TranscriptWord>) -> Vec<TranscriptWord> {
    let script = timestamp_script(&words);
    if script != "zh" && script != "ja" && script != "th" {
        return words;
    }
    let cluster = script == "th";
    let mut out: Vec<TranscriptWord> = Vec::new();
    let mut pending = String::new();
    let start_unit =
        |out: &mut Vec<TranscriptWord>, pending: &mut String, mut unit: TranscriptWord| {
            if !pending.is_empty() {
                unit.display = format!("{pending} {}", unit.display);
            }
            pending.clear();
            unit.index = out.len();
            out.push(unit);
        };
    for word in words {
        if event_tag(&word.word) {
            let mut tag = word.clone();
            tag.kind = UnitKind::Word;
            tag.display = word.word.clone();
            start_unit(&mut out, &mut pending, tag);
            continue;
        }
        let first = out.len();
        let mut prefix = String::new();
        for cp in code_points(&word.word) {
            let mut character = String::new();
            append_utf8(&mut character, cp);
            if cluster && out.len() > first && in_ranges(ctc_unit_tables::MARK, cp) {
                out.last_mut().unwrap().word.push_str(&character);
                out.last_mut().unwrap().display.push_str(&character);
            } else if in_ranges(ctc_unit_tables::LEXICAL, cp) {
                let joins_previous = out.len() > first;
                start_unit(
                    &mut out,
                    &mut pending,
                    TranscriptWord {
                        word: character.clone(),
                        tag: word.tag,
                        index: 0,
                        kind: if cluster {
                            UnitKind::Cluster
                        } else {
                            UnitKind::Character
                        },
                        display: format!("{prefix}{character}"),
                        joins_previous,
                    },
                );
                prefix.clear();
            } else if out.len() > first {
                out.last_mut().unwrap().display.push_str(&character);
            } else {
                prefix.push_str(&character);
            }
        }
        if out.len() == first {
            if out.is_empty() {
                if !pending.is_empty() {
                    pending.push(' ');
                }
                pending.push_str(&prefix);
            } else {
                let last = out.last_mut().unwrap();
                last.display.push(' ');
                last.display.push_str(&prefix);
            }
        }
    }
    for unit in &mut out {
        if unit.display == unit.word {
            unit.display.clear();
        }
    }
    out
}

// ---------------------------------------------------------------------------
// CTC targets
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) struct SpeakerStream {
    pub(crate) tag: i32,
    pub(crate) words: Vec<usize>,
    pub(crate) labels: Vec<u32>,
    pub(crate) state_to_word: Vec<i32>,
}

#[derive(Clone, Debug)]
pub(crate) struct AlignmentTargets {
    pub(crate) words: Vec<TranscriptWord>,
    pub(crate) streams: Vec<SpeakerStream>,
    pub(crate) columns: Vec<u32>,
}

fn character_token_ids(
    spm: &CtcSpm,
    unit: &str,
    blank: u32,
    normalized: bool,
) -> Result<Vec<u32>, String> {
    let mut ids = Vec::new();
    let own = spm.piece_to_id(unit);
    if own != spm.unk_id() {
        ids.push(own);
    } else {
        let encoded = spm.encode(unit);
        for id in encoded {
            if spm.id_to_piece(id) != "\u{2581}" {
                ids.push(id);
            }
        }
    }
    let ok = !ids.is_empty() && ids.iter().all(|&id| id != spm.unk_id() && id < blank);
    let decoded = if ok { Some(spm.decode(&ids)) } else { None };
    let expected = if normalized {
        // The tokenizer's normalization (nmt_nfkc). For the lexical characters
        // this model handles, NFKC is the identity; the reference only needs
        // this to compare SARA AM decomposition for Thai clusters.
        joined(&lexical_characters(unit))
    } else {
        unit.to_string()
    };
    let decoded_ok = decoded
        .as_ref()
        .is_some_and(|d| joined(&lexical_characters(d)) == expected);
    if !ok || !decoded_ok {
        return Err(format!("Unrepresentable character: '{unit}'"));
    }
    Ok(ids)
}

fn encode_stream(spm: &CtcSpm, text: &str) -> Result<Vec<u32>, String> {
    let ids = spm.encode(text);
    if ids.is_empty() {
        return Err("SentencePiece could not encode the transcript".to_string());
    }
    Ok(ids)
}

/// NeMo's `_tokenize_word_stream` over the units of one speaker stream.
fn tokenize_stream(
    spm: &CtcSpm,
    units: &[TranscriptWord],
    members: &[usize],
    blank: u32,
) -> Result<Vec<Vec<u32>>, String> {
    let mut ids_by_word: Vec<Vec<u32>> = Vec::new();
    if members.iter().any(|&w| units[w].kind != UnitKind::Word) {
        for &w in members {
            if units[w].kind != UnitKind::Word {
                ids_by_word.push(character_token_ids(
                    spm,
                    &units[w].word,
                    blank,
                    units[w].kind == UnitKind::Cluster,
                )?);
            } else {
                ids_by_word.push(tokenize_stream(spm, units, &[w], blank)?.remove(0));
            }
        }
        return Ok(ids_by_word);
    }

    let words: Vec<String> = members.iter().map(|&w| units[w].word.clone()).collect();
    let stream = encode_stream(spm, &join_words(&words))?;
    if stream.iter().any(|&id| id >= blank) {
        return Err("Tokenizer produced IDs outside the non-blank CTC vocabulary.".to_string());
    }
    let mut flat = Vec::new();
    for word in &words {
        let ids = encode_stream(spm, word)?;
        flat.extend_from_slice(&ids);
        ids_by_word.push(ids);
    }
    if flat != stream {
        return Err("Per-word and full-stream SentencePiece tokenization disagree.".to_string());
    }
    Ok(ids_by_word)
}

pub(crate) fn minimum_frames(labels: &[u32]) -> i32 {
    let mut frames = 0;
    let mut s = 1usize;
    while s < labels.len() {
        frames += 1;
        if s >= 3 && labels[s] == labels[s - 2] {
            frames += 1;
        }
        s += 2;
    }
    frames
}

pub(crate) fn alignment_targets(
    spm: &CtcSpm,
    transcript: &str,
    blank: u32,
    ctc_frames: usize,
) -> Result<AlignmentTargets, String> {
    let mut out = AlignmentTargets {
        words: timestamp_units(parse_sot_words(transcript)),
        streams: Vec::new(),
        columns: Vec::new(),
    };
    for w in 0..out.words.len() {
        let tag = out.words[w].tag;
        if let Some(stream) = out.streams.iter_mut().find(|s| s.tag == tag) {
            stream.words.push(w);
        } else {
            let mut stream = SpeakerStream {
                tag,
                words: Vec::new(),
                labels: Vec::new(),
                state_to_word: Vec::new(),
            };
            stream.words.push(w);
            out.streams.push(stream);
        }
    }

    // Every stream is tokenized before any target is built.
    let mut per_stream: Vec<Vec<Vec<u32>>> = Vec::with_capacity(out.streams.len());
    for stream in &out.streams {
        per_stream.push(tokenize_stream(spm, &out.words, &stream.words, blank)?);
    }

    let mut tokens: Vec<u32> = Vec::new();
    for (s, stream) in out.streams.iter_mut().enumerate() {
        stream.labels.push(blank);
        stream.state_to_word.push(-1);
        for (w, ids) in per_stream[s].iter().enumerate() {
            for &id in ids {
                stream.labels.push(id);
                stream.labels.push(blank);
                stream.state_to_word.push(w as i32);
                stream.state_to_word.push(-1);
                tokens.push(id);
            }
        }
        if minimum_frames(&stream.labels) as usize > ctc_frames {
            let name = if stream.tag < 0 {
                "None".to_string()
            } else {
                stream.tag.to_string()
            };
            return Err(format!(
                "Speaker {name} transcript is too long for the CTC timeline."
            ));
        }
    }
    tokens.sort_unstable();
    tokens.dedup();
    tokens.retain(|&t| t != blank);

    out.columns.push(blank);
    out.columns.extend_from_slice(&tokens);
    for stream in &mut out.streams {
        for label in &mut stream.labels {
            *label = if *label == blank {
                0
            } else {
                1 + tokens.binary_search(label).unwrap_or(0) as u32
            };
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Head windows
// ---------------------------------------------------------------------------

#[allow(dead_code)] // reference head-window API (paged head path)
#[derive(Clone, Copy, Debug)]
pub(crate) struct HeadWindow {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) context_start: usize,
    pub(crate) context_end: usize,
}

#[allow(dead_code)] // reference head-window API
pub(crate) fn head_windows(
    frames: usize,
    core: usize,
    left: usize,
    right: usize,
) -> Vec<HeadWindow> {
    let mut windows = Vec::new();
    if frames == 0 {
        return windows;
    }
    if core == 0 || frames <= core {
        return vec![HeadWindow {
            start: 0,
            end: frames,
            context_start: 0,
            context_end: frames,
        }];
    }
    let mut start = 0;
    while start < frames {
        let end = (start + core).min(frames);
        windows.push(HeadWindow {
            start,
            end,
            context_start: start.saturating_sub(left),
            context_end: (end + right).min(frames),
        });
        start += core;
    }
    windows
}

// ---------------------------------------------------------------------------
// Numerics (NumPy / PyTorch / SciPy / OpenBLAS)
// ---------------------------------------------------------------------------

/// NumPy's AVX2 `log` of a float32; exact rational minimax polynomial with FMA.
// The polynomial coefficients are copied verbatim from NumPy: the extra
// decimal digits select specific f32 bit patterns, so they must not be
// rounded by a lint.
#[allow(clippy::excessive_precision)]
pub(crate) fn numpy_log(x: f32) -> f32 {
    const P: [f32; 6] = [
        0.0,
        9.999999999999998702521e-01,
        2.112677543073053063722e+00,
        1.480000633576506585156e+00,
        3.808837741388407920751e-01,
        2.589979117907922693523e-02,
    ];
    const Q: [f32; 6] = [
        1.0,
        2.612677543073109236779e+00,
        2.453006071784736363091e+00,
        9.864942958519418960339e-01,
        1.546476374983906719538e-01,
        5.875095259e-03,
    ];
    const LN2: f32 = std::f32::consts::LN_2;
    const SQRT_HALF: f32 = std::f32::consts::FRAC_1_SQRT_2;
    if x.is_nan() {
        return x;
    }
    if x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 {
        return f32::NEG_INFINITY;
    }
    if x.is_infinite() {
        return x;
    }
    let mut x = x;
    let denormal = x < f32::MIN_POSITIVE;
    if denormal {
        x *= f32::from_bits(0x7180_0000); // 2^100
    }
    let mut exponent = ((x.to_bits() >> 23) as i32 - 0x7e) as f32;
    if denormal {
        exponent -= 100.0;
    }
    let mut m = f32::from_bits((x.to_bits() & 0x007f_ffff) | (126u32 << 23));
    if m <= SQRT_HALF {
        m += m;
        exponent -= 1.0;
    }
    m -= 1.0;
    let mut num = m.mul_add(P[5], P[4]);
    let mut den = m.mul_add(Q[5], Q[4]);
    for k in (0..=3).rev() {
        num = num.mul_add(m, P[k]);
        den = den.mul_add(m, Q[k]);
    }
    exponent.mul_add(LN2, num / den)
}

/// `torch.linspace(start, end, steps)[idx]` in float32 as PyTorch's CPU kernel.
pub(crate) fn torch_linspace(start: f32, end: f32, steps: usize, idx: usize) -> f32 {
    if steps == 1 {
        return start;
    }
    let step = (end - start) / (steps - 1) as f32;
    if idx < steps / 2 {
        start + step * idx as f32
    } else {
        (-step).mul_add((steps - idx - 1) as f32, end)
    }
}

/// NumPy's pairwise summation over a contiguous f64 slice.
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        return a.iter().sum();
    }
    if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        let stop = n - n % 8;
        while i < stop {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

/// OpenBLAS Haswell `dgemm` dot product.
fn blas_dot(a: &[f64], b: &[f64]) -> f64 {
    const KQ: usize = 256;
    const KUNROLL: usize = 4;
    let n = a.len();
    let mut c = 0.0;
    let mut ls = 0;
    while ls < n {
        let mut block = n - ls;
        if block >= 2 * KQ {
            block = KQ;
        } else if block > KQ {
            block = (block / 2).div_ceil(KUNROLL) * KUNROLL;
        }
        let mut acc = 0.0;
        for k in ls..ls + block {
            acc = a[k].mul_add(b[k], acc);
        }
        c = if ls != 0 { c + acc } else { acc };
        ls += block;
    }
    c
}

/// NeMo's `_resample_speaker_probs` (clamp, then torch.linspace interpolation).
pub(crate) fn resample_speaker_probs(
    probs: &[f32],
    rows: usize,
    speakers: usize,
    frames: usize,
) -> Vec<f32> {
    if rows < 1 {
        return vec![0.0; frames * speakers];
    }
    let clamped: Vec<f32> = probs
        .iter()
        .take(rows.saturating_mul(speakers))
        .map(|&v| v.clamp(0.0, 1.0))
        .collect();
    if rows == frames {
        return clamped;
    }
    let mut out = vec![0.0f32; frames * speakers];
    for t in 0..frames {
        let position = torch_linspace(0.0, (rows - 1) as f32, frames, t);
        let lower_f = position.floor();
        let upper_f = position.ceil();
        let fraction = position - lower_f;
        let lower = lower_f as usize;
        let upper = upper_f as usize;
        for c in 0..speakers {
            let a = clamped[lower * speakers + c] * (1.0 - fraction);
            let b = clamped[upper * speakers + c] * fraction;
            out[t * speakers + c] = if rows == 1 { clamped[c] } else { a + b };
        }
    }
    out
}

/// SciPy's `linear_sum_assignment` (rectangular shortest augmenting path).
pub(crate) fn linear_sum_assignment(cost_in: &[f64], nr: usize, nc: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if nr == 0 || nc == 0 {
        return out;
    }
    let transpose = nc < nr;
    let (mut rows, mut cols) = (nr, nc);
    let mut cost = cost_in.to_vec();
    if transpose {
        for i in 0..nr {
            for j in 0..nc {
                cost[j * nr + i] = cost_in[i * nc + j];
            }
        }
        std::mem::swap(&mut rows, &mut cols);
    }
    let inf = f64::INFINITY;
    let mut u = vec![0.0; rows];
    let mut v = vec![0.0; cols];
    let mut shortest = vec![0.0; cols];
    let mut path = vec![0usize.wrapping_sub(1); cols];
    let mut col4row = vec![usize::MAX; rows];
    let mut row4col = vec![usize::MAX; cols];
    let mut remaining = vec![0usize; cols];
    let mut sr = vec![false; rows];
    let mut sc = vec![false; cols];
    for cur in 0..rows {
        let mut min_val = 0.0;
        let mut i = cur;
        let mut sink = usize::MAX;
        let mut num_remaining = cols;
        for (it, r) in remaining.iter_mut().enumerate() {
            *r = cols - it - 1;
        }
        sr.fill(false);
        sc.fill(false);
        shortest.fill(inf);
        while sink == usize::MAX {
            let mut index = usize::MAX;
            let mut lowest = inf;
            sr[i] = true;
            for (it, &j) in remaining[..num_remaining].iter().enumerate() {
                let r = min_val + cost[i * cols + j] - u[i] - v[j];
                if r < shortest[j] {
                    path[j] = i;
                    shortest[j] = r;
                }
                if shortest[j] < lowest || (shortest[j] == lowest && row4col[j] == usize::MAX) {
                    lowest = shortest[j];
                    index = it;
                }
            }
            min_val = lowest;
            let j = remaining[index];
            if row4col[j] == usize::MAX {
                sink = j;
            } else {
                i = row4col[j];
            }
            sc[j] = true;
            remaining[index] = remaining[num_remaining - 1];
            num_remaining -= 1;
        }
        u[cur] += min_val;
        for r in 0..rows {
            if sr[r] && r != cur {
                u[r] += min_val - shortest[col4row[r]];
            }
        }
        for c in 0..cols {
            if sc[c] {
                v[c] -= min_val - shortest[c];
            }
        }
        let mut j = sink;
        loop {
            let r = path[j];
            row4col[j] = r;
            std::mem::swap(&mut col4row[r], &mut j);
            if r == cur {
                break;
            }
        }
    }
    if transpose {
        let mut order: Vec<usize> = (0..cols).collect();
        order.sort_by_key(|&k| col4row[k]);
        for k in order {
            out.push((col4row[k], k));
        }
    } else {
        for (r, &c) in col4row.iter().enumerate() {
            out.push((r, c));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Speaker mapping
// ---------------------------------------------------------------------------

/// NeMo's `_one_pass_speaker_mapping`: map each speaker stream to a Sortformer
/// column by IDF-weighted cosine similarity of sqrt token counts, minus a
/// log-total-count penalty.
pub(crate) fn speaker_mapping(
    greedy: &[u32],
    columns: usize,
    streams: &[SpeakerStream],
    speaker: &[f32],
    _frames: usize,
    speakers: usize,
    threshold: f32,
) -> Vec<i32> {
    let mut mapping = vec![-1i32; streams.len()];
    if streams.is_empty() || speakers == 0 {
        return mapping;
    }
    let (c, s, n) = (columns, speakers, streams.len());
    let mut observed = vec![0i64; s * c];
    let mut expected = vec![0i64; n * c];
    for (t, &g) in greedy.iter().enumerate() {
        let g = g as usize;
        if g == 0 || (t > 0 && greedy[t - 1] as usize == g) {
            continue;
        }
        for ch in 0..s {
            if speaker[t * s + ch] >= threshold {
                observed[ch * c + g] += 1;
            }
        }
    }
    for (st, stream) in streams.iter().enumerate() {
        let mut k = 1;
        while k < stream.labels.len() {
            expected[st * c + stream.labels[k] as usize] += 1;
            k += 2;
        }
    }
    let mut idf = vec![0.0f64; c];
    for k in 0..c {
        let mut present = 0i64;
        for ch in 0..s {
            present += i64::from(observed[ch * c + k] > 0);
        }
        idf[k] = (s as f64 / (1 + present) as f64).ln_1p();
    }
    let mut a = vec![0.0f64; n * c];
    let mut b = vec![0.0f64; s * c];
    let mut squares = vec![0.0f64; c];
    for st in 0..n {
        for k in 0..c {
            a[st * c + k] = (expected[st * c + k] as f64 * idf[k]).sqrt();
        }
    }
    for ch in 0..s {
        for k in 0..c {
            b[ch * c + k] = (observed[ch * c + k] as f64 * idf[k]).sqrt();
        }
    }
    let norm = |row: &[f64], squares: &mut [f64]| {
        for k in 0..c {
            squares[k] = row[k] * row[k];
        }
        pairwise_sum(squares).sqrt()
    };
    let mut norm_b = vec![0.0f64; s];
    let mut total_b = vec![0i64; s];
    for ch in 0..s {
        norm_b[ch] = norm(&b[ch * c..(ch + 1) * c], &mut squares);
        for k in 0..c {
            total_b[ch] += observed[ch * c + k];
        }
    }
    let mut cost = vec![0.0f64; n * s];
    for st in 0..n {
        let norm_a = norm(&a[st * c..(st + 1) * c], &mut squares);
        let mut total_a = 0i64;
        for k in 0..c {
            total_a += expected[st * c + k];
        }
        for ch in 0..s {
            let mut similarity = blas_dot(&a[st * c..(st + 1) * c], &b[ch * c..(ch + 1) * c])
                / (1e-8f64).max(norm_a * norm_b[ch]);
            similarity -= 0.1 * ((total_a + 1) as f64 / (total_b[ch] + 1) as f64).ln().abs();
            cost[st * s + ch] = -similarity;
        }
    }
    for (row, col) in linear_sum_assignment(&cost, n, s) {
        mapping[row] = col as i32;
    }
    mapping
}

// ---------------------------------------------------------------------------
// Banded CTC search
// ---------------------------------------------------------------------------

fn trigram_positions(tokens: &[u32]) -> BTreeMap<[u32; 3], (usize, usize)> {
    let mut table = BTreeMap::new();
    for i in 0..tokens.len().saturating_sub(2) {
        let key = [tokens[i], tokens[i + 1], tokens[i + 2]];
        let entry = table.entry(key).or_insert((i, 0));
        entry.1 += 1;
    }
    table
}

/// NeMo's `_ctc_anchor_band_starts`.
fn anchor_band_starts(
    greedy: &[u32],
    target: &[u32],
    states: i64,
    half: i64,
    width: i64,
) -> Vec<i64> {
    let frames = greedy.len() as i64;
    let mut observed = Vec::new();
    let mut times = Vec::new();
    for (i, &g) in greedy.iter().enumerate() {
        if g != 0 && (i == 0 || greedy[i - 1] != g) {
            observed.push(g);
            times.push(i as i64);
        }
    }
    let left = trigram_positions(&observed);
    let right = trigram_positions(target);
    let mut pairs: Vec<(i64, i64)> = Vec::new();
    for (key, position) in &left {
        if position.1 != 1 {
            continue;
        }
        if let Some(rp) = right.get(key).filter(|rp| rp.1 == 1) {
            pairs.push((times[position.0 + 1], 2 * (rp.0 as i64 + 1) + 1));
        }
    }
    pairs.sort_unstable();
    // Longest chain increasing in state (patience sort with parents).
    let mut tails: Vec<i64> = Vec::new();
    let mut tail_ids: Vec<usize> = Vec::new();
    let mut parent: Vec<i64> = Vec::with_capacity(pairs.len());
    for (i, pair) in pairs.iter().enumerate() {
        let j = tails.partition_point(|&t| t < pair.1);
        parent.push(if j != 0 { tail_ids[j - 1] as i64 } else { -1 });
        if j == tails.len() {
            tails.push(pair.1);
            tail_ids.push(i);
        } else {
            tails[j] = pair.1;
            tail_ids[j] = i;
        }
    }
    let mut chosen: Vec<(i64, i64)> = Vec::new();
    let mut k = tail_ids.last().map(|&id| id as i64).unwrap_or(-1);
    while k >= 0 {
        chosen.push(pairs[k as usize]);
        k = parent[k as usize];
    }
    chosen.reverse();

    let mut xs = vec![0.0f64];
    let mut ys = vec![0.0f64];
    for (t, st) in chosen {
        if t > 0 && t < frames - 1 {
            xs.push(t as f64);
            ys.push(st as f64);
        }
    }
    xs.push((frames - 1) as f64);
    ys.push((states - 1) as f64);

    let mut starts = vec![0i64; frames as usize];
    let n = xs.len();
    for t in 0..frames {
        let x = t as f64;
        let j = xs.partition_point(|&v| v <= x) - 1;
        let center = if j == n - 1 || xs[j] == x {
            ys[j]
        } else {
            (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j]) * (x - xs[j]) + ys[j]
        };
        let start = center.round_ties_even() as i64 - half;
        starts[t as usize] = start.clamp(0, states - width);
    }
    let mut running = i64::MAX;
    for t in 0..frames {
        running = running.min(starts[t as usize] - 2 * t);
        // max(running + 2t, max(0, states - width - 2*(frames-1-t)))
        starts[t as usize] = (running + 2 * t).max(0.max(states - width - 2 * (frames - 1 - t)));
    }
    starts
}

#[derive(Debug)]
pub(crate) struct BandedPath {
    pub(crate) path: Vec<i32>,
    pub(crate) score: f32,
}

/// NeMo's `_banded_ctc_search`: max-sum CTC in an anchor-centred band.
pub(crate) fn banded_ctc_search(
    log_probs: &[f32],
    columns: usize,
    labels: &[u32],
    positions: &[usize],
    prior: Option<&[f32]>,
    greedy: &[u32],
) -> BandedPath {
    let neg_inf = f32::NEG_INFINITY;
    let frames = positions.len() as i64;
    let states = labels.len() as i64;
    let mut out = BandedPath {
        path: vec![-1; frames as usize],
        score: neg_inf,
    };
    let half = if frames < 1000 {
        states
    } else {
        states.min((0.10 * states as f64 + 100.0).ceil() as i64)
    };
    let width = states.min(2 * half + 1);
    let target: Vec<u32> = labels.iter().skip(1).step_by(2).copied().collect();
    let mut needed = target.len() as i64;
    for k in 1..target.len() {
        needed += i64::from(target[k] == target[k - 1]);
    }
    if frames < needed || frames == 0 {
        return out;
    }
    let searched: Vec<u32> = positions.iter().map(|&p| greedy[p]).collect();
    let starts = anchor_band_starts(&searched, &target, states, half, width);
    let row = |t: usize| -> &[f32] {
        let base = positions[t] * columns;
        &log_probs[base..base + columns]
    };

    let w = width as usize;
    let mut previous = vec![neg_inf; w];
    let mut padded = vec![neg_inf; w + 4];
    let row_bytes = w.div_ceil(4);
    let mut pointers = vec![0u8; frames as usize * row_bytes];

    for i in 0..2.min(w) {
        previous[i] = row(0)[labels[i] as usize];
        if let Some(p) = prior.filter(|_| labels[i] != 0) {
            previous[i] += p[positions[0]];
        }
    }
    let mut can_skip = vec![false; states as usize];
    for s in 2..states as usize {
        can_skip[s] = labels[s] != 0 && labels[s] != labels[s - 2];
    }

    let padded_get = |padded: &[f32], idx: i64| -> f32 {
        if idx < 0 || idx as usize >= padded.len() {
            neg_inf
        } else {
            padded[idx as usize]
        }
    };

    for t in 1..frames as usize {
        let start = starts[t];
        let delta = start - starts[t - 1];
        padded[2..2 + w].copy_from_slice(&previous);
        let emissions = row(t);
        let p = prior.map(|p| p[positions[t]]).unwrap_or(0.0);
        let back = &mut pointers[t * row_bytes..(t + 1) * row_bytes];
        for i in 0..w {
            let mut best = padded_get(&padded, 2 + delta + i as i64);
            let mut move_code = 0u8;
            let step = padded_get(&padded, 1 + delta + i as i64);
            if step > best {
                best = step;
                move_code = 1;
            }
            let skip = padded_get(&padded, delta + i as i64);
            if skip > best && can_skip[(start + i as i64) as usize] {
                best = skip;
                move_code = 2;
            }
            let label = labels[(start + i as i64) as usize] as usize;
            let mut score = best + emissions[label];
            if prior.is_some() && label != 0 {
                score += p;
            }
            previous[i] = score;
            back[i >> 2] |= move_code << (2 * (i & 3));
        }
    }

    let mut end_state = -1i64;
    for candidate in [states - 1, states - 2] {
        let local = candidate - starts[frames as usize - 1];
        if local < 0 || local >= width {
            continue;
        }
        if end_state < 0 || previous[local as usize] > out.score {
            end_state = candidate;
            out.score = previous[local as usize];
        }
    }
    if end_state < 0 || !out.score.is_finite() {
        out.score = neg_inf;
        return out;
    }

    let mut state = end_state;
    for t in (0..frames as usize).rev() {
        out.path[t] = state as i32;
        if t > 0 {
            let i = (state - starts[t]) as usize;
            let code = (pointers[t * row_bytes + (i >> 2)] >> (2 * (i & 3))) & 3;
            state -= i64::from(code);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Diarization segments and word refinement
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) struct Segment {
    pub(crate) speaker: usize,
    pub(crate) start: f64,
    pub(crate) end: f64,
}

pub(crate) fn diarization_segments(
    activity10: Option<&[f32]>,
    frames: usize,
    speakers: usize,
    frame_seconds: f64,
    duration: f64,
    time_offset: f64,
) -> Vec<Segment> {
    let mut segments = Vec::new();
    let Some(activity) = activity10 else {
        return segments;
    };
    if frames == 0 || speakers == 0 {
        return segments;
    }
    for c in 0..speakers {
        let mut begin: Option<usize> = None;
        for t in 0..=frames {
            let active = t < frames && activity[t * speakers + c] > 0.5;
            if active && begin.is_none() {
                begin = Some(t);
            } else if !active && begin.is_some() {
                let b = begin.unwrap();
                let start = time_offset + b as f64 * frame_seconds;
                let end = (time_offset + t as f64 * frame_seconds).min(time_offset + duration);
                if end > start {
                    segments.push(Segment {
                        speaker: c,
                        start,
                        end,
                    });
                }
                begin = None;
            }
        }
    }
    segments.sort_by(|a, b| {
        a.start
            .total_cmp(&b.start)
            .then(a.end.total_cmp(&b.end))
            .then(a.speaker.cmp(&b.speaker))
    });
    segments
}

#[allow(dead_code)] // diagnostic fields mirror the reference's WordRow shape
#[derive(Clone, Debug)]
pub(crate) struct WordRow {
    pub(crate) word: usize,
    pub(crate) tag: i32,
    pub(crate) start: f64,
    pub(crate) end: f64,
    pub(crate) start_frame: i32,
    pub(crate) end_frame: i32,
    pub(crate) column: i32,
    pub(crate) speaker_confidence: f32,
}

#[derive(Debug, Default)]
pub(crate) struct RefinementStats {
    pub(crate) max_seconds: f64,
    pub(crate) words: i32,
    pub(crate) onsets_moved: i32,
    pub(crate) unmatched_words: i32,
}

pub(crate) fn refine_word_timestamps(
    words: &mut [WordRow],
    mapping: &[(i32, i32)],
    diarization: &[Segment],
    ctc_step: f64,
    max_frames: usize,
) -> Result<RefinementStats, String> {
    if max_frames < 1 {
        return Err("max_frames must be a positive integer".to_string());
    }
    if !ctc_step.is_finite() || ctc_step <= 0.0 {
        return Err("ctc_frame_seconds must be positive and finite".to_string());
    }
    let mut stats = RefinementStats::default();
    let limit = max_frames as f64 * ctc_step.min(0.08);
    stats.max_seconds = limit;

    let mut segs: BTreeMap<usize, Vec<(f64, f64)>> = BTreeMap::new();
    for seg in diarization {
        if seg.end > seg.start {
            segs.entry(seg.speaker)
                .or_default()
                .push((seg.start, seg.end));
        }
    }
    for intervals in segs.values_mut() {
        intervals.sort_by(|a, b| a.0.total_cmp(&b.0));
    }
    let none: Vec<(f64, f64)> = Vec::new();

    let mut first = 0usize;
    while first < words.len() {
        let mut last = first;
        while last < words.len() && words[last].tag == words[first].tag {
            last += 1;
        }
        words[first..last].sort_by(|a, b| a.start.total_cmp(&b.start));
        let column = mapping
            .iter()
            .find(|(tag, _)| *tag == words[first].tag)
            .map(|(_, c)| *c)
            .unwrap_or(-1);
        let intervals: &Vec<(f64, f64)> = if column >= 0 {
            segs.get(&(column as usize)).unwrap_or(&none)
        } else {
            &none
        };
        let n = last - first;
        let mut member = vec![-1i32; n];
        for k in 0..n {
            let start = words[first + k].start;
            let idx = intervals.partition_point(|iv| iv.0 <= start) as i32 - 1;
            member[k] = if idx >= 0 && start < intervals[idx as usize].1 {
                idx
            } else {
                -1
            };
        }
        let mut seen = vec![false; intervals.len()];
        for k in 0..n {
            let m = member[k] as usize;
            if member[k] < 0 {
                stats.unmatched_words += 1;
            } else if !seen[m] {
                let onset = intervals[m].0;
                if onset < words[first + k].start && words[first + k].start - onset <= limit {
                    words[first + k].start = onset;
                    stats.onsets_moved += 1;
                }
                seen[m] = true;
            }
        }
        for k in 0..n {
            let start = words[first + k].start;
            let stop = if member[k] < 0 {
                if k + 1 < n {
                    words[first + k + 1].start
                } else {
                    words[first + k].end
                }
            } else if k + 1 < n && member[k + 1] == member[k] {
                words[first + k + 1].start
            } else {
                intervals[member[k] as usize].1
            };
            words[first + k].end = start.max(stop.min(start + limit));
        }
        stats.words += n as i32;
        first = last;
    }
    Ok(stats)
}

// ---------------------------------------------------------------------------
// align_record / public result
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub(crate) struct AlignmentConfig {
    pub(crate) speaker_activity_threshold: f32,
    pub(crate) speaker_logprob_weight: f32,
    pub(crate) maximum_token_len: f64,
    /// CTC frames; the reference uses 12 encoder frames x upsampling.
    pub(crate) max_word_frames: usize,
    pub(crate) epsilon: f32,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            speaker_activity_threshold: 0.5,
            speaker_logprob_weight: 1.0,
            maximum_token_len: 1.0,
            max_word_frames: 12,
            epsilon: 1.0e-6,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct AlignmentInputs<'a> {
    pub(crate) log_probs: &'a [f32],
    pub(crate) frames: usize,
    pub(crate) speaker_probs: Option<&'a [f32]>,
    pub(crate) speaker_frames: usize,
    pub(crate) speakers: usize,
    pub(crate) activity10: Option<&'a [f32]>,
    pub(crate) label_frames: usize,
    pub(crate) label_speakers: usize,
    pub(crate) label_seconds: f64,
    pub(crate) duration: f64,
    pub(crate) time_offset: f64,
}

#[allow(dead_code)] // diagnostic fields mirror the reference's failure report
#[derive(Debug)]
pub(crate) struct StreamFailure {
    pub(crate) tag: i32,
    pub(crate) reason: String,
    pub(crate) words: Vec<usize>,
}

#[derive(Debug)]
pub(crate) struct AlignmentResult {
    pub(crate) ctc_step: f64,
    pub(crate) mapping: Vec<(i32, i32)>,
    pub(crate) scores: Vec<f32>,
    pub(crate) failed_streams: Vec<StreamFailure>,
    pub(crate) words: Vec<WordRow>,
    pub(crate) diarization: Vec<Segment>,
    pub(crate) refinement: RefinementStats,
}

pub(crate) fn align_record(
    targets: &AlignmentTargets,
    inputs: &AlignmentInputs<'_>,
    config: &AlignmentConfig,
) -> Result<AlignmentResult, String> {
    let frames = inputs.frames;
    let c = targets.columns.len();
    if frames < 1
        || c < 1
        || inputs.log_probs.len() < frames * c
        || inputs.duration.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
    {
        return Err("align_record: invalid CTC inputs".to_string());
    }
    let mut result = AlignmentResult {
        ctc_step: inputs.duration / frames as f64,
        mapping: Vec::new(),
        scores: Vec::new(),
        failed_streams: Vec::new(),
        words: Vec::new(),
        diarization: Vec::new(),
        refinement: RefinementStats::default(),
    };

    // Greedy column per frame (first maximum, as numpy.argmax, NaN-aware).
    let mut greedy = vec![0u32; frames];
    for t in 0..frames {
        let row = &inputs.log_probs[t * c..(t + 1) * c];
        let mut best = 0usize;
        for col in 0..c {
            if row[best].is_nan() {
                break;
            }
            if row[col].is_nan() || row[col] > row[best] {
                best = col;
            }
        }
        greedy[t] = best as u32;
    }

    let mut speaker = Vec::new();
    let mut mapping = vec![-1i32; targets.streams.len()];
    let s = if inputs.speaker_probs.is_some() {
        inputs.speakers
    } else {
        0
    };
    if let Some(probs) = inputs.speaker_probs {
        speaker = resample_speaker_probs(probs, inputs.speaker_frames, s, frames);
        mapping = speaker_mapping(
            &greedy,
            c,
            &targets.streams,
            &speaker,
            frames,
            s,
            config.speaker_activity_threshold,
        );
    }

    let mut positions = Vec::new();
    let mut prior = vec![0.0f32; frames];
    for (stream_idx, stream) in targets.streams.iter().enumerate() {
        let column = mapping[stream_idx];
        result.mapping.push((stream.tag, column));
        positions.clear();
        for t in 0..frames {
            let activity = if inputs.speaker_probs.is_none() {
                1.0
            } else if column < 0 {
                0.0
            } else {
                speaker[t * s + column as usize]
            };
            if activity >= 0.01 {
                positions.push(t);
            }
            prior[t] = config.speaker_logprob_weight * numpy_log(activity.max(config.epsilon));
        }
        let searched = banded_ctc_search(
            inputs.log_probs,
            c,
            &stream.labels,
            &positions,
            inputs.speaker_probs.map(|_| prior.as_slice()),
            &greedy,
        );
        result.scores.push(searched.score);
        if !searched.score.is_finite() {
            result.failed_streams.push(StreamFailure {
                tag: stream.tag,
                reason: "no_legal_path_in_anchor_band".to_string(),
                words: stream.words.clone(),
            });
            continue;
        }
        let mut path = vec![-1i32; frames];
        for (k, &pos) in positions.iter().enumerate() {
            path[pos] = searched.path[k];
        }
        let mut word_frames: Vec<Vec<usize>> = vec![Vec::new(); stream.words.len()];
        for (t, &st) in path.iter().enumerate() {
            if st >= 0 {
                let word = stream.state_to_word[st as usize];
                if word >= 0 {
                    word_frames[word as usize].push(t);
                }
            }
        }
        for (w, &unit) in stream.words.iter().enumerate() {
            let f = &word_frames[w];
            if f.is_empty() {
                return Err(format!(
                    "CTC path did not visit word '{}'.",
                    targets.words[unit].word
                ));
            }
            let start = inputs.time_offset + f[0] as f64 * result.ctc_step;
            let end = (inputs.time_offset + (f[f.len() - 1] + 1) as f64 * result.ctc_step)
                .min(start + config.maximum_token_len);
            let end_frame = ((end - inputs.time_offset) / result.ctc_step).ceil() as i32 - 1;
            result.words.push(WordRow {
                word: unit,
                tag: stream.tag,
                start,
                end,
                start_frame: f[0] as i32,
                end_frame: end_frame.max(f[0] as i32).min(f[f.len() - 1] as i32),
                column,
                speaker_confidence: if inputs.speaker_probs.is_some() && column >= 0 {
                    let sum: f64 = f
                        .iter()
                        .map(|&t| f64::from(speaker[t * s + column as usize]))
                        .sum();
                    (sum / f.len() as f64) as f32
                } else {
                    -1.0
                },
            });
        }
    }

    result.diarization = diarization_segments(
        inputs.activity10,
        inputs.label_frames,
        inputs.label_speakers,
        inputs.label_seconds,
        inputs.duration,
        inputs.time_offset,
    );
    result.refinement = refine_word_timestamps(
        &mut result.words,
        &result.mapping,
        &result.diarization,
        result.ctc_step,
        config.max_word_frames,
    )?;
    Ok(result)
}

/// Python `round(x, decimals)`: nearest, ties to even.
pub(crate) fn round_decimals(x: f64, decimals: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.decimals$}").parse::<f64>().unwrap_or(x)
}

pub(crate) fn public_words(words: &[WordRow]) -> Vec<WordRow> {
    let mut out: Vec<WordRow> = words.to_vec();
    for word in &mut out {
        word.start = round_decimals(word.start, 3);
        word.end = round_decimals(word.end, 3);
    }
    out.sort_by(|a, b| a.start.total_cmp(&b.start).then(a.end.total_cmp(&b.end)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace_split_matches_python_split() {
        assert_eq!(split_whitespace("  hello\tworld\n"), ["hello", "world"]);
        assert_eq!(split_whitespace(""), Vec::<String>::new());
    }

    #[test]
    fn speaker_tags_open_turns_and_label() {
        let words = parse_sot_words("<spk:0> hello <spk:1> world");
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert_eq!(words[0].tag, 0);
        assert_eq!(words[1].word, "world");
        assert_eq!(words[1].tag, 1);
        assert_eq!(speaker_label(-1), "unknown");
        assert_eq!(speaker_label(3), "spk:3");

        let words = parse_sot_words("a <spk:2> b");
        assert_eq!(words[0].tag, -1);
        assert_eq!(words[1].tag, 2);
    }

    #[test]
    fn chinese_transcript_is_split_into_character_units() {
        let words = parse_sot_words("\u{4f60}\u{597d} \u{4e16}\u{754c}");
        let units = timestamp_units(words);
        assert_eq!(units.len(), 4);
        assert!(units.iter().all(|u| u.kind == UnitKind::Character));
        // display == word is cleared; joins_previous marks the 2nd char of a word.
        assert!(
            units.iter().any(|u| u.joins_previous) || units.iter().any(|u| !u.display.is_empty())
        );
        assert_eq!(units[0].index, 0);
        assert_eq!(units[3].index, 3);
    }

    #[test]
    fn head_windows_split_with_context() {
        let w = head_windows(10, 4, 1, 1);
        assert_eq!(w.len(), 3);
        assert_eq!(
            (w[0].start, w[0].end, w[0].context_start, w[0].context_end),
            (0, 4, 0, 5)
        );
        assert_eq!(
            (w[1].start, w[1].end, w[1].context_start, w[1].context_end),
            (4, 8, 3, 9)
        );
        assert_eq!(
            (w[2].start, w[2].end, w[2].context_start, w[2].context_end),
            (8, 10, 7, 10)
        );
        assert_eq!(head_windows(3, 4, 1, 1).len(), 1);
        assert!(head_windows(0, 4, 1, 1).is_empty());
    }

    #[test]
    fn numpy_log_matches_natural_log() {
        for x in [0.5f32, 1.0, 2.0, 10.0, 123.456] {
            let got = numpy_log(x);
            let want = x.ln();
            assert!(
                (got - want).abs() < 2e-6,
                "numpy_log({x}) = {got}, ln = {want}"
            );
        }
        assert_eq!(numpy_log(0.0), f32::NEG_INFINITY);
        assert!(numpy_log(-1.0).is_nan());
    }

    #[test]
    fn linspace_viterbi_and_numerics() {
        for (i, want) in [0.0, 2.5, 5.0, 7.5, 10.0].into_iter().enumerate() {
            assert_eq!(torch_linspace(0.0, 10.0, 5, i), want);
        }
        assert_eq!(pairwise_sum(&[1.0; 100]), 100.0);
        assert_eq!(
            linear_sum_assignment(&[1.0, 2.0, 3.0, 4.0], 2, 2),
            vec![(0, 0), (1, 1)]
        );
        assert_eq!(minimum_frames(&[0, 1, 0, 2, 0]), 2);
        assert_eq!(minimum_frames(&[0, 1, 0, 1, 0]), 3);
        assert_eq!(round_decimals(1.2346, 3), 1.235);
        assert_eq!(round_decimals(1.2344, 3), 1.234);
    }

    #[test]
    fn diarization_runs_and_refinement() {
        let activity = [0.6f32, 0.6, 0.1, 0.8, 0.8];
        let segs = diarization_segments(Some(&activity), 5, 1, 0.01, 0.05, 0.0);
        assert_eq!(segs.len(), 2);
        assert_eq!((segs[0].start, segs[0].end), (0.0, 0.02));
        assert_eq!((segs[1].start, segs[1].end), (0.03, 0.05));
    }

    #[test]
    fn align_record_reports_word_and_failed_stream() {
        let targets = AlignmentTargets {
            words: vec![TranscriptWord {
                word: "a".into(),
                tag: 0,
                index: 0,
                kind: UnitKind::Word,
                display: String::new(),
                joins_previous: false,
            }],
            streams: vec![SpeakerStream {
                tag: 0,
                words: vec![0],
                labels: vec![0, 1, 0],
                state_to_word: vec![-1, 0, -1],
            }],
            columns: vec![0, 1],
        };
        // One successful record: token 1 peaks on frames 2..4.
        let mut lp = vec![0.0f32; 6 * 2];
        for t in 0..6 {
            lp[t * 2] = if (2..4).contains(&t) { -5.0 } else { 5.0 };
            lp[t * 2 + 1] = if (2..4).contains(&t) { 5.0 } else { -5.0 };
        }
        let inputs = AlignmentInputs {
            log_probs: &lp,
            frames: 6,
            speaker_probs: None,
            speaker_frames: 0,
            speakers: 0,
            activity10: None,
            label_frames: 0,
            label_speakers: 0,
            label_seconds: 0.01,
            duration: 0.24,
            time_offset: 0.0,
        };
        let res = align_record(&targets, &inputs, &AlignmentConfig::default()).unwrap();
        assert_eq!(res.words.len(), 1);
        let w = &res.words[0];
        assert_eq!(w.word, 0);
        assert_eq!(w.start_frame, 2);
        assert_eq!(w.start, 0.08);
        assert_eq!(w.column, -1);
        assert_eq!(w.speaker_confidence, -1.0);

        // Too few frames for the target -> stream is reported as failed.
        // Too few frames for a two-token target -> stream is reported as failed.
        let targets_two = AlignmentTargets {
            words: targets.words.clone(),
            streams: vec![SpeakerStream {
                tag: 0,
                words: vec![0],
                labels: vec![0, 1, 0, 2, 0],
                state_to_word: vec![-1, 0, -1, 1, -1],
            }],
            columns: vec![0, 1, 2],
        };
        let lp_two = vec![0.0f32; 3];
        let inputs_short = AlignmentInputs {
            frames: 1,
            duration: 0.04,
            log_probs: &lp_two,
            speaker_probs: None,
            speaker_frames: 0,
            speakers: 0,
            activity10: None,
            label_frames: 0,
            label_speakers: 0,
            label_seconds: 0.01,
            time_offset: 0.0,
        };
        let res = align_record(&targets_two, &inputs_short, &AlignmentConfig::default()).unwrap();
        assert!(res.words.is_empty());
        assert_eq!(res.failed_streams.len(), 1);
        assert_eq!(res.failed_streams[0].tag, 0);
        assert_eq!(res.failed_streams[0].reason, "no_legal_path_in_anchor_band");
        assert_eq!(res.failed_streams[0].words, vec![0]);
    }
}
