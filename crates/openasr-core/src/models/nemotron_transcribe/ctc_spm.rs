//! SentencePiece BPE tokenizer for the nemotron CTC timestamp head.
//!
//! The head's classifier has `ctc.vocab_size` classes: the first
//! `ctc.vocab_size - 1` are the SentencePiece pieces embedded (base64) in the
//! pack's `ctc.tokenizer.spm_model` metadata, and the last is the CTC blank.
//! This module parses that `ModelProto` and reproduces
//! `sentencepiece::SentencePieceProcessor::Encode` for this model exactly:
//!
//! - the model is BPE with synthetic scores `score(id) = min(0, 49 - id)` and
//!   no `bpe_model` merge list; `Encoder::Encode` therefore starts from a
//!   single-codepoint split and greedily merges adjacent symbols whose
//!   concatenation is a vocabulary piece, highest score (= lowest id) first,
//!   ties by leftmost position (`src/bpe_model.cc`, `SampleEncode(alpha=0)`);
//! - the 48 `UNUSED` user symbols (`<spk:N>`, `<ev:...>`, `<sty:...>`) have no
//!   mergeable prefixes in the vocabulary, so the processor's prefix matcher
//!   consumes them atomically;
//! - the processor prepends U+2581 (▁) to each whitespace-separated segment.
//!
//! The algorithms and numerics were cross-checked against the parity oracle
//! (the locally built NeMo-Speech.cpp reference linked with the same
//! SentencePiece library) over a corpus of English, CJK, Thai, punctuation and
//! special-token inputs. This is trust-boundary parsing of pack bytes, so it
//! is fully bounds-checked and fails closed.

use std::collections::HashMap;

const DUMMY_PREFIX: char = '\u{2581}'; // ▁
const MODEL_TYPE_BPE: u32 = 2;

// sentencepiece::ModelProto::SentencePiece::Type
const TYPE_NORMAL: u8 = 1;
const TYPE_USER_DEFINED: u8 = 3;
const TYPE_UNUSED: u8 = 4;

#[derive(Debug, thiserror::Error)]
pub(crate) enum CtcSpmError {
    #[error("CTC SentencePiece proto truncated near byte {at}")]
    Truncated { at: usize },
    #[error("CTC SentencePiece proto has invalid wire type {wire} near byte {at}")]
    BadWire { wire: u32, at: usize },
    #[error("CTC SentencePiece piece {id} is not valid UTF-8")]
    BadUtf8 { id: usize },
    #[error("CTC SentencePiece model_type {got} is not BPE ({want})")]
    UnexpectedModelType { got: u32, want: u32 },
    #[error("CTC SentencePiece piece {id} ('{piece}') is duplicated")]
    DuplicatePiece { id: usize, piece: String },
    #[error("CTC SentencePiece vocabulary is empty")]
    EmptyVocab,
    #[error("CTC SentencePiece vocab has {got} pieces, expected {want}")]
    VocabSizeMismatch { got: usize, want: usize },
    #[error("CTC SentencePiece base64 metadata is invalid")]
    InvalidBase64,
}

/// Minimal RFC 4648 base64 decoder (no external crate). The SPM proto is
/// embedded as base64 in `ctc.tokenizer.spm_model`; malformed input fails
/// closed.
pub(crate) fn decode_base64(s: &str) -> Result<Vec<u8>, CtcSpmError> {
    fn val(b: u8) -> Option<u8> {
        match b {
            b'A'..=b'Z' => Some(b - b'A'),
            b'a'..=b'z' => Some(b - b'a' + 26),
            b'0'..=b'9' => Some(b - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let bytes = s.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err(CtcSpmError::InvalidBase64);
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);

    for chunk in bytes.as_chunks::<4>().0 {
        let a = val(chunk[0]).ok_or(CtcSpmError::InvalidBase64)?;
        let b = val(chunk[1]).ok_or(CtcSpmError::InvalidBase64)?;
        let c = if chunk[2] == b'=' {
            0
        } else {
            val(chunk[2]).ok_or(CtcSpmError::InvalidBase64)?
        };
        let d = if chunk[3] == b'=' {
            0
        } else {
            val(chunk[3]).ok_or(CtcSpmError::InvalidBase64)?
        };
        let triple =
            (u32::from(a) << 18) | (u32::from(b) << 12) | (u32::from(c) << 6) | u32::from(d);
        out.push((triple >> 16) as u8);
        if chunk[2] != b'=' {
            out.push((triple >> 8) as u8);
        }
        if chunk[3] != b'=' {
            out.push(triple as u8);
        }
    }
    Ok(out)
}

/// Minimal, bounds-checked protobuf reader for the fields this module needs.
struct ProtoReader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> ProtoReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn varint(&mut self) -> Result<u64, CtcSpmError> {
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            let byte = *self
                .data
                .get(self.offset)
                .ok_or(CtcSpmError::Truncated { at: self.offset })?;
            self.offset += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
            if shift >= 64 {
                return Err(CtcSpmError::BadWire {
                    wire: 0,
                    at: self.offset,
                });
            }
        }
    }

    fn tag(&mut self) -> Result<(u32, u32), CtcSpmError> {
        let tag = self.varint()?;
        Ok(((tag >> 3) as u32, (tag & 7) as u32))
    }

    fn length_delimited(&mut self) -> Result<&'a [u8], CtcSpmError> {
        let len = usize::try_from(self.varint()?)
            .map_err(|_| CtcSpmError::Truncated { at: self.offset })?;
        let end = self
            .offset
            .checked_add(len)
            .ok_or(CtcSpmError::Truncated { at: self.offset })?;
        let slice = self
            .data
            .get(self.offset..end)
            .ok_or(CtcSpmError::Truncated { at: self.offset })?;
        self.offset = end;
        Ok(slice)
    }

    fn skip_field(&mut self, wire: u32) -> Result<(), CtcSpmError> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => {
                self.offset = self
                    .offset
                    .checked_add(8)
                    .ok_or(CtcSpmError::Truncated { at: self.offset })?;
                if self.offset > self.data.len() {
                    return Err(CtcSpmError::Truncated { at: self.offset });
                }
            }
            2 => {
                self.length_delimited()?;
            }
            5 => {
                self.offset = self
                    .offset
                    .checked_add(4)
                    .ok_or(CtcSpmError::Truncated { at: self.offset })?;
                if self.offset > self.data.len() {
                    return Err(CtcSpmError::Truncated { at: self.offset });
                }
            }
            other => {
                return Err(CtcSpmError::BadWire {
                    wire: other,
                    at: self.offset,
                });
            }
        }
        Ok(())
    }

    fn eof(&self) -> bool {
        self.offset >= self.data.len()
    }
}

/// Parsed CTC SentencePiece vocabulary.
pub(crate) struct CtcSpm {
    id_to_piece: Vec<String>,
    types: Vec<u8>,
    /// piece -> id, including the special/control pieces.
    vocab: HashMap<String, u32>,
    /// UNUSED pieces, longest first, for atomic longest-prefix matching.
    specials: Vec<(String, u32)>,
    unk_id: u32,
}

impl CtcSpm {
    /// Parse a raw `ModelProto`. `expected_piece_count` is `ctc.vocab_size - 1`.
    pub(crate) fn parse(proto: &[u8], expected_piece_count: usize) -> Result<Self, CtcSpmError> {
        let mut reader = ProtoReader::new(proto);
        let mut id_to_piece: Vec<String> = Vec::new();
        let mut types: Vec<u8> = Vec::new();
        let mut model_type = None;

        while !reader.eof() {
            let (field, wire) = reader.tag()?;
            match field {
                1 if wire == 2 => {
                    // SentencePiece message.
                    let bytes = reader.length_delimited()?;
                    let mut piece_msg = ProtoReader::new(bytes);
                    let mut piece: Option<String> = None;
                    let mut ptype = 0u8;
                    while !piece_msg.eof() {
                        let (pf, pw) = piece_msg.tag()?;
                        match (pf, pw) {
                            (1, 2) => {
                                let raw = piece_msg.length_delimited()?;
                                piece = Some(
                                    std::str::from_utf8(raw)
                                        .map_err(|_| CtcSpmError::BadUtf8 {
                                            id: id_to_piece.len(),
                                        })?
                                        .to_string(),
                                );
                            }
                            (3, 0) => {
                                ptype = u8::try_from(piece_msg.varint()?).unwrap_or(u8::MAX);
                            }
                            _ => piece_msg.skip_field(pw)?,
                        }
                    }
                    let piece = piece.ok_or(CtcSpmError::BadUtf8 {
                        id: id_to_piece.len(),
                    })?;
                    id_to_piece.push(piece);
                    types.push(ptype);
                }
                2 if wire == 2 => {
                    // TrainerSpec: read model_type (field 3).
                    let bytes = reader.length_delimited()?;
                    let mut trainer = ProtoReader::new(bytes);
                    while !trainer.eof() {
                        let (tf, tw) = trainer.tag()?;
                        if tf == 3 && tw == 0 {
                            model_type = Some(u32::try_from(trainer.varint()?).unwrap_or(u32::MAX));
                        } else {
                            trainer.skip_field(tw)?;
                        }
                    }
                }
                _ => reader.skip_field(wire)?,
            }
        }

        let got = model_type.ok_or(CtcSpmError::UnexpectedModelType {
            got: 0,
            want: MODEL_TYPE_BPE,
        })?;
        if got != MODEL_TYPE_BPE {
            return Err(CtcSpmError::UnexpectedModelType {
                got,
                want: MODEL_TYPE_BPE,
            });
        }
        if id_to_piece.is_empty() {
            return Err(CtcSpmError::EmptyVocab);
        }
        if id_to_piece.len() != expected_piece_count {
            return Err(CtcSpmError::VocabSizeMismatch {
                got: id_to_piece.len(),
                want: expected_piece_count,
            });
        }

        let mut vocab = HashMap::with_capacity(id_to_piece.len());
        let mut specials: Vec<(String, u32)> = Vec::new();
        let mut unk_id = 0u32;
        for (id, piece) in id_to_piece.iter().enumerate() {
            let id_u32 = u32::try_from(id).map_err(|_| CtcSpmError::EmptyVocab)?;
            if vocab.insert(piece.clone(), id_u32).is_some() {
                return Err(CtcSpmError::DuplicatePiece {
                    id,
                    piece: piece.clone(),
                });
            }
            match types[id] {
                TYPE_UNUSED => specials.push((piece.clone(), id_u32)),
                TYPE_USER_DEFINED => unk_id = unk_id.min(id_u32), // informational
                _ => {}
            }
            // SentencePiece marks <unk> as CONTROL (type 2). The processor's
            // `unk_id()` resolves to that piece's id (0 in this model).
            if piece == "<unk>" {
                unk_id = id_u32;
            }
        }
        // Longest first so a single left-to-right pass picks maximal matches.
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));

        // The merge priority uses the id directly: for this model
        // `score(id) = min(0, 49 - id)`, so ordering by id is exactly ordering
        // by score (highest first), with ties (identical concat at two spots)
        // broken by leftmost position.

        Ok(Self {
            id_to_piece,
            types,
            vocab,
            specials,
            unk_id,
        })
    }

    pub(crate) fn unk_id(&self) -> u32 {
        self.unk_id
    }

    pub(crate) fn piece_to_id(&self, piece: &str) -> u32 {
        self.vocab.get(piece).copied().unwrap_or(self.unk_id)
    }

    pub(crate) fn id_to_piece(&self, id: u32) -> &str {
        self.id_to_piece
            .get(id as usize)
            .map(String::as_str)
            .unwrap_or("")
    }

    fn is_reserved(&self, id: u32) -> bool {
        matches!(self.types.get(id as usize), Some(&t) if t != TYPE_NORMAL && t != TYPE_USER_DEFINED && t != TYPE_UNUSED)
    }

    /// Concatenate pieces into text (▁ -> space). Used only to validate
    /// character units; CONTROL pieces are skipped like SentencePiece Decode.
    #[allow(dead_code)]
    pub(crate) fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            let Some(t) = self.types.get(id as usize) else {
                continue;
            };
            if *t == 2 {
                continue; // CONTROL
            }
            let piece = self.id_to_piece(id);
            if piece == "\u{2581}" {
                out.push(' ');
            } else {
                out.push_str(piece);
            }
        }
        out
    }

    /// `SentencePieceProcessor::Encode`: normalize, split into whitespace
    /// segments, and BPE-encode each with a leading ▁ marker.
    pub(crate) fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for word in text.split_whitespace() {
            out.extend(self.bpe_word(word));
        }
        out
    }

    fn initial_split(&self, text: &str, into: &mut Vec<String>) {
        into.clear();
        let bytes = text.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            let mut matched: Option<&(String, u32)> = None;
            for special in &self.specials {
                if bytes[i..].starts_with(special.0.as_bytes()) {
                    matched = Some(special);
                    break;
                }
            }
            if let Some((piece, _)) = matched {
                into.push(piece.clone());
                i += piece.len();
            } else {
                let ch = text[i..].chars().next().expect("non-empty remainder");
                into.push(ch.to_string());
                i += ch.len_utf8();
            }
        }
    }

    /// Greedy BPE over one ▁-prefixed word, matching `SampleEncode(0.0)`.
    fn bpe_word(&self, word: &str) -> Vec<u32> {
        let mut prefixed = String::with_capacity(word.len() + 4);
        prefixed.push(DUMMY_PREFIX);
        prefixed.push_str(word);

        let mut pieces: Vec<String> = Vec::new();
        self.initial_split(&prefixed, &mut pieces);
        let mut frozen: Vec<bool> = vec![false; pieces.len()];
        for (idx, piece) in pieces.iter().enumerate() {
            if self.specials.iter().any(|(sp, _)| sp == piece) {
                frozen[idx] = true;
            }
        }

        loop {
            let mut best: Option<(u32, usize)> = None; // (merged id, left index)
            for left in 0..pieces.len().saturating_sub(1) {
                if frozen[left] {
                    continue;
                }
                let mut concat = String::with_capacity(pieces[left].len() + pieces[left + 1].len());
                concat.push_str(&pieces[left]);
                concat.push_str(&pieces[left + 1]);
                let id = self.vocab.get(&concat).copied().unwrap_or(self.unk_id);
                if id == self.unk_id || self.is_reserved(id) {
                    continue;
                }
                if best.is_none_or(|(best_id, best_left)| (id, left) < (best_id, best_left)) {
                    best = Some((id, left));
                }
            }
            let Some((_, left)) = best else {
                break;
            };
            let mut merged = String::with_capacity(pieces[left].len() + pieces[left + 1].len());
            merged.push_str(&pieces[left]);
            merged.push_str(&pieces[left + 1]);
            pieces[left] = merged;
            pieces.remove(left + 1);
            frozen.remove(left + 1);
        }

        let mut out = Vec::with_capacity(pieces.len());
        for (piece, &frozen) in pieces.iter().zip(frozen.iter()) {
            let id = self.vocab.get(piece).copied().unwrap_or(self.unk_id);
            if frozen || (id != self.unk_id && !self.is_reserved(id)) {
                out.push(id);
            } else {
                out.push(self.unk_id);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_varint(out: &mut Vec<u8>, mut value: u64) {
        while value >= 0x80 {
            out.push((value as u8) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }
    fn push_varint_field(out: &mut Vec<u8>, field: u32, value: u64) {
        push_varint(out, u64::from(field << 3));
        push_varint(out, value);
    }
    fn push_bytes_field(out: &mut Vec<u8>, field: u32, data: &[u8]) {
        push_varint(out, ((field << 3) | 2) as u64);
        push_varint(out, data.len() as u64);
        out.extend_from_slice(data);
    }
    fn piece_msg(piece: &str, ptype: u8) -> Vec<u8> {
        let mut msg = Vec::new();
        push_bytes_field(&mut msg, 1, piece.as_bytes());
        push_varint_field(&mut msg, 3, u64::from(ptype));
        msg
    }
    fn build_proto(pieces: &[(&str, u8)], model_type: u32) -> Vec<u8> {
        let mut trainer = Vec::new();
        push_varint_field(&mut trainer, 3, u64::from(model_type));
        let mut proto = Vec::new();
        for (piece, ptype) in pieces {
            push_bytes_field(&mut proto, 1, &piece_msg(piece, *ptype));
        }
        push_bytes_field(&mut proto, 2, &trainer);
        proto
    }

    fn sample_spm() -> CtcSpm {
        // id: 0=<unk>(CONTROL), 1=▁, 2=a, 3=b, 4=ab, 5=▁a, 6=<spk:0>(UNUSED)
        let proto = build_proto(
            &[
                ("<unk>", 2),
                ("\u{2581}", 1),
                ("a", 1),
                ("b", 1),
                ("ab", 1),
                ("\u{2581}a", 1),
                ("<spk:0>", 4),
            ],
            2,
        );
        CtcSpm::parse(&proto, 7).expect("parse sample proto")
    }

    #[test]
    fn bpe_merge_prefers_lowest_id_and_atomic_specials() {
        let spm = sample_spm();
        assert_eq!(spm.unk_id(), 0);
        // "a" -> ▁a (single piece)
        assert_eq!(spm.encode("a"), vec![5]);
        // "ab": the two candidate merges are ▁a (id 5) and ab (id 4); the
        // lower id (ab) wins first, leaving ▁ + ab.
        assert_eq!(spm.encode("ab"), vec![1, 4]);
        // The UNUSED user symbol is consumed atomically after the ▁ marker.
        assert_eq!(spm.encode("<spk:0>"), vec![1, 6]);
        assert_eq!(spm.piece_to_id("ab"), 4);
        assert_eq!(spm.piece_to_id("zz"), 0);
    }

    #[test]
    fn rejects_wrong_model_type_and_bad_base64() {
        let proto = build_proto(&[("<unk>", 2)], 1); // UNIGRAM
        assert!(CtcSpm::parse(&proto, 1).is_err());
        assert_eq!(decode_base64("YWJj").unwrap(), b"abc");
        assert!(decode_base64("a").is_err());
    }
}
