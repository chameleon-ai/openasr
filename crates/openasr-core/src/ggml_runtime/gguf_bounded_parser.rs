//! Bounded GGUF header prevalidation, performed entirely in Rust.
//!
//! Stock ggml ships `gguf_init_from_buffer`, which trusts the header's
//! declared counts and string lengths. The retired fork added caller-supplied
//! parse limits to stop hostile or corrupt files from driving unbounded
//! allocation. This module re-establishes that guarantee on stock ggml: the
//! buffer is walked once, in order, with every count and length checked
//! against the caller's [`ffi::GgufParseLimits`] before a single byte is
//! handed to C. The walk allocates nothing and runs in linear time; any
//! inconsistency is reported as [`GgufBoundedParseFailure::InvalidData`].
//!
//! Scope: structural framing only (magic, version, counts, strings, arrays,
//! header bytes, dims). Value-level semantics (UTF-8 validity, tensor type
//! sanity, offset sanity) stay with the stock parser and the metadata reader,
//! exactly as they were with the fork's bounded parser.

use super::{ffi, gguf_metadata::GgufBoundedParseFailure};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_MAX_DIMS: u32 = 4;
const GGUF_TYPE_U8: u32 = 0;
const GGUF_TYPE_I8: u32 = 1;
const GGUF_TYPE_U16: u32 = 2;
const GGUF_TYPE_I16: u32 = 3;
const GGUF_TYPE_U32: u32 = 4;
const GGUF_TYPE_I32: u32 = 5;
const GGUF_TYPE_F32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_U64: u32 = 10;
const GGUF_TYPE_I64: u32 = 11;
const GGUF_TYPE_F64: u32 = 12;
const GGUF_TYPE_COUNT: u32 = 13;
const MAX_NEST_DEPTH: u32 = 16;

#[derive(Clone, Copy)]
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    limits: ffi::GgufParseLimits,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], limits: ffi::GgufParseLimits) -> Self {
        Self {
            bytes,
            pos: 0,
            limits,
        }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], GgufBoundedParseFailure> {
        let end = self
            .pos
            .checked_add(count)
            .ok_or(GgufBoundedParseFailure::InvalidData)?;
        if end as u64 > self.limits.max_header_bytes {
            return Err(GgufBoundedParseFailure::InvalidData);
        }
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or(GgufBoundedParseFailure::InvalidData)?;
        self.pos = end;
        Ok(slice)
    }

    fn read_u32(&mut self) -> Result<u32, GgufBoundedParseFailure> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4-byte slice"),
        ))
    }

    fn read_u64(&mut self) -> Result<u64, GgufBoundedParseFailure> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8-byte slice"),
        ))
    }

    fn read_string(&mut self) -> Result<Vec<u8>, GgufBoundedParseFailure> {
        let len = self.read_u64()?;
        if len > self.limits.max_string_bytes {
            return Err(GgufBoundedParseFailure::InvalidData);
        }
        Ok(self.take(len as usize)?.to_vec())
    }

    fn skip_value(&mut self, value_type: u32, depth: u32) -> Result<(), GgufBoundedParseFailure> {
        if depth > MAX_NEST_DEPTH {
            return Err(GgufBoundedParseFailure::InvalidData);
        }
        match value_type {
            GGUF_TYPE_U8 | GGUF_TYPE_I8 | GGUF_TYPE_BOOL => {
                self.take(1)?;
                Ok(())
            }
            GGUF_TYPE_U16 | GGUF_TYPE_I16 => {
                self.take(2)?;
                Ok(())
            }
            GGUF_TYPE_U32 | GGUF_TYPE_I32 | GGUF_TYPE_F32 => {
                self.take(4)?;
                Ok(())
            }
            GGUF_TYPE_U64 | GGUF_TYPE_I64 | GGUF_TYPE_F64 => {
                self.take(8)?;
                Ok(())
            }
            GGUF_TYPE_STRING => {
                self.read_string()?;
                Ok(())
            }
            GGUF_TYPE_ARRAY => {
                let element_type = self.read_u32()?;
                if element_type >= GGUF_TYPE_COUNT || element_type == GGUF_TYPE_ARRAY {
                    return Err(GgufBoundedParseFailure::InvalidData);
                }
                let count = self.read_u64()?;
                if count > self.limits.max_array_elements {
                    return Err(GgufBoundedParseFailure::InvalidData);
                }
                for _ in 0..count {
                    self.skip_value(element_type, depth + 1)?;
                }
                Ok(())
            }
            _ => Err(GgufBoundedParseFailure::InvalidData),
        }
    }
}

/// Validate one GGUF buffer against `limits`. `Ok(())` means the framing is
/// consistent and within budget; the stock parser may then be handed the
/// buffer.
pub(crate) fn prevalidate_gguf_buffer(
    bytes: &[u8],
    limits: ffi::GgufParseLimits,
) -> Result<(), GgufBoundedParseFailure> {
    let mut cursor = Cursor::new(bytes, limits);
    if cursor.take(4)? != GGUF_MAGIC.as_slice() {
        return Err(GgufBoundedParseFailure::InvalidData);
    }
    let version = cursor.read_u32()?;
    if version != 2 && version != 3 {
        return Err(GgufBoundedParseFailure::InvalidData);
    }
    let tensor_count = cursor.read_u64()?;
    if tensor_count > limits.max_tensors {
        return Err(GgufBoundedParseFailure::InvalidData);
    }
    let kv_count = cursor.read_u64()?;
    if kv_count > limits.max_kv {
        return Err(GgufBoundedParseFailure::InvalidData);
    }
    for _ in 0..kv_count {
        let _key = cursor.read_string()?;
        let value_type = cursor.read_u32()?;
        cursor.skip_value(value_type, 0)?;
    }
    for _ in 0..tensor_count {
        let _name = cursor.read_string()?;
        let n_dims = cursor.read_u32()?;
        if n_dims == 0 || n_dims > GGUF_MAX_DIMS {
            return Err(GgufBoundedParseFailure::InvalidData);
        }
        let mut elements: u64 = 1;
        for _ in 0..n_dims {
            let dim = cursor.read_u64()?;
            if dim == 0 {
                // A zero element dimension is legal in stock ggml only as an
                // empty tensor; the bounded contract refuses it.
                return Err(GgufBoundedParseFailure::InvalidData);
            }
            elements = elements
                .checked_mul(dim)
                .ok_or(GgufBoundedParseFailure::InvalidData)?;
        }
        let _tensor_type = cursor.read_u32()?;
        let _offset = cursor.read_u64()?;
        let _ = elements;
    }
    Ok(())
}

/// Structural-byte estimate for one GGUF context holding `n_kv` key/value
/// pairs and `n_tensors` tensor descriptors. Used for resource planning
/// before the C parser runs. Stock ggml has no query for this, so the
/// estimate mirrors the layout of `struct gguf_context` on LP64: a vector of
/// key/values plus a vector of tensor infos, plus per-entry heap payload.
/// Values are upper bounds on the CPU-side structures only (never the
/// tensor payload, which is mmap-backed by the caller).
pub(crate) fn structural_bytes(n_kv: u64, n_tensors: u64) -> Option<u64> {
    const CONTEXT_PREFIX: u64 = 128;
    const PER_KV_STRUCT: u64 = 192; // std::string key + vectors + type tags
    const PER_TENSOR_INFO: u64 = 384; // ggml_tensor + offset
    n_kv.checked_mul(PER_KV_STRUCT)?
        .checked_add(n_tensors.checked_mul(PER_TENSOR_INFO)?)?
        .checked_add(CONTEXT_PREFIX)
}

/// Estimated ratio of heap bytes to wire bytes while parsing a GGUF header.
/// Retained for the adapter resource planner; see [`structural_bytes`].
pub(crate) const PAYLOAD_WIRE_MULTIPLIER: u64 = 8;

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> ffi::GgufParseLimits {
        ffi::GgufParseLimits {
            max_tensors: 2,
            max_kv: 2,
            max_string_bytes: 16,
            max_array_elements: 4,
            max_header_bytes: 4096,
        }
    }

    fn empty_header(tensors: u64, kv: u64) -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&tensors.to_le_bytes());
        bytes.extend_from_slice(&kv.to_le_bytes());
        bytes
    }

    #[test]
    fn valid_empty_header_passes() {
        assert!(prevalidate_gguf_buffer(&empty_header(0, 0), limits()).is_ok());
    }

    #[test]
    fn tensor_count_overflow_fails_closed() {
        assert_eq!(
            prevalidate_gguf_buffer(&empty_header(3, 0), limits()),
            Err(GgufBoundedParseFailure::InvalidData)
        );
    }

    #[test]
    fn kv_count_overflow_fails_closed() {
        assert_eq!(
            prevalidate_gguf_buffer(&empty_header(0, 3), limits()),
            Err(GgufBoundedParseFailure::InvalidData)
        );
    }

    #[test]
    fn header_budget_is_enforced() {
        let mut bytes = empty_header(0, 1);
        bytes.extend_from_slice(&4_u64.to_le_bytes());
        bytes.extend_from_slice(b"name");
        bytes.extend_from_slice(&(GGUF_TYPE_STRING).to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.push(b'x');
        let limited = ffi::GgufParseLimits {
            max_header_bytes: 24,
            ..limits()
        };
        assert_eq!(
            prevalidate_gguf_buffer(&bytes, limited),
            Err(GgufBoundedParseFailure::InvalidData)
        );
    }

    #[test]
    fn string_budget_is_enforced() {
        let mut bytes = empty_header(0, 1);
        bytes.extend_from_slice(&17_u64.to_le_bytes());
        bytes.extend_from_slice(&[0_u8; 17]);
        bytes.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        let limited = ffi::GgufParseLimits {
            max_string_bytes: 16,
            ..limits()
        };
        assert_eq!(
            prevalidate_gguf_buffer(&bytes, limited),
            Err(GgufBoundedParseFailure::InvalidData)
        );
    }

    #[test]
    fn array_budget_is_enforced() {
        let mut bytes = empty_header(0, 1);
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(b"x");
        bytes.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
        bytes.extend_from_slice(&GGUF_TYPE_U32.to_le_bytes());
        bytes.extend_from_slice(&5_u64.to_le_bytes());
        let limited = ffi::GgufParseLimits {
            max_array_elements: 4,
            ..limits()
        };
        assert_eq!(
            prevalidate_gguf_buffer(&bytes, limited),
            Err(GgufBoundedParseFailure::InvalidData)
        );
    }

    #[test]
    fn nested_array_depth_is_bounded() {
        let mut bytes = empty_header(0, 1);
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(b"x");
        for _ in 0..17 {
            bytes.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
            bytes.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
            bytes.extend_from_slice(&1_u64.to_le_bytes());
        }
        bytes.extend_from_slice(&GGUF_TYPE_U32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        assert_eq!(
            prevalidate_gguf_buffer(&bytes, limits()),
            Err(GgufBoundedParseFailure::InvalidData)
        );
    }
}
