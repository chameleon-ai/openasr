//! OpenASR graph operators implemented on stock ggml's `GGML_OP_CUSTOM`
//! (`ggml_custom_4d`) with callbacks compiled into this crate.
//!
//! Stock ggml has no rel-pos flash attention, LSTM sequence scan, swoosh
//! activation, or first-max argmax. These are behavioral equivalents of the
//! retired fork-only kernels (see ggml-migration-report.md); they run on the
//! CPU backend only and always with `n_tasks = 1`.
//!
//! Parameter passing: `ggml_custom_4d` stores `{fun, n_tasks, userdata}` in
//! the tensor's `op_params` block; the remainder of the 64-byte block is
//! openasr's. Kernel parameters are serialized into `op_params` starting at
//! byte offset 24, the size of `struct ggml_custom_op_params` on every
//! LP64/LLP64 target openasr builds for (`fun` at 0, `n_tasks` at 8, the
//! userdata pointer at 16). Every write goes through
//! [`write_custom_op_params`], which verifies that the first 8 bytes still
//! hold the callback pointer before trusting the offset. If stock ggml ever
//! changes that layout, graph construction fails closed with a typed error
//! instead of silently misreading parameters.
//!
//! Callbacks run inside ggml's CPU threadpool. They must stay panic-free and
//! read-only on `src` tensors. Both invariants are enforced by construction
//! and by `catch_unwind` at the FFI edge, which aborts the process rather
//! than unwind into C.

use std::ffi::c_int;

use super::ffi;

const CUSTOM_OP_PARAMS_OFFSET: usize = 24;
const CUSTOM_OP_PARAMS_BLOCK: usize = 64;

fn op_params_bytes(tensor: ffi::GgmlTensorRaw) -> *mut u8 {
    // SAFETY: `tensor` is a live ggml tensor allocated by the graph context;
    // `op_params` is 64 bytes of writable memory inside it, and the
    // containing struct is mirror-declared as `#[repr(C)] GgmlTensorAllocPrefix`.
    unsafe {
        (*tensor.cast::<ffi::GgmlTensorAllocPrefix>())
            .op_params
            .as_mut_ptr()
            .cast::<u8>()
    }
}

fn read_params(tensor: ffi::GgmlTensorRaw) -> &'static [u8] {
    let base = op_params_bytes(tensor) as *const u8;
    // SAFETY: same backing as write_custom_op_params; read-only.
    unsafe {
        std::slice::from_raw_parts(
            base.add(CUSTOM_OP_PARAMS_OFFSET),
            CUSTOM_OP_PARAMS_BLOCK - CUSTOM_OP_PARAMS_OFFSET,
        )
    }
}

/// Serialize `params` into the tensor's `op_params` block, failing closed if
/// the stock ggml `ggml_custom_op_params` layout is not the expected one.
pub(crate) fn write_custom_op_params(
    tensor: ffi::GgmlTensorRaw,
    fun: ffi::GgmlCustomOpFn,
    params: &[u8],
) -> Result<(), CustomOpParamError> {
    const MAX: usize = CUSTOM_OP_PARAMS_BLOCK - CUSTOM_OP_PARAMS_OFFSET;
    if params.len() > MAX {
        return Err(CustomOpParamError::TooLarge {
            len: params.len(),
            max: MAX,
        });
    }
    let base = op_params_bytes(tensor);
    // SAFETY: see above; 64 bytes of writable memory.
    let raw = unsafe { std::slice::from_raw_parts_mut(base, CUSTOM_OP_PARAMS_BLOCK) };
    // The first 8 bytes must be the callback pointer ggml stored for
    // GGML_OP_CUSTOM. If they are not, `ggml_custom_op_params` changed shape
    // and no offset reasoning is trustworthy.
    let stored = u64::from_le_bytes(raw[..8].try_into().expect("8-byte slice"));
    if stored != fun as usize as u64 {
        return Err(CustomOpParamError::LayoutMismatch);
    }
    raw[CUSTOM_OP_PARAMS_OFFSET..CUSTOM_OP_PARAMS_OFFSET + params.len()].copy_from_slice(params);
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CustomOpParamError {
    TooLarge { len: usize, max: usize },
    LayoutMismatch,
}

impl std::fmt::Display for CustomOpParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge { len, max } => {
                write!(f, "custom op parameters are {len} bytes (max {max})")
            }
            Self::LayoutMismatch => write!(
                f,
                "GGML_OP_CUSTOM parameter layout does not match the stock ggml 1.0.0 ABI"
            ),
        }
    }
}

impl std::error::Error for CustomOpParamError {}

/// Create a custom op tensor and serialize its parameters. `raw` is the value
/// returned by `ggml_custom_4d`; a null pointer means the graph was out of
/// compute budget and is reported as a layout/ABI failure (the only plausible
/// local cause).
pub(crate) fn finish_custom_op(
    raw: ffi::GgmlTensorRaw,
    fun: ffi::GgmlCustomOpFn,
    params: &[u8],
) -> Result<ffi::GgmlTensorRaw, CustomOpParamError> {
    if raw.is_null() {
        return Err(CustomOpParamError::LayoutMismatch);
    }
    write_custom_op_params(raw, fun, params)?;
    Ok(raw)
}

fn tensor_at<'a>(tensor: ffi::GgmlTensorRaw) -> Option<&'a ffi::GgmlTensorAllocPrefix> {
    // SAFETY: callers only pass live ggml tensors owned by a no_alloc context.
    unsafe { tensor.cast::<ffi::GgmlTensorAllocPrefix>().as_ref() }
}

fn src_tensor<'a>(
    dst: &ffi::GgmlTensorAllocPrefix,
    index: usize,
) -> Option<&'a ffi::GgmlTensorAllocPrefix> {
    // SAFETY: source tensor pointers stored by ggml are owned by the same
    // context and outlive the dst tensor within the graph build/compute.
    unsafe { dst.src[index].cast::<ffi::GgmlTensorAllocPrefix>().as_ref() }
}

fn strides_f32_elements(tensor: &ffi::GgmlTensorAllocPrefix) -> [i64; 4] {
    let nb = &tensor.nb;
    let type_size = std::mem::size_of::<f32>() as i64;
    [
        nb[0] as i64 / type_size,
        nb[1] as i64 / type_size,
        nb[2] as i64 / type_size,
        nb[3] as i64 / type_size,
    ]
}

/// swoosh(x) = softplus(x - offset) - linear_scale * x - shift
/// with softplus(t) = t > 20.0 ? t : ln(1 + exp(t))
pub(crate) unsafe extern "C" fn swoosh_custom_op(
    dst: ffi::GgmlTensorRaw,
    _ith: c_int,
    _nth: c_int,
    _userdata: *mut std::ffi::c_void,
) {
    catch_kernel(|| {
        let params = read_params(dst);
        let offset = f32::from_le_bytes(params[0..4].try_into().expect("4 bytes"));
        let shift = f32::from_le_bytes(params[4..8].try_into().expect("4 bytes"));
        let linear_scale = f32::from_le_bytes(params[8..12].try_into().expect("4 bytes"));

        let Some(dst_t) = tensor_at(dst) else { return };
        let Some(src_t) = src_tensor(dst_t, 0) else {
            return;
        };
        if src_t.type_ != ffi::GGML_TYPE_F32 || dst_t.type_ != ffi::GGML_TYPE_F32 {
            return;
        }
        let ne = dst_t.ne;
        let strides = strides_f32_elements(src_t);
        let src_data = src_t.data.cast::<f32>();
        let out = dst_t.data.cast::<f32>();
        let nelements = ne[0] * ne[1] * ne[2] * ne[3];
        for index in 0..nelements {
            let i0 = index % ne[0];
            let rest = index / ne[0];
            let i1 = rest % ne[1];
            let rest = rest / ne[1];
            let i2 = rest % ne[2];
            let i3 = rest / ne[2];
            let x = unsafe {
                *src_data.add(
                    (i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + i3 * strides[3])
                        as usize,
                )
            };
            let shifted = x - offset;
            let softplus = if shifted > 20.0 {
                shifted
            } else {
                (1.0f32 + shifted.exp()).ln()
            };
            // The builder guarantees dst is contiguous (dup tensor).
            unsafe {
                out.add(index as usize)
                    .write(softplus - linear_scale * x - shift)
            };
        }
    });
}

/// First-max argmax: per row of a contiguous F32 matrix, the lowest column
/// index whose value is maximal, or -1 if any value in the row is non-finite.
pub(crate) unsafe extern "C" fn argmax_first_custom_op(
    dst: ffi::GgmlTensorRaw,
    _ith: c_int,
    _nth: c_int,
    _userdata: *mut std::ffi::c_void,
) {
    catch_kernel(|| {
        let Some(dst_t) = tensor_at(dst) else { return };
        let Some(src_t) = src_tensor(dst_t, 0) else {
            return;
        };
        if src_t.type_ != ffi::GGML_TYPE_F32 || dst_t.type_ != ffi::GGML_TYPE_I32 {
            return;
        }
        let strides = strides_f32_elements(src_t);
        let ncols = src_t.ne[0];
        let nrows = src_t.ne[1] * src_t.ne[2] * src_t.ne[3];
        let row_stride = strides[1].max(1);
        let data = src_t.data.cast::<f32>();
        let out = dst_t.data.cast::<i32>();
        for row in 0..nrows {
            let row_ptr = unsafe { data.add((row * row_stride) as usize) };
            let mut best: i32 = 0;
            if !unsafe { *row_ptr }.is_finite() {
                unsafe { out.add(row as usize).write(-1) };
                continue;
            }
            for col in 1..ncols {
                let value = unsafe { *row_ptr.add(col as usize) };
                if !value.is_finite() {
                    best = -1;
                    break;
                }
                if value > unsafe { *row_ptr.add(best as usize) } {
                    best = col as i32;
                }
            }
            unsafe { out.add(row as usize).write(best) };
        }
    });
}

/// ggml_lstm_seq equivalent. h/c are zero-initialized; for each step the input
/// is `x[:, step, batch]` (or reversed). Gate order IOFC (i, o, f, c) or IFGO
/// (i, f, g(cell), o). b may carry concatenated input/recurrent biases when
/// its length is `8 * hidden`; the recurrent half is added to the same
/// pre-activations.
pub(crate) unsafe extern "C" fn lstm_seq_custom_op(
    dst: ffi::GgmlTensorRaw,
    _ith: c_int,
    _nth: c_int,
    _userdata: *mut std::ffi::c_void,
) {
    catch_kernel(|| {
        let params = read_params(dst);
        let gate_order = i32::from_le_bytes(params[0..4].try_into().expect("4 bytes"));
        let reverse = i32::from_le_bytes(params[4..8].try_into().expect("4 bytes")) != 0;
        let Some(dst_t) = tensor_at(dst) else { return };
        let mut src = [None; 4];
        for (index, slot) in src.iter_mut().enumerate() {
            *slot = src_tensor(dst_t, index);
        }
        let (Some(x_t), Some(w_t), Some(r_t), Some(b_t)) = (src[0], src[1], src[2], src[3]) else {
            return;
        };
        if x_t.type_ != ffi::GGML_TYPE_F32
            || w_t.type_ != ffi::GGML_TYPE_F32
            || r_t.type_ != ffi::GGML_TYPE_F32
            || b_t.type_ != ffi::GGML_TYPE_F32
            || dst_t.type_ != ffi::GGML_TYPE_F32
        {
            return;
        }

        let input = x_t.ne[0];
        let seq = x_t.ne[1];
        let batch = x_t.ne[2];
        let hidden = dst_t.ne[0];
        if input <= 0 || seq <= 0 || batch <= 0 || hidden <= 0 || hidden > 65_536 {
            return;
        }
        let hidden = hidden as usize;
        let has_recurrent_bias = b_t.ne[0] == 8 * hidden as i64;

        let (input_gate, forget_gate, cell_gate, output_gate) = if gate_order == 0 {
            (0i64, 2i64, 3i64, 1i64) // IOFC
        } else {
            (0i64, 1i64, 2i64, 3i64) // IFGO
        };

        let x_data = x_t.data.cast::<f32>();
        let w_data = w_t.data.cast::<f32>();
        let r_data = r_t.data.cast::<f32>();
        let b_data = b_t.data.cast::<f32>();
        let y_data = dst_t.data.cast::<f32>();

        let sigmoid = |value: f32| {
            if value >= 0.0 {
                let z = (-value).exp();
                1.0 / (1.0 + z)
            } else {
                let z = value.exp();
                z / (1.0 + z)
            }
        };

        for ib in 0..batch {
            let mut h = vec![0.0f32; hidden];
            let mut c = vec![0.0f32; hidden];
            let mut h_next = vec![0.0f32; hidden];

            for step in 0..seq {
                let it = if reverse { seq - 1 - step } else { step };
                let x_row = unsafe { x_data.add(((ib * seq + it) * input) as usize) };
                let y_row = unsafe { y_data.add(((ib * seq + it) * hidden as i64) as usize) };

                for ih in 0..hidden {
                    let i_col = input_gate * hidden as i64 + ih as i64;
                    let f_col = forget_gate * hidden as i64 + ih as i64;
                    let c_col = cell_gate * hidden as i64 + ih as i64;
                    let o_col = output_gate * hidden as i64 + ih as i64;

                    let mut i_pre = unsafe { *b_data.add(i_col as usize) };
                    let mut f_pre = unsafe { *b_data.add(f_col as usize) };
                    let mut c_pre = unsafe { *b_data.add(c_col as usize) };
                    let mut o_pre = unsafe { *b_data.add(o_col as usize) };
                    if has_recurrent_bias {
                        let base = 4 * hidden;
                        i_pre += unsafe { *b_data.add(base + i_col as usize) };
                        f_pre += unsafe { *b_data.add(base + f_col as usize) };
                        c_pre += unsafe { *b_data.add(base + c_col as usize) };
                        o_pre += unsafe { *b_data.add(base + o_col as usize) };
                    }

                    for (pre, col) in [
                        (&mut i_pre, i_col),
                        (&mut f_pre, f_col),
                        (&mut c_pre, c_col),
                        (&mut o_pre, o_col),
                    ] {
                        let mut dot = 0.0f32;
                        for k in 0..input {
                            dot += unsafe { *x_row.add(k as usize) }
                                * unsafe { *w_data.add((col * input + k) as usize) };
                        }
                        *pre += dot;
                    }
                    for (pre, col) in [
                        (&mut i_pre, i_col),
                        (&mut f_pre, f_col),
                        (&mut c_pre, c_col),
                        (&mut o_pre, o_col),
                    ] {
                        let mut dot = 0.0f32;
                        for k in 0..hidden as i64 {
                            dot += h[k as usize]
                                * unsafe { *r_data.add((col * hidden as i64 + k) as usize) };
                        }
                        *pre += dot;
                    }

                    let input_value = sigmoid(i_pre);
                    let forget_value = sigmoid(f_pre);
                    let cell_value = c_pre.tanh();
                    let output_value = sigmoid(o_pre);

                    c[ih] = forget_value * c[ih] + input_value * cell_value;
                    h_next[ih] = output_value * c[ih].tanh();
                }

                h.copy_from_slice(&h_next);
                unsafe {
                    for (ih, value) in h.iter().enumerate() {
                        y_row.add(ih).write(*value);
                    }
                }
            }
        }
    });
}

/// Transformer-XL AC+BD relative attention with online softmax. Never
/// materializes the Nq x Nk score matrix. Argument order in `src[]`:
/// q_u, q_v, k, r, v, mask(optional). Same output layout as
/// ggml_flash_attn_ext: `[Dv, H, Nq, B]`.
pub(crate) unsafe extern "C" fn flash_attn_rel_pos_custom_op(
    dst: ffi::GgmlTensorRaw,
    _ith: c_int,
    _nth: c_int,
    _userdata: *mut std::ffi::c_void,
) {
    catch_kernel(|| {
        let params = read_params(dst);
        let scale = f32::from_le_bytes(params[0..4].try_into().expect("4 bytes"));
        let Some(dst_t) = tensor_at(dst) else { return };
        let mut src = [None; 6];
        for (index, slot) in src.iter_mut().enumerate() {
            *slot = src_tensor(dst_t, index);
        }
        let (Some(q_u_t), Some(q_v_t), Some(k_t), Some(r_t), Some(v_t)) =
            (src[0], src[1], src[2], src[3], src[4])
        else {
            return;
        };
        let mask_t = src[5];
        if q_u_t.type_ != ffi::GGML_TYPE_F32
            || q_v_t.type_ != ffi::GGML_TYPE_F32
            || k_t.type_ != ffi::GGML_TYPE_F32
            || r_t.type_ != ffi::GGML_TYPE_F32
            || v_t.type_ != ffi::GGML_TYPE_F32
            || dst_t.type_ != ffi::GGML_TYPE_F32
        {
            return;
        }

        let dk = k_t.ne[0];
        let dv = v_t.ne[0];
        let nq = q_u_t.ne[1];
        let nk = k_t.ne[1];
        let h = q_u_t.ne[2];
        let hkv = k_t.ne[2];
        if dk <= 0 || dv <= 0 || nk <= 0 || h <= 0 || hkv <= 0 || h % hkv != 0 {
            return;
        }

        let rk2 = h / hkv;
        let rv2 = h / v_t.ne[2];
        let rr2 = h / r_t.ne[2];

        let sq = strides_f32_elements(q_u_t);
        let sqv = strides_f32_elements(q_v_t);
        let sk = strides_f32_elements(k_t);
        let sr = strides_f32_elements(r_t);
        let sv = strides_f32_elements(v_t);
        let sm = mask_t.map(strides_f32_elements);
        let sd = strides_f32_elements(dst_t);

        let q_u_data = q_u_t.data.cast::<f32>();
        let q_v_data = q_v_t.data.cast::<f32>();
        let k_data = k_t.data.cast::<f32>();
        let r_data = r_t.data.cast::<f32>();
        let v_data = v_t.data.cast::<f32>();
        let mask_data = mask_t.map(|t| t.data.cast::<f32>());
        let dst_data = dst_t.data.cast::<f32>();

        for iq3 in 0..q_u_t.ne[3] {
            for iq2 in 0..h {
                let ik2 = iq2 / rk2;
                let ik3 = iq3 / (q_u_t.ne[3] / k_t.ne[3]);
                let iv2 = iq2 / rv2;
                let iv3 = iq3 / (q_u_t.ne[3] / v_t.ne[3]);
                let ir2 = iq2 / rr2;
                let ir3 = iq3 / (q_u_t.ne[3] / r_t.ne[3]);

                for iq1 in 0..nq {
                    let q_u_ptr =
                        unsafe { q_u_data.add((iq1 * sq[1] + iq2 * sq[2] + iq3 * sq[3]) as usize) };
                    let q_v_ptr = unsafe {
                        q_v_data.add((iq1 * sqv[1] + iq2 * sqv[2] + iq3 * sqv[3]) as usize)
                    };

                    let mask_base = mask_t.map(|m| {
                        let strides_m = sm.as_ref().expect("mask strides");
                        let im1 = iq1.rem_euclid(m.ne[1]);
                        let im2 = iq2.rem_euclid(m.ne[2]);
                        let im3 = iq3.rem_euclid(m.ne[3]);
                        let data = mask_data.expect("mask data");
                        let offset = im1 * strides_m[1] + im2 * strides_m[2] + im3 * strides_m[3];
                        (data, strides_m[0], offset)
                    });

                    let mut s = 0.0f32;
                    let mut m = f32::NEG_INFINITY;
                    let mut v_kq = vec![0.0f32; dv as usize];

                    for ic in 0..nk {
                        let mv = mask_base.as_ref().map(|(data, stride0, offset)| unsafe {
                            *data.add((offset + ic * stride0) as usize)
                        });
                        if mv == Some(f32::NEG_INFINITY) {
                            continue;
                        }

                        let k_ptr = unsafe {
                            k_data.add((ic * sk[1] + ik2 * sk[2] + ik3 * sk[3]) as usize)
                        };
                        let mut s_ac = 0.0f32;
                        for d in 0..dk {
                            s_ac += unsafe { *k_ptr.add((d * sk[0]) as usize) }
                                * unsafe { *q_u_ptr.add((d * sq[0]) as usize) };
                        }

                        let rel = (nq - 1) + ic - iq1;
                        let r_ptr = unsafe {
                            r_data.add((rel * sr[1] + ir2 * sr[2] + ir3 * sr[3]) as usize)
                        };
                        let mut s_bd = 0.0f32;
                        for d in 0..dk {
                            s_bd += unsafe { *r_ptr.add((d * sr[0]) as usize) }
                                * unsafe { *q_v_ptr.add((d * sqv[0]) as usize) };
                        }

                        let score = scale * (s_ac + s_bd) + mv.unwrap_or(0.0);

                        let mold = m;
                        let (ms, vs) = if score > m {
                            m = score;
                            let ms = (mold - m).exp();
                            for value in v_kq.iter_mut() {
                                *value *= ms;
                            }
                            (ms, 1.0f32)
                        } else {
                            (1.0f32, (score - m).exp())
                        };

                        let v_ptr = unsafe {
                            v_data.add((ic * sv[1] + iv2 * sv[2] + iv3 * sv[3]) as usize)
                        };
                        for d in 0..dv {
                            v_kq[d as usize] += vs * unsafe { *v_ptr.add((d * sv[0]) as usize) };
                        }
                        s = s * ms + vs;
                    }

                    let s_inv = if s == 0.0 { 0.0 } else { 1.0 / s };
                    let dst_ptr =
                        unsafe { dst_data.add((iq3 * sd[3] + iq1 * sd[2] + iq2 * sd[1]) as usize) };
                    for d in 0..dv {
                        unsafe {
                            dst_ptr
                                .add((d * sd[0]) as usize)
                                .write(v_kq[d as usize] * s_inv);
                        }
                    }
                }
            }
        }
    });
}

fn catch_kernel(kernel: impl FnOnce()) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(kernel));
    if result.is_err() {
        eprintln!(
            "openasr: custom ggml operator panicked; aborting rather than unwinding into ggml"
        );
        std::process::abort();
    }
}
