//! Nemotron-transcribe mel frontend: 16 kHz mono -> log-mel (128 bins).
//!
//! Faithful port of the reference frontend (`src/asr/features/fe.cpp`,
//! `MelSpecConfig` path used by the SALM engine): pre-emphasis on the raw
//! signal, zero-pad `n_fft/2` on both sides (NeMo's `torch.stft(center=True,
//! pad_mode='constant')`), periodic Hann of `win_length=400` placed at frame
//! offset 0 of the 512-point FFT, power spectrum, the *trained* mel filterbank
//! from the pack (`preprocessor.fb`) applied only over each row's nonzero span,
//! and `log(mel + 2^-24)`. No normalization happens here: the ASR expert
//! z-scores per feature per utterance, the diar expert consumes the raw
//! log-mel.

use thiserror::Error;

pub(crate) const NEMOTRON_TRANSCRIBE_SAMPLE_RATE_HZ: u32 = 16_000;
pub(crate) const NEMOTRON_TRANSCRIBE_N_FFT: usize = 512;
pub(crate) const NEMOTRON_TRANSCRIBE_WIN_LENGTH: usize = 400;
pub(crate) const NEMOTRON_TRANSCRIBE_HOP_LENGTH: usize = 160;
pub(crate) const NEMOTRON_TRANSCRIBE_N_MELS: usize = 128;
const PREEMPH: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 1.0 / 16_777_216.0; // 2^-24
const FFT_BINS: usize = NEMOTRON_TRANSCRIBE_N_FFT / 2 + 1;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum NemotronMelFrontendError {
    #[error("nemotron-transcribe frontend received no samples")]
    Empty,
    #[error("nemotron-transcribe frontend sample count {count} overflows internal arithmetic")]
    Overflow { count: usize },
    #[error("nemotron-transcribe frontend failed to read pack tensor '{tensor}': {reason}")]
    PackRead {
        tensor: &'static str,
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NemotronMelFeatures {
    /// `frames` rows of `NEMOTRON_TRANSCRIBE_N_MELS` log-mel values each,
    /// frame-major (`features[frame * N_MELS + mel]`).
    pub features: Vec<f32>,
    pub frames: usize,
}

/// Periodic Hann window of the reference (`hann_periodic` = true) placed at
/// offset 0 of the 512-point FFT (`stft_center_window` = false for SALM).
fn hann_periodic(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / length as f32).cos()))
        .collect()
}

/// Trained mel filterbank from the pack, row-major `(n_mels, n_fft/2+1)`.
pub(crate) struct NemotronMelFrontend {
    r2c: std::sync::Arc<dyn realfft::RealToComplex<f32>>,
    mel_basis: Vec<f32>,
    /// First nonzero bin of each row; the dot product runs `[kbeg, kend)`.
    mel_kbeg: Vec<usize>,
    mel_kend: Vec<usize>,
    window: Vec<f32>,
}

impl NemotronMelFrontend {
    /// Production constructor: reads the pack's `preprocessor.fb` filterbank
    /// (F32 `[257, 128]`) via the preflight's tensor reader.
    pub(crate) fn from_preflight_fb(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    ) -> Result<Self, NemotronMelFrontendError> {
        let reader =
            crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight(preflight)
                .map_err(|error| NemotronMelFrontendError::PackRead {
                    tensor: "preprocessor.fb",
                    reason: error.to_string(),
                })?;
        let expected_shape = [257u64, NEMOTRON_TRANSCRIBE_N_MELS as u64];
        let fb = reader
            .host_tensor_f32_copy_dequantized_by_name("preprocessor.fb", &expected_shape)
            .map_err(|error| NemotronMelFrontendError::PackRead {
                tensor: "preprocessor.fb",
                reason: error.to_string(),
            })?;
        Self::from_pack_filterbank(&fb)
    }

    /// `mel_basis` is the pack's `preprocessor.fb` tensor, flattened
    /// row-major with `n_bins` columns (must be `NEMOTRON_TRANSCRIBE_N_MELS x
    /// (N_FFT/2+1)`).
    pub(crate) fn from_pack_filterbank(
        mel_basis: &[f32],
    ) -> Result<Self, NemotronMelFrontendError> {
        let expected = NEMOTRON_TRANSCRIBE_N_MELS * FFT_BINS;
        if mel_basis.len() != expected {
            return Err(NemotronMelFrontendError::Overflow {
                count: mel_basis.len(),
            });
        }
        let mut mel_kbeg = vec![0usize; NEMOTRON_TRANSCRIBE_N_MELS];
        let mut mel_kend = vec![0usize; NEMOTRON_TRANSCRIBE_N_MELS];
        for m in 0..NEMOTRON_TRANSCRIBE_N_MELS {
            let row = &mel_basis[m * FFT_BINS..(m + 1) * FFT_BINS];
            let mut b = 0usize;
            while b < FFT_BINS && row[b] == 0.0 {
                b += 1;
            }
            let mut e = FFT_BINS;
            while e > b && row[e - 1] == 0.0 {
                e -= 1;
            }
            mel_kbeg[m] = b;
            mel_kend[m] = e;
        }
        let r2c = realfft::RealFftPlanner::<f32>::new().plan_fft_forward(NEMOTRON_TRANSCRIBE_N_FFT);
        Ok(Self {
            r2c,
            mel_basis: mel_basis.to_vec(),
            mel_kbeg,
            mel_kend,
            window: hann_periodic(NEMOTRON_TRANSCRIBE_WIN_LENGTH),
        })
    }

    pub(crate) fn output_frame_count(sample_count: usize) -> usize {
        // floor(n / 160) + 1 (the zero-padding contributes nothing beyond the
        // trailing frame).
        sample_count / NEMOTRON_TRANSCRIBE_HOP_LENGTH + 1
    }

    /// `sample_count` must match `samples.len()`; the sample rate is checked
    /// by the caller (the whole pipeline is 16 kHz mono).
    pub(crate) fn extract(
        &self,
        samples: &[f32],
    ) -> Result<NemotronMelFeatures, NemotronMelFrontendError> {
        if samples.is_empty() {
            return Err(NemotronMelFrontendError::Empty);
        }
        let sample_count = samples.len();
        let n_frames = Self::output_frame_count(sample_count);
        let mut pre = samples.to_vec();
        if sample_count >= 2 {
            for i in (1..sample_count).rev() {
                pre[i] -= PREEMPH * pre[i - 1];
            }
        }
        let n_fft = NEMOTRON_TRANSCRIBE_N_FFT;
        let hop = NEMOTRON_TRANSCRIBE_HOP_LENGTH;
        let win = NEMOTRON_TRANSCRIBE_WIN_LENGTH;
        let window = &self.window;
        let mut features = vec![0.0f32; n_frames * NEMOTRON_TRANSCRIBE_N_MELS];

        let mut fft_in = self.r2c.make_input_vec();
        let mut fft_out = self.r2c.make_output_vec();
        let mut scratch = self.r2c.make_scratch_vec();
        let r2c = &*self.r2c;

        // Zero-padded signal: n_fft/2 zeros | pre-emphasized samples | n_fft/2 zeros.
        let mut padded = vec![0.0f32; sample_count + n_fft];
        padded[n_fft / 2..n_fft / 2 + sample_count].copy_from_slice(&pre);

        for frame in 0..n_frames {
            let offset = frame * hop;
            for value in fft_in.iter_mut() {
                *value = 0.0;
            }
            for i in 0..win {
                fft_in[i] = padded[offset + i] * window[i];
            }
            r2c.process_with_scratch(&mut fft_in, &mut fft_out, &mut scratch)
                .map_err(|_| NemotronMelFrontendError::Overflow { count: n_fft })?;
            let row_start = frame * NEMOTRON_TRANSCRIBE_N_MELS;
            for m in 0..NEMOTRON_TRANSCRIBE_N_MELS {
                let kbeg = self.mel_kbeg[m];
                let kend = self.mel_kend[m];
                let row = &self.mel_basis[m * FFT_BINS..];
                let mut acc = 0.0f32;
                for k in kbeg..kend {
                    let re = fft_out[k].re;
                    let im = fft_out[k].im;
                    acc += row[k] * (re * re + im * im);
                }
                features[row_start + m] = (acc + LOG_ZERO_GUARD).ln();
            }
        }
        Ok(NemotronMelFeatures {
            features,
            frames: n_frames,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_hann_matches_reference_definition() {
        let window = hann_periodic(4);
        assert_eq!(window[0], 0.0);
        assert!((window[1] - 0.5).abs() < 1e-6);
        assert!((window[2] - 1.0).abs() < 1e-6);
        assert!((window[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn output_frame_count_matches_padded_formula() {
        assert_eq!(NemotronMelFrontend::output_frame_count(0), 1);
        assert_eq!(NemotronMelFrontend::output_frame_count(160), 2);
        assert_eq!(NemotronMelFrontend::output_frame_count(16000), 101);
    }
}

#[cfg(test)]
mod golden_tests {
    use super::*;

    /// `NEMOTRON_GOLDEN_PACK` must point at the `ad_q2_0_m` `.oasr` pack; the
    /// mel is compared against an independently computed reference dump.
    #[test]
    #[ignore = "requires the ad_q2_0_m pack"]
    fn jfk_mel_matches_reference_dump() {
        let Some(pack) = std::env::var_os("NEMOTRON_GOLDEN_PACK") else {
            eprintln!("set NEMOTRON_GOLDEN_PACK to the ad_q2_0_m .oasr pack");
            return;
        };
        let reference_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/nemotron-reference-mel.f32");
        let pack_path = std::path::PathBuf::from(&pack);
        let preflight =
            crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index(
                &pack_path,
            )
            .expect("preflight");
        let frontend = NemotronMelFrontend::from_preflight_fb(&preflight).expect("frontend");
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav"),
            "test",
            "test",
        )
        .expect("load jfk");
        let out = frontend.extract(&samples).expect("extract");
        let reference: Vec<f32> = std::fs::read(&reference_path)
            .expect("read reference mel dump (run tool to generate)")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(out.features.len(), reference.len());
        let mut max_abs = 0.0f32;
        for (a, b) in out.features.iter().zip(reference.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        eprintln!("max abs mel diff = {max_abs}");
        assert!(max_abs < 1e-3, "mel differs from reference by {max_abs}");
    }
}
