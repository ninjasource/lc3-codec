use crate::common::complex::Scaler;

// checked against spec

/// Convert from floating point to signed integer by rounding to the
/// nearest integer value then clipping to max and min of a 16 bit integer
///
/// # Arguments
///
/// * `x_hat_ltpf` - Input samples from the output of the long term post filter
/// * `_bits_per_audio_sample_dec` - Bits per audio sample (e.g. 16). Assumed to be 16 and not currently used
/// * `x_hat_clip` - 16 bit integer output samples rounded and clipped
pub fn scale_and_round(x_hat_ltpf: &[Scaler], _bits_per_audio_sample_dec: usize, x_hat_clip: &mut [i16]) {
    // only 16 bit audio is supported so this is always 1 (otherwise we would multiply by the output scale)
    // let _output_scale = 1 << (-15 + _bits_per_audio_sample_dec as i32 - 1);

    for (to, from) in x_hat_clip.iter_mut().zip(x_hat_ltpf) {
        let tmp = if *from > 0. {
            (*from + 0.5) as i32
        } else {
            (*from - 0.5) as i32
        };

        *to = tmp.min(32767).max(-32768) as i16;
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn scale_and_round_test() {
        let x_hat_ltpf = [0.0, -0.4, -0.5, -0.6, 0.4, 0.5, 0.6, 32767.6, -32768.6];
        let mut x_hat_clip = [0; 9];

        scale_and_round(&x_hat_ltpf, 16, &mut x_hat_clip);

        assert_eq!(x_hat_clip, [0, 0, -1, -1, 0, 1, 1, 32767, -32768])
    }
}
