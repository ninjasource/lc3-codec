use crate::common::complex::Scaler;
#[allow(unused_imports)]
use num_traits::real::Real;

// checked against spec

/// Adjusts the loudness for all spectral lines in the frame
///
/// # Arguments
///
/// * `frame_num_bits` - Number of bits in the frame
/// * `fs_ind` - Sampling frequency index (e.g. 4 for 48khz)
/// * `global_gain_index` - Computed global gain index (e.g 204)
/// * `spec_lines` - All the spectral lines in a frame to be mutated
pub fn apply_global_gain(frame_num_bits: usize, fs_ind: usize, global_gain_index: usize, spec_lines: &mut [Scaler]) {
    let fs = fs_ind as i32 + 1;
    let nbits = frame_num_bits as i32;
    let gg_off = -((nbits / (10 * fs)).min(115)) - 105 - (5 * fs);
    let exponent: Scaler = (global_gain_index as Scaler + gg_off as Scaler) / 28.0;
    let gg = (10.0).powf(exponent);

    for f in spec_lines.iter_mut() {
        *f *= gg;
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn global_gain_decode() {
        let mut spec_lines = [1.0, 10.0, 100.0];

        apply_global_gain(1200, 4, 204, &mut spec_lines);

        assert_eq!(spec_lines, [61.0540199, 610.540199, 6105.40199])
    }
}
