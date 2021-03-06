use super::side_info::LongTermPostFilterInfo;
use crate::{
    common::{
        complex::Scaler,
        config::{FrameDuration, Lc3Config},
    },
    tables::long_term_post_filter_coef::*,
};
#[allow(unused_imports)]
use num_traits::real::Real;

pub struct LongTermPostFilter<'a> {
    // constant (never changed after new())
    config: Lc3Config,
    num_mem_blocks: usize,
    c_den_zeros: &'a [Scaler], // used to clear a buffer
    norm: usize,

    // state (mutated every time run() is called)
    ltpf_active_prev: bool,           // true if ltpf was active in the previous frame
    block_start_index: usize,         // the current block in the circular buffer
    c_num: &'a mut [Scaler],          // numerator coeficients
    c_den: &'a mut [Scaler],          // denominator coeficients
    c_num_mem: &'a mut [Scaler],      // previous numerator coeficients
    c_den_mem: &'a mut [Scaler],      // previous denominator coeficients
    p_int_mem: usize,                 // previous pitch integer
    p_fr_mem: usize,                  // prefious pitch fraction
    x_hat_ltpf_mem: &'a mut [Scaler], // circular buffer containing 2x or 3x number of samples - critical working buffer
    x_hat_mem: &'a mut [Scaler], // circular buffer containing 2x or 3x number of samples - used to record the input into a chunk of memory and reference later
    scratch: &'a mut [Scaler], // a buffer to be used to temporarily store values when computing filters for the 5th case
}

struct LongTermPostFilterBufferLengths {
    pub num_mem_blocks: usize,
    pub x_hat_mem: usize,
    pub x_hat_ltpf_mem: usize,
    pub c_num: usize,
    pub c_den: usize,
    pub norm: usize,
    pub scratch: usize,
}

enum Transition {
    /// First case: inactive -> inactive
    Inactive,

    /// Second case: inactive -> active
    Activated,

    /// Third case: active -> inactive
    Deactivated,

    /// Forth case: active -> active with no pitch change
    ActiveNoPitchChange,

    /// Fifth case: active -> active with pitch change
    ActivePitchChanged,
}

impl<'a> LongTermPostFilter<'a> {
    pub fn new(config: Lc3Config, scaler_buf: &'a mut [Scaler]) -> (Self, &'a mut [Scaler]) {
        let len = Self::get_buffer_lengths(&config);
        let (x_hat_mem, scaler_buf) = scaler_buf.split_at_mut(len.x_hat_mem);
        let (x_hat_ltpf_mem, scaler_buf) = scaler_buf.split_at_mut(len.x_hat_ltpf_mem);

        let (c_num, scaler_buf) = scaler_buf.split_at_mut(len.c_num);
        let (c_den, scaler_buf) = scaler_buf.split_at_mut(len.c_den);
        let (c_num_mem, scaler_buf) = scaler_buf.split_at_mut(len.c_num);
        let (c_den_mem, scaler_buf) = scaler_buf.split_at_mut(len.c_den);
        let (c_den_zeros, scaler_buf) = scaler_buf.split_at_mut(len.c_den);
        let (scratch, scaler_buf) = scaler_buf.split_at_mut(len.scratch);

        for item in c_den_zeros.iter_mut() {
            *item = 0.0;
        }

        let p_int_mem = 0;
        let p_fr_mem = 0;
        let ltpf_active_prev = false;
        let block_start_index = 0;

        (
            Self {
                x_hat_mem,
                x_hat_ltpf_mem,
                c_num,
                c_den,
                c_num_mem,
                c_den_mem,
                p_int_mem,
                p_fr_mem,
                ltpf_active_prev,
                block_start_index,
                num_mem_blocks: len.num_mem_blocks,
                config,
                c_den_zeros,
                scratch,
                norm: len.norm,
            },
            scaler_buf,
        )
    }

    const fn get_buffer_lengths(config: &Lc3Config) -> LongTermPostFilterBufferLengths {
        // TODO: would probably be nicer to use the SamplingFrequency enum to avoid the unreachable branch
        let l_den = match config.fs {
            8000 => 4,
            16000 => 4,
            24000 => 6,
            32000 => 8,
            44100 => 11,
            48000 => 12,
            _ => unreachable!(),
        };

        let l_num = l_den - 2;

        let (num_mem_blocks, norm) = match config.n_ms {
            FrameDuration::TenMs => (2, config.nf / 4),
            FrameDuration::SevenPointFiveMs => (3, config.nf / 3),
        };

        let scratch = l_num + norm;

        LongTermPostFilterBufferLengths {
            num_mem_blocks,
            x_hat_mem: config.nf * num_mem_blocks,
            x_hat_ltpf_mem: config.nf * num_mem_blocks,
            c_num: l_num + 1,
            c_den: l_den + 1,
            norm,
            scratch,
        }
    }

    pub const fn calc_working_buffer_length(config: &Lc3Config) -> usize {
        let len = Self::get_buffer_lengths(config);
        len.c_den * 3 + len.c_num * 2 + len.x_hat_ltpf_mem + len.x_hat_mem + len.scratch
    }

    // returns (gain_ltpf, gain_ind)
    fn compute_gains_params(&self, nbits: usize) -> (Scaler, usize) {
        // gains parameters
        let t_nbits = match self.config.n_ms {
            FrameDuration::SevenPointFiveMs => (nbits as f64 * 10. / 7.5).round() as usize,
            FrameDuration::TenMs => nbits,
        };
        let sampling_factor = self.config.fs_ind * 80;
        if t_nbits < 320 + sampling_factor {
            (0.4, 0)
        } else if t_nbits < 400 + sampling_factor {
            (0.35, 1)
        } else if t_nbits < 480 + sampling_factor {
            (0.3, 2)
        } else if t_nbits < 560 + sampling_factor {
            (0.25, 3)
        } else {
            (0.0, 0) // the spec does not say what to do with gain_ind here
        }
    }

    // section 3.4.9.4 Filter parameters (checked with spec)
    // returns (pitch_integer, pitch_fraction) (aka p_int and p_fr)
    fn compute_filter_parameters(&self, info: &LongTermPostFilterInfo) -> (usize, usize) {
        if !info.is_active {
            // LTPF not active
            (0, 0)
        } else {
            let pitch_index = info.pitch_index;
            let (pitch_int, pitch_fr) = if pitch_index >= 440 {
                (pitch_index - 283, 0.0)
            } else if pitch_index >= 380 {
                let pitch_int = pitch_index / 2 - 63;
                let pitch_fr = (2 * pitch_index - 4 * pitch_int - 252) as f64;
                (pitch_int, pitch_fr)
            } else {
                let pitch_int = pitch_index / 4 + 32;
                let pitch_fr = (pitch_index + 128 - 4 * pitch_int) as f64;
                (pitch_int, pitch_fr)
            };

            let pitch = pitch_int as f64 + pitch_fr / 4.;
            let pitch_fs = pitch * (8000.0 * (self.config.fs as f64 / 8000.0).ceil() / 12800.0);
            let p_up = ((pitch_fs * 4.) + 0.5) as usize;
            let p_int = p_up / 4;
            let p_fr = p_up - 4 * p_int;
            (p_int, p_fr)
        }
    }

    // mutates c_num_mem, c_den_mem, c_num and c_den
    fn compute_filter_coeffs(&mut self, info: &LongTermPostFilterInfo, nbits: usize, pitch_frac: usize) {
        self.c_num_mem.copy_from_slice(self.c_num);
        self.c_den_mem.copy_from_slice(self.c_den);

        if !info.is_active {
            // LTPF not active, zero c_num and c_den
            // Note: c_num.len() < c_den.len() so we can use the same buffer
            self.c_num.copy_from_slice(&self.c_den_zeros[..self.c_num.len()]);
            self.c_den.copy_from_slice(self.c_den_zeros);
            return;
        }

        let (gain_ltpf, gain_ind) = self.compute_gains_params(nbits);

        // update index parameters for current frame
        let (tab_ltpf_num_fs, tab_ltpf_den_fs) = match self.config.fs {
            8000 => (
                TAB_LTPF_NUM_8000[gain_ind].as_slice(),
                TAB_LTPF_DEN_8000[pitch_frac].as_slice(),
            ),
            16000 => (
                TAB_LTPF_NUM_16000[gain_ind].as_slice(),
                TAB_LTPF_DEN_16000[pitch_frac].as_slice(),
            ),
            24000 => (
                TAB_LTPF_NUM_24000[gain_ind].as_slice(),
                TAB_LTPF_DEN_24000[pitch_frac].as_slice(),
            ),
            32000 => (
                TAB_LTPF_NUM_32000[gain_ind].as_slice(),
                TAB_LTPF_DEN_32000[pitch_frac].as_slice(),
            ),
            44100 => (
                TAB_LTPF_NUM_48000[gain_ind].as_slice(),
                TAB_LTPF_DEN_48000[pitch_frac].as_slice(),
            ),
            48000 => (
                TAB_LTPF_NUM_48000[gain_ind].as_slice(),
                TAB_LTPF_DEN_48000[pitch_frac].as_slice(),
            ),
            _ => panic!("Cannot lookup ltpf table. Invalid fs: {}", self.config.fs),
        };

        for (c_num, tab_num) in self.c_num.iter_mut().zip(tab_ltpf_num_fs) {
            *c_num = 0.85 * gain_ltpf * *tab_num;
        }

        for (c_den, tab_den) in self.c_den.iter_mut().zip(tab_ltpf_den_fs) {
            *c_den = gain_ltpf * *tab_den;
        }
    }

    fn wrap_if_negative(&self, index: i32) -> usize {
        if index < 0 {
            (index + self.num_mem_blocks as i32 * self.config.nf as i32) as usize
        } else {
            index as usize
        }
    }

    pub fn run(&mut self, info: &LongTermPostFilterInfo, nbits: usize, freq_samples: &mut [Scaler]) {
        let (pitch_int, pitch_frac) = self.compute_filter_parameters(info);
        self.compute_filter_coeffs(info, nbits, pitch_frac);

        // copy input into input circular buffer
        self.x_hat_mem[self.block_start_index..(self.block_start_index + self.config.nf)].copy_from_slice(freq_samples);

        let sample_2p5ms = if self.config.fs == 44100 {
            48000 / 400
        } else {
            self.config.fs / 400
        };

        let blk_start = self.block_start_index;
        let nf = self.config.nf;

        let transition = if !info.is_active && !self.ltpf_active_prev {
            Transition::Inactive
        } else if info.is_active && !self.ltpf_active_prev {
            Transition::Activated
        } else if !info.is_active && self.ltpf_active_prev {
            Transition::Deactivated
        } else if pitch_int == self.p_int_mem && pitch_frac == self.p_fr_mem {
            Transition::ActiveNoPitchChange
        } else {
            Transition::ActivePitchChanged
        };

        match transition {
            Transition::Inactive => {
                // transition case 1
                self.x_hat_ltpf_mem[blk_start..(blk_start + nf)]
                    .copy_from_slice(&self.x_hat_mem[blk_start..(blk_start + nf)]);
            }
            Transition::Activated => {
                // transition case 2
                for n in 0..sample_2p5ms {
                    self.x_hat_ltpf_mem[blk_start + n] = self.x_hat_mem[blk_start + n];
                    let mut filt_out = self.compute_filter(blk_start + n, pitch_int);
                    filt_out *= n as Scaler / self.norm as Scaler;
                    self.x_hat_ltpf_mem[blk_start + n] -= filt_out;
                }

                for n in sample_2p5ms..nf {
                    self.x_hat_ltpf_mem[blk_start + n] = self.x_hat_mem[blk_start + n];
                    let filt_out = self.compute_filter(blk_start + n, pitch_int);
                    self.x_hat_ltpf_mem[blk_start + n] -= filt_out;
                }
            }
            Transition::Deactivated => {
                // transision case 3
                self.deactive_first_2p5ms(sample_2p5ms, blk_start);

                for n in sample_2p5ms..nf {
                    self.x_hat_ltpf_mem[blk_start + n] = self.x_hat_mem[blk_start + n];
                }
            }
            Transition::ActiveNoPitchChange => {
                // transition case 4
                for n in 0..nf {
                    self.x_hat_ltpf_mem[blk_start + n] = self.x_hat_mem[blk_start + n];
                    let filt_out = self.compute_filter(blk_start + n, pitch_int);
                    self.x_hat_ltpf_mem[blk_start + n] -= filt_out;
                }
            }
            Transition::ActivePitchChanged => {
                // transition case 5
                self.deactive_first_2p5ms(sample_2p5ms, blk_start);
                self.activate_first_2p5ms_from_mem(blk_start, pitch_int, sample_2p5ms);

                for n in sample_2p5ms..nf {
                    self.x_hat_ltpf_mem[blk_start + n] = self.x_hat_mem[blk_start + n];
                    let filt_out = self.compute_filter(blk_start + n, pitch_int);
                    self.x_hat_ltpf_mem[blk_start + n] -= filt_out;
                }
            }
        }

        // copy to output
        freq_samples.copy_from_slice(&self.x_hat_ltpf_mem[blk_start..(blk_start + nf)]);

        // increment current block in block ring bufffer
        self.block_start_index += nf;
        if self.block_start_index > ((self.num_mem_blocks - 1) * nf) {
            self.block_start_index = 0;
        }

        // register updates
        self.ltpf_active_prev = info.is_active;
        self.p_int_mem = pitch_int;
        self.p_fr_mem = pitch_frac;
    }

    fn activate_first_2p5ms_from_mem(&mut self, blk_start: usize, pitch_int: usize, sample_2p5ms: usize) {
        let l_num = self.c_num.len() - 1;

        // make a copy of x_hat_ltpf_mem from -l_num to norm for the relevant block
        // and store it in scratch. Scratch is used for the numerator calculation
        if blk_start < l_num {
            // wrap
            let from = self.num_mem_blocks * self.config.nf - l_num;
            self.scratch[..l_num].copy_from_slice(&self.x_hat_ltpf_mem[from..(from + l_num)]);
            self.scratch[l_num..].copy_from_slice(&self.x_hat_ltpf_mem[..self.norm]);
        } else {
            self.scratch
                .copy_from_slice(&self.x_hat_ltpf_mem[blk_start - l_num..blk_start + self.norm]);
        }

        for n in 0..sample_2p5ms {
            self.x_hat_ltpf_mem[blk_start + n] = self.scratch[n + l_num];
            let l_den = self.c_den.len() as i32 - 1;
            let mut filt_out = 0.0;
            let start_index_num = l_num + n;
            for (k, c_num) in self.c_num.iter().enumerate() {
                let index = start_index_num as i32 - k as i32;
                filt_out += *c_num * self.scratch[index as usize];
            }
            let start_index_den = (blk_start + n) as i32 - pitch_int as i32 + l_den / 2;
            for (k, c_den) in self.c_den.iter().enumerate() {
                let index = start_index_den - k as i32;
                let index = self.wrap_if_negative(index);
                filt_out -= *c_den * self.x_hat_ltpf_mem[index];
            }
            filt_out *= n as Scaler / self.norm as Scaler;
            self.x_hat_ltpf_mem[blk_start + n] -= filt_out;
        }
    }

    fn compute_filter(&self, start_index: usize, pitch_int: usize) -> Scaler {
        let l_den = self.c_den.len() as i32 - 1;
        let mut filter_out = 0.0;
        for (k, c_num) in self.c_num.iter().enumerate() {
            let index = start_index as i32 - k as i32;
            let index = self.wrap_if_negative(index);
            filter_out += *c_num * self.x_hat_mem[index];
        }
        let start_index_den = start_index as i32 - pitch_int as i32 + l_den / 2;
        for (k, c_den) in self.c_den.iter().enumerate() {
            let index = start_index_den - k as i32;
            let index = self.wrap_if_negative(index);
            filter_out -= *c_den * self.x_hat_ltpf_mem[index];
        }

        filter_out
    }

    // same as compute_filter but using c_num_mem and c_den_mem
    fn compute_filter_mem(&self, start_index: usize, pitch_int: usize) -> Scaler {
        let l_den = self.c_den.len() as i32 - 1;
        let mut filter_out = 0.0;
        for (k, c_num) in self.c_num_mem.iter().enumerate() {
            let index = start_index as i32 - k as i32;
            let index = self.wrap_if_negative(index);
            filter_out += *c_num * self.x_hat_mem[index];
        }
        let start_index_den = start_index as i32 - pitch_int as i32 + l_den / 2;
        for (k, c_den) in self.c_den_mem.iter().enumerate() {
            let index = start_index_den - k as i32;
            let index = self.wrap_if_negative(index);
            filter_out -= *c_den * self.x_hat_ltpf_mem[index];
        }

        filter_out
    }

    fn deactive_first_2p5ms(&mut self, sample_2p5ms: usize, blk_start: usize) {
        for n in 0..sample_2p5ms {
            self.x_hat_ltpf_mem[blk_start + n] = self.x_hat_mem[blk_start + n];
            let mut filter_out = self.compute_filter_mem(blk_start + n, self.p_int_mem);
            filter_out *= 1.0 - (n as Scaler / self.norm as Scaler);
            self.x_hat_ltpf_mem[blk_start + n] -= filter_out;
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::common::config::{FrameDuration, SamplingFrequency};

    #[test]
    fn long_term_post_filter_activated() {
        const CONFIG: Lc3Config = Lc3Config::new(SamplingFrequency::Hz48000, FrameDuration::TenMs);
        const SCALER_LEN: usize = LongTermPostFilter::calc_working_buffer_length(&CONFIG);
        let mut scaler_buf = [0.; SCALER_LEN];
        let (mut post_filter, _) = LongTermPostFilter::new(CONFIG, &mut scaler_buf);
        #[rustfmt::skip]
        let mut freq_samples = [
            9855.562, 9917.566, 10005.261, 10120.888, 10234.55, 10361.796, 10491.287, 10620.813, 10750.864, 10872.629,
            11002.74, 11121.663, 11231.113, 11337.209, 11437.316, 11537.072, 11632.652, 11742.589, 11860.357,
            11991.356, 12136.03, 12264.641, 12397.722, 12528.338, 12648.783, 12775.773, 12899.189, 13023.282,
            13137.482, 13241.209, 13335.1455, 13402.426, 13469.602, 13546.963, 13626.523, 13710.95, 13781.829,
            13840.011, 13900.063, 13949.859, 13986.881, 14019.218, 14032.414, 14024.433, 14008.859, 13989.182,
            13961.398, 13912.517, 13855.993, 13804.8545, 13744.502, 13686.821, 13634.785, 13573.915, 13513.515,
            13445.138, 13366.799, 13300.5205, 13238.609, 13184.963, 13153.88, 13132.771, 13124.816, 13120.422,
            13104.559, 13082.388, 13037.949, 12979.922, 12920.758, 12845.244, 12775.473, 12721.137, 12665.74,
            12621.998, 12585.169, 12542.781, 12518.69, 12509.328, 12493.061, 12483.564, 12478.1045, 12474.463,
            12492.284, 12519.242, 12549.169, 12584.644, 12607.496, 12624.256, 12644.422, 12674.797, 12730.668,
            12795.609, 12879.023, 12987.815, 13089.303, 13201.322, 13310.01, 13395.618, 13492.285, 13572.554,
            13643.012, 13721.199, 13776.686, 13831.684, 13872.826, 13891.593, 13916.311, 13930.98, 13943.338,
            13953.295, 13943.105, 13934.076, 13943.393, 13951.496, 13947.896, 13943.783, 13920.041, 13878.515,
            13827.716, 13757.447, 13683.881, 13595.454, 13503.945, 13425.848, 13342.355, 13258.035, 13170.186,
            13082.378, 13006.727, 12927.238, 12847.024, 12779.564, 12716.449, 12659.647, 12617.696, 12568.529,
            12527.834, 12505.78, 12470.474, 12451.495, 12442.96, 12427.237, 12418.055, 12382.621, 12337.795, 12290.037,
            12226.481, 12162.639, 12079.796, 11988.431, 11894.545, 11797.618, 11711.063, 11618.033, 11504.882,
            11371.962, 11248.445, 11135.992, 11027.073, 10914.627, 10762.407, 10595.1, 10422.087, 10262.904, 10138.305,
            9991.471, 9842.319, 9700.6455, 9556.256, 9441.692, 9339.707, 9260.463, 9198.844, 9139.331, 9096.878,
            9051.312, 9000.729, 8934.586, 8842.711, 8739.059, 8623.649, 8494.162, 8341.745, 8182.9673, 8033.113,
            7890.5464, 7762.976, 7656.5522, 7577.397, 7512.7803, 7468.167, 7438.9585, 7392.6143, 7341.1367, 7286.0986,
            7213.8115, 7135.6313, 7053.1997, 6967.601, 6895.043, 6835.753, 6779.1045, 6737.633, 6700.1787, 6652.505,
            6610.9985, 6572.9966, 6537.1826, 6502.142, 6473.537, 6460.026, 6449.2593, 6440.4326, 6420.6733, 6392.3525,
            6369.301, 6341.985, 6318.461, 6287.427, 6237.29, 6191.657, 6160.076, 6134.14, 6109.088, 6082.9775,
            6050.0547, 6017.8057, 5984.0913, 5953.57, 5940.273, 5916.587, 5888.9756, 5869.027, 5835.652, 5809.4375,
            5795.1616, 5780.3423, 5765.359, 5749.959, 5737.257, 5705.277, 5651.1826, 5589.069, 5507.203, 5411.21,
            5310.16, 5202.7715, 5100.1772, 5002.1143, 4915.1626, 4847.8354, 4784.089, 4738.0273, 4701.5093, 4641.388,
            4579.9414, 4513.0796, 4440.5444, 4371.5537, 4279.019, 4183.7686, 4077.3438, 3942.913, 3807.1123, 3649.0535,
            3471.5417, 3300.851, 3133.4453, 2962.1482, 2781.6843, 2586.013, 2368.5417, 2155.0505, 1949.0055, 1742.4028,
            1544.5067, 1331.1459, 1118.0337, 915.6696, 711.80493, 519.69324, 319.71164, 137.62807, 2.213461,
            -100.40379, -151.82158, -179.75862, -208.9043, -219.8871, -226.34267, -232.57024, -239.98558, -266.843,
            -299.2365, -315.07404, -319.66702, -309.66202, -286.48257, -266.889, -239.64825, -206.53584, -173.83635,
            -143.72298, -142.50597, -153.62009, -164.21924, -187.73383, -201.14044, -218.58998, -242.7555, -253.40593,
            -278.79718, -310.7887, -346.9234, -409.37033, -477.39633, -552.88165, -638.8955, -716.6706, -795.39685,
            -891.4058, -1002.1742, -1116.8259, -1239.5403, -1366.8159, -1487.0908, -1603.3215, -1731.4861, -1889.7358,
            -2055.7173, -2212.8157, -2351.4463, -2455.8792, -2562.0364, -2667.2065, -2764.0796, -2868.059, -2935.8674,
            -2982.7478, -3033.1213, -3071.353, -3114.0203, -3134.3762, -3132.4204, -3145.5613, -3160.3367, -3179.9258,
            -3209.6628, -3234.5923, -3256.9905, -3281.801, -3314.2883, -3365.5205, -3429.076, -3491.5815, -3566.9045,
            -3645.547, -3708.0098, -3774.7349, -3835.4158, -3892.3599, -3975.3743, -4059.7576, -4155.2817, -4275.83,
            -4388.325, -4510.5923, -4631.0757, -4734.775, -4855.5386, -4972.5537, -5096.596, -5236.578, -5366.973,
            -5513.627, -5644.107, -5756.233, -5889.118, -5998.174, -6105.8247, -6233.732, -6342.7183, -6460.438,
            -6582.3433, -6681.794, -6789.526, -6898.3906, -6992.804, -7095.941, -7190.937, -7275.0874, -7365.499,
            -7425.3096, -7466.3027, -7505.8906, -7514.177, -7517.876, -7519.8433, -7499.0464, -7482.647, -7481.0547,
            -7500.747, -7543.6797, -7581.2563, -7601.7666, -7603.4404, -7587.517, -7584.735, -7592.989, -7596.2813,
            -7617.3364, -7644.5693, -7693.5303, -7785.3877, -7866.3823, -7942.4624, -8022.9243, -8083.197, -8151.433,
            -8230.286, -8320.681, -8424.271, -8519.547, -8623.19, -8727.481, -8817.017, -8907.424, -8981.751,
            -9052.048, -9145.353, -9226.115, -9289.694, -9348.728, -9391.421, -9436.57, -9478.702, -9514.521,
            -9550.879, -9572.372, -9616.987, -9683.013, -9731.245, -9782.168, -9822.187, -9840.694, -9868.124,
            -9896.476, -9920.14, -9956.94, -9995.62, -10034.032, -10079.254, -10102.077, -10115.083, -10130.439,
            -10125.355, -10124.556, -10123.945, -10125.191, -10153.799, -10187.783, -10244.149, -10327.773, -10410.779,
            -10504.616, -10603.6045, -10699.09, -10803.099, -10906.434, -10984.714, -11041.041, -11080.222, -11094.879,
            -11109.067, -11127.135, -11133.012, -11131.895, -11127.913, -11125.259, -11128.385,
        ];
        let info = LongTermPostFilterInfo {
            pitch_present: true,
            is_active: true,
            pitch_index: 473,
        };

        post_filter.run(&info, 600, &mut freq_samples);
    }

    #[test]
    fn long_term_post_filter_full_cycle() {
        const CONFIG: Lc3Config = Lc3Config::new(SamplingFrequency::Hz48000, FrameDuration::TenMs);
        const SCALER_LEN: usize = LongTermPostFilter::calc_working_buffer_length(&CONFIG);
        let mut scaler_buf = [0.; SCALER_LEN];
        let (mut post_filter, _) = LongTermPostFilter::new(CONFIG, &mut scaler_buf);

        // Inactive - transition case 1
        let info = LongTermPostFilterInfo::new(false, true, 134);
        #[rustfmt::skip]
        let mut freq_samples = [
            58.819393, 109.76993, 146.13187, 171.12459, 183.76201, 183.26862, 171.50392, 153.74846, 136.00961,
            121.80272, 111.48834, 103.990265, 99.532196, 101.14639, 112.7906, 135.5299, 165.0787, 196.15884, 224.9827,
            250.5431, 274.52618, 295.1435, 305.5937, 295.62616, 258.9342, 198.34666, 125.70483, 57.751877, 2.6190033,
            -38.589508, -72.57579, -108.70824, -146.6611, -187.3291, -224.06477, -247.3674, -258.3492, -258.02042,
            -254.71426, -254.75, -258.58203, -266.91016, -274.858, -281.69357, -284.87228, -280.90448, -267.02454,
            -238.05745, -200.51181, -161.06471, -119.168274, -79.19075, -46.14868, -15.239769, 16.860542, 45.04934,
            67.25763, 78.887856, 78.33379, 70.21389, 57.012802, 39.363586, 11.711079, -29.732513, -87.21059, -155.3208,
            -220.37448, -278.96088, -329.77835, -379.429, -449.33652, -549.7191, -667.98425, -796.4076, -931.855,
            -1058.7336, -1173.297, -1271.1577, -1346.4891, -1396.2488, -1409.3469, -1392.7959, -1351.6752, -1300.4618,
            -1253.5239, -1203.5798, -1160.2394, -1113.6416, -1069.9187, -1049.7609, -1031.5128, -1011.4468, -982.8891,
            -946.4585, -925.1284, -917.4631, -924.5065, -938.4893, -950.4643, -960.33044, -968.0591, -977.982,
            -983.4437, -989.99567, -995.76587, -1001.3642, -1033.6366, -1101.1494, -1201.4219, -1330.1234, -1481.5342,
            -1642.0104, -1797.366, -1950.472, -2099.4565, -2241.3704, -2361.2925, -2443.267, -2484.2148, -2495.6113,
            -2510.2786, -2527.6897, -2545.258, -2574.0662, -2609.4634, -2658.521, -2715.6897, -2783.1514, -2872.6453,
            -2965.771, -3048.6228, -3117.3728, -3174.8276, -3217.3108, -3241.4304, -3260.4983, -3277.638, -3294.413,
            -3318.0466, -3358.9788, -3413.3137, -3474.827, -3558.0674, -3660.628, -3782.3652, -3916.5693, -4038.0527,
            -4142.371, -4224.248, -4295.353, -4364.9214, -4408.7207, -4420.981, -4400.163, -4336.8477, -4229.314,
            -4091.1794, -3955.2656, -3836.133, -3733.7039, -3646.8354, -3574.509, -3536.625, -3540.0002, -3563.7488,
            -3599.8533, -3637.89, -3649.3171, -3624.2083, -3572.5337, -3504.5889, -3428.7239, -3340.7769, -3231.3267,
            -3098.3306, -2937.7249, -2748.9712, -2537.5615, -2305.6973, -2072.6257, -1863.025, -1675.0913, -1509.5488,
            -1372.1879, -1260.2831, -1176.7883, -1116.0103, -1071.9922, -1037.3132, -985.8291, -905.1432, -803.6178,
            -696.98303, -603.00757, -528.0028, -461.94656, -384.36053, -291.32056, -188.4169, -72.39171, 49.311478,
            175.08928, 316.87576, 478.5452, 656.88434, 844.65094, 1037.9285, 1219.6411, 1376.7842, 1531.121, 1679.177,
            1804.7701, 1918.188, 2010.5298, 2077.183, 2129.5835, 2166.594, 2191.2551, 2210.8132, 2224.607, 2222.8464,
            2207.3958, 2178.7554, 2118.0376, 2027.5754, 1922.1364, 1810.9786, 1703.194, 1599.5087, 1494.5358,
            1384.9102, 1284.124, 1193.5399, 1110.5286, 1044.7048, 974.7428, 886.38025, 781.4219, 660.6767, 536.1736,
            404.93625, 270.1938, 128.7827, -32.117737, -203.27695, -375.92297, -535.7374, -681.557, -819.9927,
            -952.56104, -1088.791, -1211.329, -1309.9143, -1378.6849, -1404.791, -1406.0771, -1384.1198, -1340.9026,
            -1287.4758, -1225.4861, -1178.9884, -1149.8586, -1127.0452, -1115.731, -1111.7362, -1115.7399, -1128.1174,
            -1149.6292, -1177.5052, -1198.2435, -1209.314, -1209.0099, -1201.4939, -1194.4609, -1179.8419, -1154.482,
            -1116.5957, -1062.1146, -1001.0329, -943.9992, -900.2267, -868.4272, -825.7453, -764.4547, -682.30524,
            -568.10474, -436.0474, -286.67776, -106.914795, 83.19826, 276.07922, 466.4378, 632.19763, 779.5844,
            917.63086, 1048.77, 1183.317, 1319.8423, 1454.6073, 1585.9534, 1707.1294, 1809.0703, 1892.3329, 1956.999,
            1991.0143, 2006.0994, 2015.8397, 2017.2856, 2029.568, 2052.4868, 2073.895, 2107.8994, 2151.79, 2194.9524,
            2239.1438, 2287.6907, 2343.8843, 2407.795, 2478.684, 2553.2815, 2642.5505, 2751.9863, 2859.4126, 2963.1113,
            3070.0378, 3171.3374, 3262.7744, 3330.8076, 3369.4229, 3388.7988, 3391.3801, 3386.008, 3373.4111,
            3352.7227, 3343.45, 3340.4985, 3342.526, 3373.8386, 3430.9912, 3504.0479, 3579.6328, 3640.3928, 3686.6199,
            3715.406, 3741.6917, 3764.9707, 3748.1313, 3686.9075, 3591.1206, 3468.6372, 3336.0498, 3194.734, 3044.054,
            2889.3435, 2734.3938, 2584.6794, 2455.0115, 2355.9702, 2287.3074, 2248.5671, 2225.6858, 2214.5474,
            2213.863, 2207.5273, 2203.8284, 2202.1301, 2193.2034, 2187.212, 2182.3528, 2183.316, 2188.407, 2187.3374,
            2199.2312, 2239.2524, 2307.0513, 2390.8054, 2470.9495, 2538.7205, 2582.4238, 2602.7004, 2611.9116,
            2609.9634, 2594.4136, 2551.3035, 2475.3464, 2388.1438, 2300.876, 2215.2773, 2132.4849, 2045.0223,
            1945.9332, 1838.088, 1739.0056, 1663.0178, 1613.2725, 1592.9021, 1601.1526, 1630.0334, 1662.0596,
            1682.6108, 1689.0599, 1676.5842, 1635.6819, 1561.4032, 1462.4318, 1349.1211, 1223.0033, 1091.6134,
            969.89044, 867.5361, 780.41626, 704.90314, 639.86633, 571.3017, 497.18524, 430.50333, 376.43744, 335.329,
            301.84308, 266.11346, 216.1726, 145.90388, 69.40163, 6.2822857, -44.526463, -94.935646, -145.99371,
            -203.92838, -276.31796, -350.2114, -413.33847, -467.1132, -518.0094, -567.5848, -615.0106, -659.67163,
            -685.6454, -683.87933, -660.50757, -622.7549, -592.06635, -574.89197, -560.0859, -550.97235, -547.23315,
            -550.5877, -565.651, -587.4596, -621.71246, -670.6035, -727.52783, -795.5536, -871.09656, -942.68823,
            -1009.519, -1076.462, -1145.0883, -1220.204, -1299.5819, -1366.1815, -1413.3549, -1445.462, -1465.7451,
            -1478.4558, -1484.0813, -1474.9672, -1438.5527, -1378.9619, -1312.5193, -1249.6793, -1199.3994, -1159.8258,
            -1123.6235, -1077.1719, -1010.4406, -935.00116, -851.9267, -762.7784, -672.0623, -572.3121,
        ];
        post_filter.run(&info, 320, &mut freq_samples);
        #[rustfmt::skip]
        let freq_samples_expected = [
            58.819393, 109.76993, 146.13187, 171.12459, 183.76201, 183.26862, 171.50392, 153.74846, 136.00961,
            121.80272, 111.48834, 103.990265, 99.532196, 101.14639, 112.7906, 135.5299, 165.0787, 196.15884, 224.9827,
            250.5431, 274.52618, 295.1435, 305.5937, 295.62616, 258.9342, 198.34666, 125.70483, 57.751877, 2.6190033,
            -38.589508, -72.57579, -108.70824, -146.6611, -187.3291, -224.06477, -247.3674, -258.3492, -258.02042,
            -254.71426, -254.75, -258.58203, -266.91016, -274.858, -281.69357, -284.87228, -280.90448, -267.02454,
            -238.05745, -200.51181, -161.06471, -119.168274, -79.19075, -46.14868, -15.239769, 16.860542, 45.04934,
            67.25763, 78.887856, 78.33379, 70.21389, 57.012802, 39.363586, 11.711079, -29.732513, -87.21059, -155.3208,
            -220.37448, -278.96088, -329.77835, -379.429, -449.33652, -549.7191, -667.98425, -796.4076, -931.855,
            -1058.7336, -1173.297, -1271.1577, -1346.4891, -1396.2488, -1409.3469, -1392.7959, -1351.6752, -1300.4618,
            -1253.5239, -1203.5798, -1160.2394, -1113.6416, -1069.9187, -1049.7609, -1031.5128, -1011.4468, -982.8891,
            -946.4585, -925.1284, -917.4631, -924.5065, -938.4893, -950.4643, -960.33044, -968.0591, -977.982,
            -983.4437, -989.99567, -995.76587, -1001.3642, -1033.6366, -1101.1494, -1201.4219, -1330.1234, -1481.5342,
            -1642.0104, -1797.366, -1950.472, -2099.4565, -2241.3704, -2361.2925, -2443.267, -2484.2148, -2495.6113,
            -2510.2786, -2527.6897, -2545.258, -2574.0662, -2609.4634, -2658.521, -2715.6897, -2783.1514, -2872.6453,
            -2965.771, -3048.6228, -3117.3728, -3174.8276, -3217.3108, -3241.4304, -3260.4983, -3277.638, -3294.413,
            -3318.0466, -3358.9788, -3413.3137, -3474.827, -3558.0674, -3660.628, -3782.3652, -3916.5693, -4038.0527,
            -4142.371, -4224.248, -4295.353, -4364.9214, -4408.7207, -4420.981, -4400.163, -4336.8477, -4229.314,
            -4091.1794, -3955.2656, -3836.133, -3733.7039, -3646.8354, -3574.509, -3536.625, -3540.0002, -3563.7488,
            -3599.8533, -3637.89, -3649.3171, -3624.2083, -3572.5337, -3504.5889, -3428.7239, -3340.7769, -3231.3267,
            -3098.3306, -2937.7249, -2748.9712, -2537.5615, -2305.6973, -2072.6257, -1863.025, -1675.0913, -1509.5488,
            -1372.1879, -1260.2831, -1176.7883, -1116.0103, -1071.9922, -1037.3132, -985.8291, -905.1432, -803.6178,
            -696.98303, -603.00757, -528.0028, -461.94656, -384.36053, -291.32056, -188.4169, -72.39171, 49.311478,
            175.08928, 316.87576, 478.5452, 656.88434, 844.65094, 1037.9285, 1219.6411, 1376.7842, 1531.121, 1679.177,
            1804.7701, 1918.188, 2010.5298, 2077.183, 2129.5835, 2166.594, 2191.2551, 2210.8132, 2224.607, 2222.8464,
            2207.3958, 2178.7554, 2118.0376, 2027.5754, 1922.1364, 1810.9786, 1703.194, 1599.5087, 1494.5358,
            1384.9102, 1284.124, 1193.5399, 1110.5286, 1044.7048, 974.7428, 886.38025, 781.4219, 660.6767, 536.1736,
            404.93625, 270.1938, 128.7827, -32.117737, -203.27695, -375.92297, -535.7374, -681.557, -819.9927,
            -952.56104, -1088.791, -1211.329, -1309.9143, -1378.6849, -1404.791, -1406.0771, -1384.1198, -1340.9026,
            -1287.4758, -1225.4861, -1178.9884, -1149.8586, -1127.0452, -1115.731, -1111.7362, -1115.7399, -1128.1174,
            -1149.6292, -1177.5052, -1198.2435, -1209.314, -1209.0099, -1201.4939, -1194.4609, -1179.8419, -1154.482,
            -1116.5957, -1062.1146, -1001.0329, -943.9992, -900.2267, -868.4272, -825.7453, -764.4547, -682.30524,
            -568.10474, -436.0474, -286.67776, -106.914795, 83.19826, 276.07922, 466.4378, 632.19763, 779.5844,
            917.63086, 1048.77, 1183.317, 1319.8423, 1454.6073, 1585.9534, 1707.1294, 1809.0703, 1892.3329, 1956.999,
            1991.0143, 2006.0994, 2015.8397, 2017.2856, 2029.568, 2052.4868, 2073.895, 2107.8994, 2151.79, 2194.9524,
            2239.1438, 2287.6907, 2343.8843, 2407.795, 2478.684, 2553.2815, 2642.5505, 2751.9863, 2859.4126, 2963.1113,
            3070.0378, 3171.3374, 3262.7744, 3330.8076, 3369.4229, 3388.7988, 3391.3801, 3386.008, 3373.4111,
            3352.7227, 3343.45, 3340.4985, 3342.526, 3373.8386, 3430.9912, 3504.0479, 3579.6328, 3640.3928, 3686.6199,
            3715.406, 3741.6917, 3764.9707, 3748.1313, 3686.9075, 3591.1206, 3468.6372, 3336.0498, 3194.734, 3044.054,
            2889.3435, 2734.3938, 2584.6794, 2455.0115, 2355.9702, 2287.3074, 2248.5671, 2225.6858, 2214.5474,
            2213.863, 2207.5273, 2203.8284, 2202.1301, 2193.2034, 2187.212, 2182.3528, 2183.316, 2188.407, 2187.3374,
            2199.2312, 2239.2524, 2307.0513, 2390.8054, 2470.9495, 2538.7205, 2582.4238, 2602.7004, 2611.9116,
            2609.9634, 2594.4136, 2551.3035, 2475.3464, 2388.1438, 2300.876, 2215.2773, 2132.4849, 2045.0223,
            1945.9332, 1838.088, 1739.0056, 1663.0178, 1613.2725, 1592.9021, 1601.1526, 1630.0334, 1662.0596,
            1682.6108, 1689.0599, 1676.5842, 1635.6819, 1561.4032, 1462.4318, 1349.1211, 1223.0033, 1091.6134,
            969.89044, 867.5361, 780.41626, 704.90314, 639.86633, 571.3017, 497.18524, 430.50333, 376.43744, 335.329,
            301.84308, 266.11346, 216.1726, 145.90388, 69.40163, 6.2822857, -44.526463, -94.935646, -145.99371,
            -203.92838, -276.31796, -350.2114, -413.33847, -467.1132, -518.0094, -567.5848, -615.0106, -659.67163,
            -685.6454, -683.87933, -660.50757, -622.7549, -592.06635, -574.89197, -560.0859, -550.97235, -547.23315,
            -550.5877, -565.651, -587.4596, -621.71246, -670.6035, -727.52783, -795.5536, -871.09656, -942.68823,
            -1009.519, -1076.462, -1145.0883, -1220.204, -1299.5819, -1366.1815, -1413.3549, -1445.462, -1465.7451,
            -1478.4558, -1484.0813, -1474.9672, -1438.5527, -1378.9619, -1312.5193, -1249.6793, -1199.3994, -1159.8258,
            -1123.6235, -1077.1719, -1010.4406, -935.00116, -851.9267, -762.7784, -672.0623, -572.3121,
        ];
        assert_eq!(freq_samples, freq_samples_expected);

        // Inactive - transition case 1
        let info = LongTermPostFilterInfo::new(false, true, 132);
        #[rustfmt::skip]
        let mut freq_samples = [
            58.819393, 109.76993, 146.13187, 171.12459, 183.76201, 183.26862, 171.50392, 153.74846, 136.00961,
            121.80272, 111.48834, 103.990265, 99.532196, 101.14639, 112.7906, 135.5299, 165.0787, 196.15884, 224.9827,
            250.5431, 274.52618, 295.1435, 305.5937, 295.62616, 258.9342, 198.34666, 125.70483, 57.751877, 2.6190033,
            -38.589508, -72.57579, -108.70824, -146.6611, -187.3291, -224.06477, -247.3674, -258.3492, -258.02042,
            -254.71426, -254.75, -258.58203, -266.91016, -274.858, -281.69357, -284.87228, -280.90448, -267.02454,
            -238.05745, -200.51181, -161.06471, -119.168274, -79.19075, -46.14868, -15.239769, 16.860542, 45.04934,
            67.25763, 78.887856, 78.33379, 70.21389, 57.012802, 39.363586, 11.711079, -29.732513, -87.21059, -155.3208,
            -220.37448, -278.96088, -329.77835, -379.429, -449.33652, -549.7191, -667.98425, -796.4076, -931.855,
            -1058.7336, -1173.297, -1271.1577, -1346.4891, -1396.2488, -1409.3469, -1392.7959, -1351.6752, -1300.4618,
            -1253.5239, -1203.5798, -1160.2394, -1113.6416, -1069.9187, -1049.7609, -1031.5128, -1011.4468, -982.8891,
            -946.4585, -925.1284, -917.4631, -924.5065, -938.4893, -950.4643, -960.33044, -968.0591, -977.982,
            -983.4437, -989.99567, -995.76587, -1001.3642, -1033.6366, -1101.1494, -1201.4219, -1330.1234, -1481.5342,
            -1642.0104, -1797.366, -1950.472, -2099.4565, -2241.3704, -2361.2925, -2443.267, -2484.2148, -2495.6113,
            -2510.2786, -2527.6897, -2545.258, -2574.0662, -2609.4634, -2658.521, -2715.6897, -2783.1514, -2872.6453,
            -2965.771, -3048.6228, -3117.3728, -3174.8276, -3217.3108, -3241.4304, -3260.4983, -3277.638, -3294.413,
            -3318.0466, -3358.9788, -3413.3137, -3474.827, -3558.0674, -3660.628, -3782.3652, -3916.5693, -4038.0527,
            -4142.371, -4224.248, -4295.353, -4364.9214, -4408.7207, -4420.981, -4400.163, -4336.8477, -4229.314,
            -4091.1794, -3955.2656, -3836.133, -3733.7039, -3646.8354, -3574.509, -3536.625, -3540.0002, -3563.7488,
            -3599.8533, -3637.89, -3649.3171, -3624.2083, -3572.5337, -3504.5889, -3428.7239, -3340.7769, -3231.3267,
            -3098.3306, -2937.7249, -2748.9712, -2537.5615, -2305.6973, -2072.6257, -1863.025, -1675.0913, -1509.5488,
            -1372.1879, -1260.2831, -1176.7883, -1116.0103, -1071.9922, -1037.3132, -985.8291, -905.1432, -803.6178,
            -696.98303, -603.00757, -528.0028, -461.94656, -384.36053, -291.32056, -188.4169, -72.39171, 49.311478,
            175.08928, 316.87576, 478.5452, 656.88434, 844.65094, 1037.9285, 1219.6411, 1376.7842, 1531.121, 1679.177,
            1804.7701, 1918.188, 2010.5298, 2077.183, 2129.5835, 2166.594, 2191.2551, 2210.8132, 2224.607, 2222.8464,
            2207.3958, 2178.7554, 2118.0376, 2027.5754, 1922.1364, 1810.9786, 1703.194, 1599.5087, 1494.5358,
            1384.9102, 1284.124, 1193.5399, 1110.5286, 1044.7048, 974.7428, 886.38025, 781.4219, 660.6767, 536.1736,
            404.93625, 270.1938, 128.7827, -32.117737, -203.27695, -375.92297, -535.7374, -681.557, -819.9927,
            -952.56104, -1088.791, -1211.329, -1309.9143, -1378.6849, -1404.791, -1406.0771, -1384.1198, -1340.9026,
            -1287.4758, -1225.4861, -1178.9884, -1149.8586, -1127.0452, -1115.731, -1111.7362, -1115.7399, -1128.1174,
            -1149.6292, -1177.5052, -1198.2435, -1209.314, -1209.0099, -1201.4939, -1194.4609, -1179.8419, -1154.482,
            -1116.5957, -1062.1146, -1001.0329, -943.9992, -900.2267, -868.4272, -825.7453, -764.4547, -682.30524,
            -568.10474, -436.0474, -286.67776, -106.914795, 83.19826, 276.07922, 466.4378, 632.19763, 779.5844,
            917.63086, 1048.77, 1183.317, 1319.8423, 1454.6073, 1585.9534, 1707.1294, 1809.0703, 1892.3329, 1956.999,
            1991.0143, 2006.0994, 2015.8397, 2017.2856, 2029.568, 2052.4868, 2073.895, 2107.8994, 2151.79, 2194.9524,
            2239.1438, 2287.6907, 2343.8843, 2407.795, 2478.684, 2553.2815, 2642.5505, 2751.9863, 2859.4126, 2963.1113,
            3070.0378, 3171.3374, 3262.7744, 3330.8076, 3369.4229, 3388.7988, 3391.3801, 3386.008, 3373.4111,
            3352.7227, 3343.45, 3340.4985, 3342.526, 3373.8386, 3430.9912, 3504.0479, 3579.6328, 3640.3928, 3686.6199,
            3715.406, 3741.6917, 3764.9707, 3748.1313, 3686.9075, 3591.1206, 3468.6372, 3336.0498, 3194.734, 3044.054,
            2889.3435, 2734.3938, 2584.6794, 2455.0115, 2355.9702, 2287.3074, 2248.5671, 2225.6858, 2214.5474,
            2213.863, 2207.5273, 2203.8284, 2202.1301, 2193.2034, 2187.212, 2182.3528, 2183.316, 2188.407, 2187.3374,
            2199.2312, 2239.2524, 2307.0513, 2390.8054, 2470.9495, 2538.7205, 2582.4238, 2602.7004, 2611.9116,
            2609.9634, 2594.4136, 2551.3035, 2475.3464, 2388.1438, 2300.876, 2215.2773, 2132.4849, 2045.0223,
            1945.9332, 1838.088, 1739.0056, 1663.0178, 1613.2725, 1592.9021, 1601.1526, 1630.0334, 1662.0596,
            1682.6108, 1689.0599, 1676.5842, 1635.6819, 1561.4032, 1462.4318, 1349.1211, 1223.0033, 1091.6134,
            969.89044, 867.5361, 780.41626, 704.90314, 639.86633, 571.3017, 497.18524, 430.50333, 376.43744, 335.329,
            301.84308, 266.11346, 216.1726, 145.90388, 69.40163, 6.2822857, -44.526463, -94.935646, -145.99371,
            -203.92838, -276.31796, -350.2114, -413.33847, -467.1132, -518.0094, -567.5848, -615.0106, -659.67163,
            -685.6454, -683.87933, -660.50757, -622.7549, -592.06635, -574.89197, -560.0859, -550.97235, -547.23315,
            -550.5877, -565.651, -587.4596, -621.71246, -670.6035, -727.52783, -795.5536, -871.09656, -942.68823,
            -1009.519, -1076.462, -1145.0883, -1220.204, -1299.5819, -1366.1815, -1413.3549, -1445.462, -1465.7451,
            -1478.4558, -1484.0813, -1474.9672, -1438.5527, -1378.9619, -1312.5193, -1249.6793, -1199.3994, -1159.8258,
            -1123.6235, -1077.1719, -1010.4406, -935.00116, -851.9267, -762.7784, -672.0623, -572.3121,
        ];
        post_filter.run(&info, 320, &mut freq_samples);
        #[rustfmt::skip]
        let freq_samples_expected = [
            58.819393, 109.76993, 146.13187, 171.12459, 183.76201, 183.26862, 171.50392, 153.74846, 136.00961,
            121.80272, 111.48834, 103.990265, 99.532196, 101.14639, 112.7906, 135.5299, 165.0787, 196.15884, 224.9827,
            250.5431, 274.52618, 295.1435, 305.5937, 295.62616, 258.9342, 198.34666, 125.70483, 57.751877, 2.6190033,
            -38.589508, -72.57579, -108.70824, -146.6611, -187.3291, -224.06477, -247.3674, -258.3492, -258.02042,
            -254.71426, -254.75, -258.58203, -266.91016, -274.858, -281.69357, -284.87228, -280.90448, -267.02454,
            -238.05745, -200.51181, -161.06471, -119.168274, -79.19075, -46.14868, -15.239769, 16.860542, 45.04934,
            67.25763, 78.887856, 78.33379, 70.21389, 57.012802, 39.363586, 11.711079, -29.732513, -87.21059, -155.3208,
            -220.37448, -278.96088, -329.77835, -379.429, -449.33652, -549.7191, -667.98425, -796.4076, -931.855,
            -1058.7336, -1173.297, -1271.1577, -1346.4891, -1396.2488, -1409.3469, -1392.7959, -1351.6752, -1300.4618,
            -1253.5239, -1203.5798, -1160.2394, -1113.6416, -1069.9187, -1049.7609, -1031.5128, -1011.4468, -982.8891,
            -946.4585, -925.1284, -917.4631, -924.5065, -938.4893, -950.4643, -960.33044, -968.0591, -977.982,
            -983.4437, -989.99567, -995.76587, -1001.3642, -1033.6366, -1101.1494, -1201.4219, -1330.1234, -1481.5342,
            -1642.0104, -1797.366, -1950.472, -2099.4565, -2241.3704, -2361.2925, -2443.267, -2484.2148, -2495.6113,
            -2510.2786, -2527.6897, -2545.258, -2574.0662, -2609.4634, -2658.521, -2715.6897, -2783.1514, -2872.6453,
            -2965.771, -3048.6228, -3117.3728, -3174.8276, -3217.3108, -3241.4304, -3260.4983, -3277.638, -3294.413,
            -3318.0466, -3358.9788, -3413.3137, -3474.827, -3558.0674, -3660.628, -3782.3652, -3916.5693, -4038.0527,
            -4142.371, -4224.248, -4295.353, -4364.9214, -4408.7207, -4420.981, -4400.163, -4336.8477, -4229.314,
            -4091.1794, -3955.2656, -3836.133, -3733.7039, -3646.8354, -3574.509, -3536.625, -3540.0002, -3563.7488,
            -3599.8533, -3637.89, -3649.3171, -3624.2083, -3572.5337, -3504.5889, -3428.7239, -3340.7769, -3231.3267,
            -3098.3306, -2937.7249, -2748.9712, -2537.5615, -2305.6973, -2072.6257, -1863.025, -1675.0913, -1509.5488,
            -1372.1879, -1260.2831, -1176.7883, -1116.0103, -1071.9922, -1037.3132, -985.8291, -905.1432, -803.6178,
            -696.98303, -603.00757, -528.0028, -461.94656, -384.36053, -291.32056, -188.4169, -72.39171, 49.311478,
            175.08928, 316.87576, 478.5452, 656.88434, 844.65094, 1037.9285, 1219.6411, 1376.7842, 1531.121, 1679.177,
            1804.7701, 1918.188, 2010.5298, 2077.183, 2129.5835, 2166.594, 2191.2551, 2210.8132, 2224.607, 2222.8464,
            2207.3958, 2178.7554, 2118.0376, 2027.5754, 1922.1364, 1810.9786, 1703.194, 1599.5087, 1494.5358,
            1384.9102, 1284.124, 1193.5399, 1110.5286, 1044.7048, 974.7428, 886.38025, 781.4219, 660.6767, 536.1736,
            404.93625, 270.1938, 128.7827, -32.117737, -203.27695, -375.92297, -535.7374, -681.557, -819.9927,
            -952.56104, -1088.791, -1211.329, -1309.9143, -1378.6849, -1404.791, -1406.0771, -1384.1198, -1340.9026,
            -1287.4758, -1225.4861, -1178.9884, -1149.8586, -1127.0452, -1115.731, -1111.7362, -1115.7399, -1128.1174,
            -1149.6292, -1177.5052, -1198.2435, -1209.314, -1209.0099, -1201.4939, -1194.4609, -1179.8419, -1154.482,
            -1116.5957, -1062.1146, -1001.0329, -943.9992, -900.2267, -868.4272, -825.7453, -764.4547, -682.30524,
            -568.10474, -436.0474, -286.67776, -106.914795, 83.19826, 276.07922, 466.4378, 632.19763, 779.5844,
            917.63086, 1048.77, 1183.317, 1319.8423, 1454.6073, 1585.9534, 1707.1294, 1809.0703, 1892.3329, 1956.999,
            1991.0143, 2006.0994, 2015.8397, 2017.2856, 2029.568, 2052.4868, 2073.895, 2107.8994, 2151.79, 2194.9524,
            2239.1438, 2287.6907, 2343.8843, 2407.795, 2478.684, 2553.2815, 2642.5505, 2751.9863, 2859.4126, 2963.1113,
            3070.0378, 3171.3374, 3262.7744, 3330.8076, 3369.4229, 3388.7988, 3391.3801, 3386.008, 3373.4111,
            3352.7227, 3343.45, 3340.4985, 3342.526, 3373.8386, 3430.9912, 3504.0479, 3579.6328, 3640.3928, 3686.6199,
            3715.406, 3741.6917, 3764.9707, 3748.1313, 3686.9075, 3591.1206, 3468.6372, 3336.0498, 3194.734, 3044.054,
            2889.3435, 2734.3938, 2584.6794, 2455.0115, 2355.9702, 2287.3074, 2248.5671, 2225.6858, 2214.5474,
            2213.863, 2207.5273, 2203.8284, 2202.1301, 2193.2034, 2187.212, 2182.3528, 2183.316, 2188.407, 2187.3374,
            2199.2312, 2239.2524, 2307.0513, 2390.8054, 2470.9495, 2538.7205, 2582.4238, 2602.7004, 2611.9116,
            2609.9634, 2594.4136, 2551.3035, 2475.3464, 2388.1438, 2300.876, 2215.2773, 2132.4849, 2045.0223,
            1945.9332, 1838.088, 1739.0056, 1663.0178, 1613.2725, 1592.9021, 1601.1526, 1630.0334, 1662.0596,
            1682.6108, 1689.0599, 1676.5842, 1635.6819, 1561.4032, 1462.4318, 1349.1211, 1223.0033, 1091.6134,
            969.89044, 867.5361, 780.41626, 704.90314, 639.86633, 571.3017, 497.18524, 430.50333, 376.43744, 335.329,
            301.84308, 266.11346, 216.1726, 145.90388, 69.40163, 6.2822857, -44.526463, -94.935646, -145.99371,
            -203.92838, -276.31796, -350.2114, -413.33847, -467.1132, -518.0094, -567.5848, -615.0106, -659.67163,
            -685.6454, -683.87933, -660.50757, -622.7549, -592.06635, -574.89197, -560.0859, -550.97235, -547.23315,
            -550.5877, -565.651, -587.4596, -621.71246, -670.6035, -727.52783, -795.5536, -871.09656, -942.68823,
            -1009.519, -1076.462, -1145.0883, -1220.204, -1299.5819, -1366.1815, -1413.3549, -1445.462, -1465.7451,
            -1478.4558, -1484.0813, -1474.9672, -1438.5527, -1378.9619, -1312.5193, -1249.6793, -1199.3994, -1159.8258,
            -1123.6235, -1077.1719, -1010.4406, -935.00116, -851.9267, -762.7784, -672.0623, -572.3121,
        ];
        assert_eq!(freq_samples, freq_samples_expected);

        // Activated - transition case 2
        let info = LongTermPostFilterInfo::new(true, true, 134);
        #[rustfmt::skip]
        let mut freq_samples = [
            -462.93643, -347.34723, -233.52136, -146.90399, -88.30592, -54.41974, -38.89972, -29.513828, -32.403988,
            -44.259785, -50.78795, -50.632652, -45.593044, -27.971249, 0.8717003, 30.826332, 69.452446, 121.2875,
            171.58945, 206.59914, 222.87296, 227.7788, 231.79756, 239.57152, 252.44443, 266.01028, 267.37997, 257.9225,
            243.43958, 217.44739, 187.97891, 154.56412, 106.44893, 49.40045, -12.6106415, -59.774506, -78.434135,
            -84.324326, -86.347145, -99.99074, -132.43604, -169.84525, -211.62694, -260.47745, -316.94797, -377.97058,
            -428.13593, -455.18884, -457.09598, -437.08044, -401.62006, -356.88644, -305.0173, -250.73701, -192.85962,
            -129.52043, -77.097824, -37.619987, -8.04094, 3.1030326, 11.228271, 23.338984, 34.778336, 55.78636,
            78.24302, 97.53674, 125.637695, 162.30896, 208.29907, 263.90472, 319.387, 364.05295, 390.1394, 395.84088,
            384.5782, 356.90155, 309.81726, 244.27762, 159.42972, 55.85165, -64.28579, -203.06984, -347.07492,
            -480.43695, -595.8098, -683.94464, -752.0352, -806.8239, -843.51196, -860.25073, -851.5403, -823.6071,
            -790.0939, -756.19116, -730.0096, -714.0216, -702.7063, -695.4344, -695.1503, -706.0311, -733.2499,
            -778.845, -839.63135, -908.7647, -979.7435, -1053.0084, -1128.7417, -1200.269, -1265.076, -1325.569,
            -1382.6608, -1438.3086, -1496.3057, -1554.5283, -1607.3094, -1655.1637, -1699.0496, -1740.3623, -1778.7306,
            -1806.207, -1818.685, -1815.409, -1799.2051, -1778.6681, -1760.7465, -1749.7568, -1745.2687, -1742.762,
            -1743.374, -1754.2769, -1786.7277, -1851.4539, -1952.1932, -2083.9783, -2234.8418, -2393.0852, -2552.093,
            -2709.082, -2862.023, -3006.3904, -3134.876, -3240.8032, -3323.919, -3392.5146, -3459.497, -3535.9849,
            -3625.2485, -3721.101, -3812.6836, -3891.5942, -3956.4817, -4012.3245, -4064.998, -4115.882, -4160.6753,
            -4192.774, -4209.187, -4213.0947, -4211.656, -4210.6763, -4209.3535, -4201.593, -4180.723, -4143.664,
            -4094.022, -4038.93, -3984.3613, -3932.6794, -3881.8433, -3829.1377, -3775.2842, -3724.513, -3680.776,
            -3644.048, -3609.507, -3567.124, -3505.859, -3419.9043, -3308.9949, -3176.3008, -3028.0173, -2872.4634,
            -2717.9429, -2572.0747, -2441.2134, -2328.8481, -2233.8389, -2149.42, -2067.1467, -1979.7733, -1881.6626,
            -1771.871, -1653.4023, -1530.0349, -1405.8168, -1285.6444, -1174.4084, -1074.1273, -983.5908, -898.9976,
            -814.13837, -722.3748, -620.1046, -509.64612, -396.73444, -286.06772, -179.28769, -75.687485, 24.826172,
            120.5242, 207.15674, 279.2503, 334.79956, 378.0835, 418.437, 464.47144, 520.79297, 587.54376, 660.0079,
            734.82043, 813.5863, 899.42596, 993.3441, 1089.2538, 1176.0238, 1244.675, 1292.6222, 1326.2156, 1356.2456,
            1390.8427, 1429.6411, 1462.1608, 1477.0234, 1468.2399, 1436.7247, 1388.8541, 1328.8508, 1257.5665,
            1173.7887, 1076.6088, 971.8205, 869.8804, 780.17786, 706.3176, 642.9203, 579.6102, 507.52985, 424.23822,
            334.74408, 247.16443, 166.47836, 91.97533, 19.68251, -53.966515, -127.942345, -197.49265, -258.49423,
            -309.1388, -352.31018, -394.37518, -440.31805, -491.68707, -546.4675, -602.0992, -658.08984, -715.7874,
            -777.32214, -842.2093, -905.86127, -962.25726, -1006.56433, -1037.2737, -1056.3359, -1066.9244, -1071.4385,
            -1070.2313, -1061.7592, -1043.7142, -1013.4083, -968.1195, -905.3945, -823.6625, -723.8656, -610.2416,
            -489.15295, -366.59894, -245.75883, -126.30047, -5.8633795, 117.427734, 243.48848, 370.7084, 497.54443,
            623.64276, 748.8652, 870.81323, 983.1976, 1076.9547, 1144.1791, 1182.6096, 1197.0028, 1196.6348, 1191.3041,
            1186.1425, 1181.8829, 1178.4749, 1179.3162, 1192.3492, 1226.663, 1286.8209, 1369.1917, 1463.2593,
            1557.2914, 1644.4396, 1725.0815, 1804.0485, 1885.3147, 1968.4823, 2049.8135, 2126.6506, 2201.2378,
            2280.2183, 2369.598, 2468.7273, 2568.087, 2653.085, 2711.7532, 2741.2566, 2748.944, 2747.494, 2747.8315,
            2754.6597, 2767.1184, 2783.0059, 2802.5186, 2828.2537, 2861.6301, 2898.9814, 2931.0005, 2946.564,
            2938.5486, 2907.5115, 2860.7595, 2807.5742, 2754.1807, 2701.8474, 2649.0103, 2595.1577, 2543.1792,
            2498.2886, 2464.5063, 2441.502, 2424.2297, 2405.5247, 2379.72, 2344.799, 2302.0784, 2254.2764, 2203.8547,
            2152.7466, 2103.1711, 2058.185, 2021.087, 1993.9574, 1976.6077, 1966.915, 1962.3864, 1961.7065, 1965.112,
            1973.4125, 1986.6711, 2003.6866, 2022.5944, 2041.7399, 2059.7954, 2074.8965, 2083.8752, 2082.9111,
            2069.9214, 2047.2725, 2022.5962, 2006.419, 2007.5945, 2029.2957, 2068.228, 2117.4487, 2170.6548, 2224.6873,
            2278.584, 2330.298, 2374.2505, 2402.1917, 2407.234, 2388.0386, 2349.8228, 2301.0642, 2248.2512, 2192.6177,
            2131.478, 2063.0989, 1991.1888, 1925.0916, 1875.0167, 1845.4666, 1831.6798, 1821.8097, 1803.447, 1769.8939,
            1721.9398, 1664.3434, 1600.2563, 1528.1843, 1443.9144, 1345.7191, 1238.514, 1133.3607, 1042.3727, 972.6399,
            923.5854, 889.47504, 864.8752, 848.6354, 843.2933, 850.4648, 866.033, 879.1532, 876.24506, 847.372,
            790.68427, 712.07196, 620.6865, 523.73883, 424.08646, 321.60657, 216.44089, 111.057236, 9.466253,
            -85.489845, -174.24252, -260.047, -346.19894, -433.07126, -517.2522, -593.3824, -657.2282, -707.7102,
            -746.5956, -776.4009, -798.28723, -811.41473, -813.88995, -804.2301, -782.2126, -748.8439, -706.1048,
            -657.14606, -606.8167, -561.62634, -528.41144, -512.0252, -513.4654, -529.8556, -556.45624, -589.32227,
            -626.6715, -668.06024, -712.2979, -756.1645, -795.3605, -827.15515, -852.56323, -875.9639, -902.0606,
            -932.3437, -963.914, -991.80914, -1013.1952, -1030.0387, -1047.8396, -1071.0021,
        ];
        post_filter.run(&info, 320, &mut freq_samples);
        #[rustfmt::skip]
        let freq_samples_expected = [
            -462.93643, -342.8044, -225.69908, -137.13239, -77.861534, -44.447205, -30.411432, -23.473688, -29.779148,
            -45.987637, -57.805084, -63.914696, -66.11426, -56.713043, -37.096333, -17.240345, 10.670216, 51.40851,
            90.62616, 115.25731, 122.6413, 120.65816, 119.78547, 124.20146, 134.56567, 145.84068, 144.86555, 133.03192,
            116.11749, 87.65164, 55.68181, 19.674408, -30.88272, -89.77165, -152.69475, -199.97537, -218.74194,
            -225.34183, -228.5459, -243.02103, -274.6768, -308.87067, -344.98865, -385.9934, -432.7312, -482.37888,
            -519.9354, -533.8307, -522.57623, -489.41028, -440.2725, -380.4821, -311.44357, -237.52147, -157.69118,
            -70.87216, 5.9308624, 70.58849, 126.09006, 164.20158, 200.58534, 241.82042, 282.8008, 333.11957, 383.86053,
            429.9729, 482.59134, 540.1742, 602.375, 668.83856, 730.1382, 777.2069, 804.80945, 813.7645, 809.4365,
            793.4553, 763.44543, 720.71387, 664.5996, 595.85693, 516.75134, 425.40222, 335.06552, 260.26495, 206.27869,
            180.03265, 172.51312, 176.29742, 195.66736, 231.34265, 286.9939, 354.49384, 419.49792, 477.35107,
            521.30865, 550.8081, 572.9923, 589.67, 599.2553, 599.4723, 587.6166, 564.27856, 534.5227, 505.47125,
            482.104, 461.47156, 440.8396, 424.22546, 411.36694, 397.5072, 379.8656, 354.9696, 318.16357, 271.32532,
            220.07971, 164.35803, 104.45972, 40.84265, -24.004028, -80.443115, -123.228516, -164.20105, -190.22412,
            -207.96375, -222.3197, -236.19275, -248.93726, -257.2528, -264.14343, -277.88855, -309.19604, -366.61304,
            -451.1194, -555.8528, -668.7477, -779.6262, -883.88367, -980.24084, -1067.603, -1142.6014, -1200.271,
            -1237.6948, -1258.8262, -1274.9216, -1299.4609, -1341.9485, -1403.2813, -1475.8555, -1549.1941, -1616.4993,
            -1677.7981, -1738.0386, -1801.5625, -1867.7019, -1930.7747, -1983.8811, -2024.1235, -2053.9272, -2078.16,
            -2099.3508, -2114.008, -2115.4812, -2099.1294, -2065.6987, -2022.9072, -1980.9585, -1947.4465, -1925.2548,
            -1912.3618, -1905.8207, -1905.496, -1913.6082, -1931.0083, -1954.334, -1976.4008, -1986.7917, -1976.2627,
            -1942.0454, -1886.6776, -1814.86, -1732.7021, -1647.0762, -1564.1986, -1489.6866, -1428.3014, -1382.3462,
            -1349.9022, -1324.0315, -1296.9377, -1262.8103, -1218.027, -1163.693, -1104.116, -1043.2169, -984.0846,
            -929.83527, -883.18317, -844.1195, -809.98956, -776.2441, -736.59827, -684.9973, -618.9375, -541.9455,
            -460.6889, -380.49918, -303.594, -229.97295, -160.26324, -96.637054, -43.27495, -4.994812, 17.075043,
            27.565796, 35.078247, 46.562744, 64.8454, 89.00101, 114.440186, 138.86743, 164.97186, 196.35944, 234.17834,
            273.08752, 304.1447, 321.7611, 326.71326, 327.2782, 334.05713, 353.4353, 383.16394, 412.15417, 429.94153,
            431.89697, 419.52124, 398.48267, 371.58606, 338.70032, 298.66852, 251.4201, 203.28845, 163.87708,
            140.30115, 133.427, 135.94186, 137.02774, 128.40643, 108.02786, 80.05032, 50.353592, 21.24228, -9.695122,
            -46.324104, -90.50096, -138.8934, -184.96338, -223.71983, -253.09445, -275.6913, -297.0499, -320.91684,
            -347.8954, -375.97058, -403.62253, -431.83868, -463.04004, -499.51184, -540.2319, -580.21436, -613.897,
            -637.8767, -652.5154, -661.26636, -667.9452, -674.79584, -681.6578, -686.5095, -686.6601, -678.9796,
            -660.1492, -627.11743, -578.0657, -514.2095, -440.3472, -363.22107, -288.66968, -219.1813, -153.51508,
            -88.4573, -21.306412, 48.95056, 122.30011, 199.12387, 280.7963, 368.21283, 459.292, 547.86975, 625.406,
            685.01624, 725.33887, 750.9395, 769.2898, 787.094, 806.2039, 825.1108, 842.9783, 863.26904, 893.7539,
            942.41626, 1012.07666, 1097.8364, 1189.4242, 1276.9745, 1356.129, 1428.896, 1499.9583, 1571.6084,
            1641.3701, 1704.1997, 1757.2407, 1803.011, 1847.98, 1897.2083, 1949.0532, 1994.3097, 2020.9917, 2021.8184,
            1999.2637, 1964.7063, 1932.368, 1912.3353, 1907.2638, 1914.3707, 1930.2656, 1954.3856, 1988.2827,
            2031.7137, 2079.1948, 2120.3833, 2144.5295, 2146.0474, 2127.1277, 2095.6384, 2059.894, 2024.1611,
            1987.8718, 1948.615, 1906.0854, 1863.7983, 1827.266, 1800.2026, 1781.7311, 1766.643, 1748.325, 1722.1537,
            1687.0586, 1644.6992, 1597.4465, 1547.032, 1494.681, 1442.1528, 1392.2659, 1348.1832, 1311.9849, 1283.7686,
            1262.0864, 1245.3816, 1233.1785, 1226.1047, 1224.803, 1228.7861, 1236.185, 1244.4578, 1251.1532, 1253.9064,
            1249.7266, 1234.6941, 1205.1504, 1160.2441, 1104.1664, 1046.0695, 996.9687, 965.14734, 952.8568, 956.44495,
            969.5974, 987.2158, 1007.00464, 1027.8516, 1046.7155, 1057.1407, 1051.1571, 1023.6068, 975.59155,
            914.18335, 848.2141, 783.13464, 718.75806, 651.6742, 580.41907, 509.25867, 447.2954, 403.20728, 379.31494,
            369.45398, 362.33923, 348.1106, 323.35107, 291.13464, 256.45203, 220.88013, 180.7677, 130.36877, 67.349976,
            -3.7459717, -72.97217, -130.3761, -171.32361, -197.80609, -214.9992, -225.98627, -228.97266, -219.2597,
            -194.48395, -158.97552, -123.71088, -101.57648, -101.13751, -123.085266, -161.48029, -208.41772,
            -258.53772, -310.39194, -364.42804, -419.92776, -473.7387, -521.8341, -562.2293, -596.6528, -629.4323,
            -664.3385, -701.7526, -738.21686, -768.54834, -788.8178, -798.0349, -797.6033, -789.3593, -773.923,
            -750.51166, -718.0343, -676.3247, -626.57715, -570.9775, -512.2859, -453.93353, -400.37228, -356.80145,
            -327.7693, -315.20984, -317.3139, -329.3084, -345.86884, -363.56342, -381.603, -400.4791, -419.79935,
            -437.3371, -450.28278, -457.70862, -462.0325, -467.86328, -478.76544, -494.39392, -510.4903, -522.0974,
            -527.72766, -531.13446, -538.96216, -555.6775,
        ];
        assert_eq!(freq_samples, freq_samples_expected);

        // ActivePitchChanged - transition case 5
        let info = LongTermPostFilterInfo::new(true, true, 136);
        #[rustfmt::skip]
        let mut freq_samples = [
            -1097.484, -1120.5142, -1129.9044, -1118.6993, -1087.8778, -1045.0278, -999.07544, -954.1733, -908.27,
            -856.5168, -796.78906, -733.19165, -674.15485, -626.50793, -590.93884, -562.6337, -534.6552, -504.32108,
            -476.53494, -458.61398, -454.42786, -459.92474, -464.4726, -456.49527, -429.74686, -389.08292, -345.7242,
            -310.80746, -290.77448, -285.3322, -292.76804, -311.76852, -342.98694, -386.69556, -439.71942, -495.4435,
            -543.1727, -577.784, -596.7043, -597.1864, -586.8573, -574.1747, -563.0739, -553.3843, -542.7365,
            -535.00586, -532.5394, -538.82935, -558.5325, -589.7922, -633.70483, -684.65625, -728.9436, -762.1153,
            -781.498, -780.8221, -755.63794, -706.4991, -641.1553, -568.64215, -496.67224, -432.4506, -379.78644,
            -342.5705, -325.42334, -327.9347, -345.42715, -370.7724, -401.93597, -436.9108, -463.03165, -482.49646,
            -507.72577, -538.40765, -574.7809, -607.1339, -623.9975, -626.57666, -612.1267, -584.5003, -549.22473,
            -512.10754, -473.54614, -414.1963, -338.65976, -268.24878, -212.05249, -178.29366, -161.54543, -151.40002,
            -140.70525, -129.6048, -127.012726, -134.15372, -152.0729, -174.99931, -198.76984, -225.96996, -250.91785,
            -278.44342, -314.07388, -355.14948, -393.81763, -421.86774, -444.84335, -456.70917, -452.7793, -442.07758,
            -421.8343, -391.7492, -361.5553, -339.6891, -325.71848, -318.6924, -313.81464, -304.54086, -301.52365,
            -307.9194, -323.96115, -340.97635, -339.84338, -331.9051, -318.43185, -298.3857, -289.66547, -286.6652,
            -282.65363, -276.39142, -272.05396, -279.72662, -300.50778, -337.33478, -386.27704, -443.09082, -501.65875,
            -565.30756, -652.24225, -755.4369, -870.88776, -1004.73615, -1152.1647, -1311.7432, -1481.893, -1668.4835,
            -1865.6188, -2055.1348, -2236.7354, -2415.5513, -2593.3982, -2764.731, -2922.301, -3059.905, -3175.2615,
            -3276.195, -3357.4734, -3415.343, -3460.653, -3492.2385, -3509.5461, -3523.9946, -3541.6313, -3577.03,
            -3640.6956, -3706.9211, -3770.1372, -3844.1833, -3915.0588, -3985.0134, -4062.8076, -4143.2583, -4219.959,
            -4278.3555, -4317.2637, -4339.5234, -4352.1406, -4362.1265, -4353.675, -4324.1665, -4283.246, -4233.4053,
            -4178.083, -4120.3447, -4056.709, -3976.1204, -3880.6602, -3774.07, -3651.3037, -3512.5, -3360.0796,
            -3203.4336, -3050.2913, -2906.6733, -2775.1296, -2644.2637, -2511.438, -2380.8552, -2248.1306, -2114.8684,
            -1995.1123, -1889.1849, -1783.9177, -1675.2041, -1559.3057, -1447.0999, -1352.8055, -1266.0538, -1179.5769,
            -1094.2219, -1011.1386, -927.7953, -843.243, -768.36755, -704.34595, -640.7532, -562.87573, -464.08563,
            -353.85126, -245.51999, -154.34895, -81.14457, -18.706238, 37.60077, 96.27663, 153.91443, 208.6984,
            269.28082, 338.66077, 412.2937, 479.95587, 536.7573, 582.5949, 613.7015, 633.15955, 651.55115, 667.9326,
            684.05835, 714.488, 753.85376, 788.3535, 813.1959, 817.9694, 792.6777, 735.78784, 657.76776, 577.14325,
            514.34155, 477.9563, 460.92963, 456.96912, 457.64975, 464.2096, 480.30005, 495.89398, 506.7548, 510.06015,
            506.0656, 500.24344, 501.76288, 514.2082, 523.3666, 531.32635, 532.68164, 512.2106, 486.38913, 454.96124,
            413.2605, 376.88562, 341.31628, 303.66742, 270.0655, 236.19926, 199.54686, 157.08835, 100.68674, 37.647774,
            -16.571394, -63.826523, -104.92358, -141.87694, -183.7624, -224.28102, -258.9445, -286.85373, -306.77243,
            -324.32092, -335.12225, -339.06268, -332.77267, -299.2921, -244.98857, -183.2632, -121.90414, -76.534065,
            -44.899715, -11.62276, 34.343388, 98.26323, 173.2465, 251.30093, 325.76276, 393.36942, 457.55164, 518.0535,
            571.5968, 615.66016, 655.04065, 695.7536, 737.0899, 786.66187, 851.136, 927.5252, 1013.6009, 1109.6206,
            1216.7493, 1334.0854, 1461.132, 1593.296, 1728.0421, 1875.9589, 2042.3734, 2216.6416, 2385.506, 2547.4172,
            2701.4587, 2844.1956, 2980.924, 3109.17, 3220.606, 3304.2102, 3350.46, 3368.6006, 3362.7087, 3332.358,
            3280.709, 3206.9717, 3119.0652, 3016.0803, 2903.9512, 2804.989, 2715.8154, 2629.551, 2547.7957, 2465.2039,
            2383.047, 2304.386, 2237.9382, 2188.3303, 2148.0142, 2115.626, 2082.2136, 2040.894, 1996.1311, 1947.9188,
            1900.9966, 1856.6047, 1819.7887, 1800.6093, 1798.6365, 1819.4702, 1854.3895, 1890.377, 1938.3956,
            1998.0531, 2062.0679, 2131.367, 2202.548, 2266.4258, 2320.0603, 2378.4614, 2446.074, 2512.4368, 2570.949,
            2612.8655, 2634.8767, 2643.635, 2656.2944, 2687.3433, 2725.503, 2749.8433, 2755.7898, 2749.5947, 2732.7625,
            2707.1287, 2673.6453, 2629.827, 2575.8474, 2514.1912, 2458.0786, 2417.8098, 2395.1814, 2392.6777,
            2402.0884, 2419.6436, 2446.2021, 2475.853, 2505.481, 2523.983, 2519.4207, 2496.9692, 2463.86, 2416.0825,
            2349.9207, 2263.9792, 2158.8147, 2044.7573, 1929.4032, 1819.4272, 1713.971, 1603.0427, 1492.3943,
            1382.7449, 1272.9819, 1175.5515, 1100.0779, 1049.4731, 1008.33795, 973.14453, 951.96136, 930.86096,
            910.54694, 894.43256, 872.1588, 846.2406, 817.3495, 792.44653, 784.7976, 791.3051, 809.1773, 833.4102,
            847.3397, 843.7097, 823.59564, 785.32666, 733.75964, 677.8262, 619.4476, 563.20123, 519.02075, 485.8358,
            459.13956, 435.27945, 408.31216, 381.87216, 351.91, 318.51614, 292.8391, 263.04898, 228.11197, 198.44838,
            173.12642, 158.98888, 147.08707, 124.9829, 97.43573, 74.75817, 66.39967, 69.28095, 88.37106, 119.103226,
            142.14178, 158.66534, 166.50217, 166.50797, 168.38155, 166.83104, 159.28036, 148.35973, 128.81505,
            95.83782, 59.783768, 29.240232, -2.2875664, -40.220715, -89.975914, -146.32771, -192.82083, -224.33153,
            -245.72568, -261.48184, -274.1277,
        ];
        post_filter.run(&info, 320, &mut freq_samples);
        #[rustfmt::skip]
        let freq_samples_expected = [
            -578.7846, -602.8267, -617.02893, -615.94867, -601.57404, -580.8959, -560.53204, -541.7056, -520.3715,
            -491.5397, -454.45645, -414.79205, -381.44672, -360.33694, -350.69574, -346.863, -342.23813, -335.2017,
            -331.21255, -336.7983, -354.02927, -377.28918, -395.8925, -400.07867, -386.46207, -362.1463, -338.67545,
            -325.58243, -326.81244, -340.01715, -362.5156, -392.8398, -431.6576, -478.998, -531.39984, -582.48126,
            -622.83813, -649.41644, -661.67566, -658.5397, -648.21826, -638.11456, -630.16626, -622.25916, -610.7409,
            -598.56384, -587.1732, -579.27704, -578.8295, -583.8025, -595.76886, -610.1743, -615.5237, -610.0842,
            -593.2602, -560.3085, -508.05508, -437.84143, -357.21445, -273.92322, -193.86328, -122.47586, -62.325928,
            -16.656586, 10.408844, 19.743057, 16.142166, 6.324997, -8.202591, -25.637695, -33.939682, -35.98027,
            -43.19937, -53.254272, -64.69644, -67.46237, -51.82962, -21.642944, 23.425598, 77.80307, 135.48596,
            191.28482, 245.75452, 317.5787, 400.47906, 473.59546, 530.47327, 566.1444, 588.31085, 607.2344, 627.8426,
            647.281, 655.25903, 650.79846, 634.073, 612.30457, 590.7401, 567.90735, 550.2997, 533.5903, 513.0824,
            491.8979, 476.9899, 474.1204, 475.06793, 483.41852, 501.23447, 518.05396, 536.5569, 557.4514, 572.4798,
            575.8922, 570.777, 559.844, 548.62244, 543.3136, 533.2593, 516.2762, 492.98047, 471.68073, 472.16302,
            470.81732, 470.87296, 472.95462, 460.55615, 441.60873, 423.9538, 409.0664, 393.17273, 367.58978, 333.1617,
            288.6438, 238.87097, 187.8349, 140.4729, 92.26648, 25.612122, -51.183716, -134.0965, -229.63916,
            -333.71478, -446.28845, -567.3235, -703.5835, -849.58136, -988.52234, -1121.6119, -1253.7953, -1385.2201,
            -1508.6704, -1616.4807, -1703.5469, -1769.6517, -1824.2926, -1863.6085, -1885.7894, -1903.0951, -1915.1052,
            -1922.1041, -1935.3611, -1959.2527, -2005.1978, -2079.2554, -2153.3113, -2222.4355, -2300.6768, -2373.8774,
            -2444.3564, -2519.662, -2593.1902, -2658.5698, -2703.4485, -2730.1577, -2744.5295, -2755.0386, -2768.4297,
            -2768.9604, -2755.3115, -2737.7732, -2718.3252, -2699.144, -2681.3123, -2659.3389, -2621.3496, -2569.4004,
            -2506.8955, -2428.6777, -2335.2021, -2229.06, -2119.169, -2012.1779, -1913.124, -1824.2719, -1735.6282,
            -1647.2045, -1565.0806, -1485.6226, -1410.2944, -1351.2362, -1305.8739, -1259.583, -1208.4817, -1149.6702,
            -1094.1477, -1054.3917, -1018.39813, -979.12494, -938.41125, -898.0473, -855.7174, -810.377, -771.79895,
            -739.35803, -701.9739, -646.4205, -568.89734, -480.79688, -395.04608, -324.31604, -266.45428, -213.11081,
            -160.60573, -102.98784, -46.497314, 5.144806, 58.927917, 116.17195, 171.64311, 216.21631, 247.44662,
            267.756, 275.52667, 275.28833, 277.5622, 280.52518, 285.1563, 304.34772, 330.81244, 350.65405, 360.52444,
            352.34436, 319.2709, 262.6707, 194.02353, 129.9755, 86.45105, 66.810394, 60.809937, 61.98828, 63.937836,
            70.36554, 86.17218, 102.02371, 114.625, 122.09442, 125.41373, 129.97742, 143.49213, 167.10513, 185.23517,
            199.8941, 206.16565, 190.82999, 171.87119, 148.82068, 116.820984, 90.583405, 64.47809, 35.915405,
            11.789124, -11.418167, -35.35898, -62.65448, -101.27803, -144.49683, -179.30328, -210.48466, -240.55984,
            -271.9748, -312.65936, -354.60715, -392.29565, -424.0747, -447.9613, -468.5365, -480.63953, -484.25143,
            -476.8697, -444.00726, -394.8321, -343.40277, -296.2135, -266.1862, -248.35864, -227.04222, -193.33144,
            -144.78932, -89.77028, -35.357117, 14.495911, 59.85669, 106.72006, 156.15884, 205.49231, 252.31876,
            300.59293, 354.1444, 409.51294, 471.30478, 542.83514, 618.5273, 695.04333, 772.6138, 852.98254, 936.2776,
            1023.30457, 1111.0156, 1198.5332, 1297.019, 1410.9958, 1529.6226, 1641.5543, 1748.2587, 1851.4727, 1949.71,
            2048.9043, 2146.1484, 2233.0713, 2299.776, 2338.8418, 2360.69, 2368.548, 2360.1758, 2336.7017, 2295.735,
            2244.002, 2179.7813, 2108.42, 2050.0088, 1998.1708, 1944.6399, 1890.9342, 1832.5669, 1772.2147, 1713.8794,
            1666.1041, 1632.3969, 1604.6522, 1582.0222, 1557.0364, 1525.1805, 1492.7473, 1460.3547, 1432.294, 1408.536,
            1392.3602, 1391.2827, 1402.2925, 1428.9365, 1461.6272, 1488.6859, 1522.6187, 1563.515, 1604.8237,
            1648.4567, 1692.0441, 1728.1387, 1755.8689, 1790.5916, 1835.0598, 1877.4358, 1911.1895, 1928.9221,
            1929.325, 1920.1799, 1917.3938, 1931.9063, 1949.4927, 1949.6743, 1930.669, 1901.0165, 1863.0205, 1818.1674,
            1766.3877, 1704.3352, 1631.7904, 1550.9219, 1473.7632, 1408.2548, 1353.8857, 1311.7145, 1273.6556,
            1237.6255, 1206.5042, 1176.4982, 1146.8325, 1109.2416, 1055.5414, 994.10583, 933.28845, 868.96716,
            797.1981, 716.2544, 626.0209, 534.98145, 447.38892, 365.8723, 286.05774, 196.51062, 103.059326, 6.8968506,
            -92.14331, -181.349, -252.27832, -303.61584, -350.6811, -394.96558, -426.59076, -457.68054, -485.62177,
            -506.3177, -529.67224, -552.6345, -574.556, -589.23706, -585.51697, -568.8382, -543.1149, -513.0699,
            -493.04504, -486.43652, -488.7608, -499.53772, -513.2479, -522.2251, -527.0096, -525.81647, -511.97375,
            -489.53488, -464.52963, -440.7222, -423.17795, -407.4401, -397.17227, -391.9896, -381.28436, -377.06674,
            -379.32077, -377.2166, -372.10382, -358.04572, -344.60416, -343.08694, -347.0165, -345.67892, -330.8783,
            -307.29242, -271.18192, -227.56721, -193.81393, -165.4403, -142.04245, -121.42282, -94.62474, -68.81453,
            -47.96602, -30.38617, -21.44278, -24.866554, -29.532852, -27.572943, -26.39436, -31.489376, -47.73922,
            -69.73716, -82.70992, -84.562485, -82.343704, -80.851, -81.43208,
        ];
        assert_eq!(freq_samples, freq_samples_expected);

        // ActiveNoPitchChange - transition case 4
        let info = LongTermPostFilterInfo::new(true, true, 136);
        #[rustfmt::skip]
        let mut freq_samples = [
            -292.14764, -313.06204, -337.59555, -384.65967, -454.30124, -529.93445, -597.26324, -643.06647, -669.54407,
            -682.19104, -679.5723, -665.3846, -643.1253, -611.8372, -568.3924, -516.38776, -470.50418, -434.22855,
            -395.94836, -349.4547, -295.5383, -239.54758, -186.52908, -145.55113, -130.745, -135.22012, -143.60983,
            -151.56058, -158.26048, -168.28256, -180.75365, -196.4275, -217.52869, -230.32733, -236.46542, -254.40222,
            -280.9839, -307.70758, -340.7851, -381.06305, -426.71344, -480.77704, -536.89465, -588.4791, -637.96234,
            -687.80164, -732.01044, -767.25, -803.4193, -839.1151, -871.4927, -904.9992, -933.52966, -966.9023,
            -1016.08484, -1066.6865, -1106.3118, -1130.532, -1147.195, -1172.9998, -1211.7219, -1256.476, -1297.8923,
            -1330.8607, -1358.1207, -1384.5087, -1411.6349, -1438.5046, -1460.81, -1468.0884, -1455.9624, -1427.3873,
            -1383.8306, -1334.095, -1289.6709, -1252.4845, -1214.0847, -1165.6658, -1112.8619, -1065.6158, -1022.6605,
            -970.6576, -904.0072, -833.1134, -760.58704, -688.4604, -607.5392, -497.93073, -375.81757, -254.83571,
            -124.67853, 17.89251, 175.42064, 335.35272, 482.82852, 611.81476, 711.5647, 783.9825, 833.0896, 861.8701,
            881.8312, 901.50946, 926.50604, 950.7421, 955.8478, 940.40265, 922.5005, 906.11646, 887.50977, 870.92334,
            854.0742, 836.95776, 821.3246, 803.4901, 776.7728, 735.6349, 686.123, 631.7012, 574.8167, 512.6657,
            436.3199, 361.8034, 301.8484, 246.77817, 185.7377, 114.13265, 37.760223, -49.17659, -154.00397, -272.4621,
            -406.02805, -545.8312, -675.88904, -794.3148, -909.11316, -1020.9018, -1120.031, -1208.5936, -1293.2699,
            -1378.9922, -1477.4784, -1586.2919, -1694.6757, -1801.7699, -1906.5596, -2004.1901, -2082.3555, -2130.8042,
            -2154.3604, -2168.0728, -2193.9995, -2236.071, -2281.538, -2337.7488, -2415.1038, -2503.5676, -2586.5488,
            -2657.291, -2721.135, -2785.1519, -2850.635, -2912.1958, -2971.892, -3032.566, -3093.7705, -3168.2212,
            -3250.2708, -3322.71, -3392.4133, -3466.4546, -3544.3318, -3617.2415, -3676.7844, -3732.1514, -3790.7356,
            -3853.258, -3909.9324, -3947.5437, -3974.3828, -4000.1714, -4026.541, -4056.205, -4081.205, -4095.7524,
            -4106.922, -4117.9233, -4124.0093, -4117.1597, -4099.751, -4083.7312, -4066.8035, -4038.9136, -3991.6768,
            -3920.4526, -3837.0417, -3750.9197, -3657.2786, -3556.7295, -3455.4924, -3355.5647, -3257.8708, -3160.043,
            -3054.3032, -2944.7534, -2846.7507, -2762.4165, -2686.7495, -2617.7605, -2541.8833, -2453.7087, -2364.959,
            -2266.5828, -2150.8748, -2030.189, -1909.4111, -1798.5577, -1698.4766, -1597.2886, -1501.2515, -1413.9253,
            -1331.0004, -1249.1351, -1157.4594, -1047.7784, -931.2974, -825.41754, -728.301, -637.39874, -553.16406,
            -475.85736, -411.34723, -342.76184, -253.04881, -146.71509, -30.985062, 79.19682, 173.76141, 255.7579,
            327.16193, 377.9758, 398.0102, 393.49536, 372.84216, 340.87152, 301.04904, 259.06433, 224.21866, 205.15297,
            203.77704, 205.90222, 202.34561, 191.93433, 176.30232, 160.93704, 137.81, 107.36654, 80.29572, 64.89971,
            68.057556, 84.20175, 113.90836, 155.33731, 203.67337, 263.69446, 326.58612, 381.22058, 426.36322,
            468.20096, 512.0286, 556.3656, 601.09985, 644.6057, 689.43286, 731.6796, 764.62836, 788.6405, 804.66394,
            825.55115, 856.23883, 889.89655, 916.0222, 929.4518, 942.72076, 952.78217, 958.1389, 959.61597, 950.0264,
            945.3514, 943.0897, 928.24457, 899.22504, 849.23956, 788.9234, 729.2224, 665.4199, 596.42255, 525.08185,
            459.7273, 408.6278, 369.40952, 336.5017, 311.96094, 293.19354, 281.44644, 278.14868, 269.89484, 262.67172,
            265.0364, 276.25534, 303.16663, 333.66382, 359.62125, 388.33643, 414.52106, 430.38913, 436.38632, 439.0949,
            447.45093, 463.13724, 476.44052, 490.4304, 514.361, 547.25824, 599.9028, 672.3621, 751.2139, 830.3177,
            904.5592, 975.7102, 1035.4247, 1074.4988, 1097.2754, 1116.7842, 1152.1069, 1191.8561, 1219.7965, 1238.8195,
            1244.3499, 1239.4135, 1226.5548, 1200.6041, 1162.3062, 1121.7297, 1091.5028, 1066.2352, 1047.4064,
            1042.9103, 1043.886, 1052.7422, 1072.3534, 1096.1565, 1124.6835, 1157.2009, 1202.1378, 1267.6134,
            1345.8907, 1433.5735, 1529.3275, 1633.1494, 1742.3405, 1843.2507, 1933.3466, 2014.3296, 2090.038,
            2170.2493, 2246.205, 2309.7522, 2366.614, 2426.044, 2496.765, 2574.198, 2657.0154, 2747.2063, 2836.2004,
            2917.5996, 2987.2751, 3041.687, 3079.5266, 3106.2292, 3127.861, 3141.59, 3145.4463, 3140.8335, 3136.3667,
            3135.2092, 3130.5261, 3127.8374, 3130.3467, 3133.5457, 3136.3472, 3132.7493, 3116.496, 3089.2544,
            3057.6692, 3020.467, 2971.8494, 2916.3596, 2861.7473, 2811.163, 2764.547, 2725.3423, 2696.004, 2669.8137,
            2641.41, 2606.5422, 2555.9304, 2484.1616, 2400.7578, 2320.2642, 2243.1072, 2167.299, 2095.1746, 2023.0106,
            1952.463, 1890.1173, 1833.0878, 1780.1436, 1738.1539, 1709.1063, 1686.6945, 1664.0538, 1639.6707,
            1611.3213, 1579.3654, 1553.6488, 1531.9762, 1505.9307, 1484.571, 1476.1371, 1479.0269, 1492.7662,
            1515.2437, 1540.8505, 1569.5668, 1601.7802, 1626.5353, 1638.3237, 1639.6497, 1633.1738, 1623.5092,
            1604.437, 1574.2678, 1540.8848, 1503.5153, 1463.5356, 1421.3494, 1378.4624, 1339.934, 1302.8973, 1266.2386,
            1221.7916, 1161.3969, 1093.6812, 1024.0099, 957.3819, 899.47156, 847.46967, 802.19543, 763.4946, 721.8335,
            669.2194, 607.38184, 540.189, 473.9681, 417.23834, 366.26886, 315.81607, 266.82968, 217.49538, 166.25597,
            114.25025, 66.24334, 21.667877, -21.782158, -63.33486, -110.37565, -163.37317, -216.44617, -268.40286,
            -320.65067,
        ];
        post_filter.run(&info, 320, &mut freq_samples);
        #[rustfmt::skip]
        let freq_samples_expected = [
            -90.18535, -102.11511, -116.384415, -150.04745, -201.31703, -254.3656, -298.61935, -326.04388, -343.03348,
            -356.2732, -363.0179, -364.2756, -360.31226, -347.62915, -322.0965, -287.36044, -257.5868, -235.04446,
            -208.0369, -171.85963, -129.05524, -85.765305, -46.52327, -18.66204, -13.400055, -21.600044, -28.541557,
            -32.607903, -35.84649, -44.486443, -57.957993, -76.51495, -101.32961, -118.45515, -130.19728, -154.23624,
            -185.80305, -216.10965, -251.65219, -293.21997, -338.95065, -391.59515, -444.6314, -492.22894, -537.7965,
            -584.1852, -625.7372, -659.70264, -695.6417, -730.9453, -761.8655, -791.9146, -814.2675, -837.88666,
            -871.7608, -900.57324, -913.87555, -910.8328, -902.242, -905.05554, -920.9866, -941.0325, -955.1504,
            -959.03345, -956.6571, -953.562, -951.49365, -949.4522, -943.40173, -924.00574, -888.64233, -841.4618,
            -784.13104, -724.48956, -671.66705, -624.8969, -574.4705, -512.3486, -445.41357, -383.6078, -324.89044,
            -256.13812, -173.14526, -86.97388, 0.38220215, 87.851746, 184.71631, 308.49005, 440.4145, 566.8278,
            698.5299, 838.24365, 987.3683, 1132.646, 1260.087, 1365.4812, 1440.3582, 1488.6586, 1515.2927, 1523.2532,
            1523.0973, 1521.5813, 1522.5657, 1519.5029, 1496.1681, 1454.6836, 1414.692, 1379.2759, 1343.4895,
            1310.6631, 1278.1453, 1246.3674, 1217.6362, 1188.754, 1153.7822, 1108.1658, 1058.2782, 1006.92554,
            955.64087, 901.2633, 835.9227, 776.50903, 734.704, 699.73804, 661.26117, 616.0951, 570.82947, 519.858,
            456.45215, 385.24927, 304.6346, 222.65045, 152.84503, 94.07538, 36.762817, -19.871033, -66.75, -107.17395,
            -148.46472, -195.22253, -257.4032, -330.16016, -401.7074, -471.63538, -540.0574, -603.8784, -653.49133,
            -682.07654, -696.99976, -713.24146, -749.6035, -805.3264, -864.6279, -933.1284, -1018.8192, -1109.7175,
            -1189.6329, -1254.0446, -1310.4315, -1366.7698, -1424.4585, -1478.575, -1532.0669, -1588.3899, -1647.4369,
            -1721.238, -1803.0068, -1876.1974, -1949.0186, -2028.5062, -2113.2764, -2194.1392, -2263.4578, -2331.0708,
            -2403.8428, -2481.4595, -2553.7856, -2608.6511, -2655.1128, -2701.714, -2747.9314, -2794.5317, -2832.8997,
            -2858.334, -2879.1675, -2899.0059, -2913.4082, -2915.369, -2908.316, -2903.7556, -2897.9248, -2880.1401,
            -2842.893, -2783.5493, -2715.4912, -2648.3572, -2577.3672, -2503.5928, -2433.1165, -2366.9211, -2304.4573,
            -2241.9082, -2170.9004, -2095.441, -2029.7302, -1974.0889, -1922.7856, -1874.3585, -1817.1624, -1748.7024,
            -1682.2876, -1609.3551, -1523.2148, -1436.3911, -1352.4536, -1279.2563, -1215.3752, -1148.2948, -1084.6399,
            -1027.7524, -972.91785, -916.601, -848.4894, -762.1411, -670.16846, -589.323, -516.3158, -447.9541,
            -384.57605, -326.46048, -279.15924, -226.37984, -153.98007, -69.67814, 17.94011, 95.02913, 154.5694,
            202.49727, 242.23863, 264.93225, 262.43802, 242.90184, 215.21776, 183.45694, 149.62909, 117.41945,
            93.403275, 82.84405, 84.45201, 82.84101, 70.250626, 48.123154, 20.627838, -5.073929, -35.630035, -69.22451,
            -95.11753, -106.5427, -98.949875, -79.63895, -48.745796, -8.369812, 36.89676, 91.87604, 147.94986,
            195.44325, 234.98277, 273.40564, 314.95578, 356.36157, 395.9458, 431.25885, 464.73682, 493.03503, 510.8879,
            520.2296, 522.93567, 531.3667, 548.6869, 566.8611, 575.89813, 572.3101, 569.348, 563.51306, 552.93805,
            538.1791, 512.4512, 491.71054, 472.70303, 441.18793, 397.58643, 337.37653, 272.51312, 213.13104, 153.09851,
            90.60928, 28.080292, -27.09842, -68.45413, -100.05014, -127.86746, -149.38477, -166.44824, -177.1708,
            -180.18402, -188.65515, -196.2493, -195.21246, -187.31564, -166.57181, -144.99927, -129.04178, -109.969574,
            -92.24628, -82.21762, -77.69809, -71.181335, -54.276855, -26.889343, 0.15188599, 29.357727, 68.77301,
            116.03836, 180.42352, 260.40427, 342.58792, 422.57196, 497.53217, 571.1029, 636.53046, 686.77, 728.0172,
            773.3823, 839.5879, 912.7599, 976.75073, 1035.6665, 1085.8202, 1130.358, 1170.5814, 1199.529, 1216.1201,
            1227.8812, 1243.9321, 1256.2347, 1265.562, 1279.8341, 1291.1082, 1303.2737, 1319.9148, 1334.97, 1349.6323,
            1363.7977, 1386.1079, 1424.0273, 1469.5052, 1519.8295, 1574.5593, 1634.256, 1696.4797, 1748.5374,
            1789.5007, 1822.196, 1850.7382, 1884.071, 1912.8168, 1929.856, 1942.3267, 1959.7051, 1989.4749, 2025.5002,
            2065.4045, 2110.1953, 2150.8743, 2181.691, 2199.8643, 2203.6353, 2193.6235, 2176.6287, 2159.0322,
            2137.9927, 2111.785, 2081.8916, 2056.0127, 2035.3936, 2011.6448, 1989.1458, 1969.8517, 1948.7745,
            1925.4426, 1895.487, 1855.2737, 1808.9897, 1764.2952, 1719.5046, 1668.029, 1613.0756, 1559.9672, 1509.0657,
            1458.3127, 1410.2285, 1367.1869, 1323.5615, 1276.167, 1223.1677, 1157.7273, 1076.7128, 990.3993, 911.5475,
            837.6959, 764.7047, 693.5857, 620.155, 546.24426, 478.27502, 413.1421, 349.89722, 295.4159, 251.14746,
            210.68652, 168.09814, 123.31738, 75.569336, 26.388916, -14.35022, -49.572388, -87.65857, -119.50879,
            -137.94043, -145.72876, -143.89429, -134.5393, -122.59546, -107.17407, -87.3385, -73.01135, -67.713135,
            -67.07056, -67.36694, -63.90564, -63.11963, -66.769165, -67.88049, -68.841675, -69.79321, -71.52429,
            -73.28711, -70.74194, -67.29993, -64.22327, -69.228516, -89.0979, -114.18359, -139.26453, -160.03082,
            -171.75061, -177.80731, -177.45435, -170.96588, -167.6059, -174.33673, -188.57501, -206.53015, -222.8635,
            -230.82147, -235.49048, -241.91226, -248.012, -254.3131, -261.21143, -267.0642, -267.74112, -265.18414,
            -262.90778, -260.9077, -266.30362, -278.23315, -289.6394, -298.7391, -306.67255,
        ];
        assert_eq!(freq_samples, freq_samples_expected);

        // Deactivated - transition case 3
        let info = LongTermPostFilterInfo::new(false, true, 132);
        #[rustfmt::skip]
        let mut freq_samples = [
            -380.83432, -446.88705, -515.7319, -589.6471, -658.37146, -715.82825, -759.0343, -788.54486, -811.9708,
            -825.85077, -832.7059, -838.9507, -836.67365, -824.868, -810.6019, -794.5058, -782.70483, -786.7115,
            -803.96924, -826.9956, -854.93726, -887.13293, -919.4763, -944.72675, -964.3407, -980.9633, -996.7174,
            -1018.3943, -1038.7148, -1055.6372, -1073.6458, -1088.1885, -1101.3804, -1111.4736, -1120.524, -1131.307,
            -1137.389, -1144.214, -1153.1848, -1167.0598, -1191.4224, -1218.054, -1246.5493, -1275.5333, -1299.6941,
            -1316.5751, -1323.2358, -1327.5497, -1332.3658, -1337.1395, -1343.1924, -1346.9203, -1355.1132, -1370.7523,
            -1392.7631, -1425.0702, -1461.239, -1498.8083, -1534.6693, -1559.9911, -1576.4115, -1578.6277, -1567.4585,
            -1555.9686, -1541.092, -1525.0547, -1513.6597, -1499.1572, -1479.721, -1454.0294, -1418.6234, -1374.4613,
            -1328.3247, -1286.1417, -1240.2705, -1187.2441, -1127.2596, -1058.658, -985.6399, -905.6291, -815.9364,
            -715.5633, -604.5736, -489.9937, -375.7823, -265.61514, -160.54572, -60.766205, 34.318848, 126.982056,
            208.97278, 281.55884, 353.39386, 420.27405, 478.4252, 526.9842, 564.8197, 591.60504, 609.4817, 621.0288,
            626.9463, 630.4244, 632.74506, 630.9957, 620.9778, 607.46326, 596.08405, 582.7889, 570.37695, 555.0814,
            528.53284, 495.39026, 454.5733, 406.04358, 354.56967, 297.1142, 232.25401, 163.21338, 93.16636, 23.001862,
            -47.036148, -115.21831, -181.18599, -248.4486, -317.69327, -384.86273, -452.0945, -521.3707, -591.94867,
            -666.6193, -744.74194, -827.2992, -917.48413, -1010.70575, -1104.5869, -1201.524, -1294.2238, -1379.6155,
            -1460.6113, -1528.9102, -1585.4523, -1631.1124, -1660.9009, -1684.4255, -1699.1821, -1701.4526, -1700.5947,
            -1697.61, -1696.6571, -1702.1078, -1714.439, -1733.531, -1754.5884, -1775.7247, -1799.2393, -1827.3556,
            -1856.2937, -1885.4236, -1918.1951, -1954.9684, -1996.948, -2042.28, -2093.781, -2146.6353, -2189.093,
            -2223.9863, -2251.689, -2273.4807, -2294.2654, -2315.8933, -2340.1543, -2367.7327, -2405.2227, -2447.4607,
            -2489.6123, -2535.0264, -2575.6077, -2611.8645, -2641.9124, -2666.7974, -2697.137, -2726.5056, -2755.8135,
            -2787.091, -2813.767, -2836.5327, -2852.9668, -2865.6873, -2873.5032, -2868.0408, -2850.715, -2822.4873,
            -2783.2732, -2736.566, -2685.327, -2627.6494, -2560.0261, -2486.4858, -2411.283, -2334.7515, -2260.1433,
            -2189.3125, -2119.3108, -2052.565, -1992.5256, -1937.0294, -1888.7611, -1848.9229, -1812.4441, -1772.8254,
            -1722.6503, -1662.5598, -1594.4043, -1515.1047, -1426.463, -1329.6918, -1224.3643, -1116.0164, -1009.5359,
            -905.4838, -802.9583, -701.79346, -605.7693, -516.5092, -436.81702, -375.58536, -334.2002, -310.5628,
            -303.5844, -308.35977, -323.23022, -346.89465, -375.84933, -406.03006, -432.0379, -450.78473, -462.28992,
            -467.39557, -466.08475, -462.6616, -459.1948, -449.0974, -431.15015, -411.08737, -392.9422, -377.49863,
            -368.67184, -370.1563, -378.75293, -392.79608, -410.79825, -435.5704, -468.88922, -502.39908, -535.69073,
            -569.6185, -602.27716, -633.3312, -658.2254, -678.0781, -689.1164, -687.72736, -678.77576, -657.07355,
            -624.56256, -587.8258, -539.33655, -477.26443, -404.74258, -320.6686, -227.36028, -126.90553, -18.687975,
            97.70469, 221.64476, 350.7937, 485.1836, 624.7636, 768.85706, 918.2838, 1067.7938, 1215.2661, 1362.7802,
            1506.2122, 1643.0493, 1769.5411, 1888.3374, 2007.5011, 2116.3704, 2204.1897, 2276.0493, 2334.2388,
            2375.0833, 2400.9397, 2416.7896, 2419.0356, 2409.31, 2391.3875, 2358.868, 2315.0203, 2262.3967, 2193.9539,
            2113.851, 2024.538, 1927.651, 1832.4061, 1739.7405, 1653.9604, 1578.7108, 1505.152, 1431.2009, 1357.3208,
            1283.5349, 1210.578, 1134.9974, 1056.7867, 974.1136, 891.709, 815.5397, 739.52954, 666.8109, 599.267,
            531.3117, 464.1205, 399.3873, 337.9558, 276.83328, 217.15677, 159.31384, 99.04352, 44.807182, -1.9447366,
            -42.968796, -72.37995, -97.51363, -120.01941, -135.05333, -145.92494, -151.81703, -153.28084, -150.64697,
            -142.95134, -131.53622, -112.060684, -84.213806, -48.56169, -4.389044, 39.323967, 82.23867, 131.60728,
            188.0253, 252.98865, 327.37976, 410.17245, 496.62167, 585.2411, 681.78894, 780.2194, 872.71906, 963.9527,
            1053.583, 1134.785, 1210.8744, 1286.9423, 1360.0349, 1431.304, 1497.583, 1556.1243, 1613.6188, 1669.1587,
            1725.1412, 1782.7856, 1837.7366, 1891.3519, 1938.9788, 1982.6478, 2025.6543, 2063.728, 2100.2334, 2135.076,
            2167.2615, 2200.1794, 2239.5994, 2288.5435, 2340.1338, 2390.3252, 2438.338, 2485.7942, 2532.0488, 2574.268,
            2615.8083, 2655.2146, 2692.5005, 2727.98, 2761.047, 2795.1404, 2825.9316, 2854.8765, 2885.123, 2909.283,
            2923.8887, 2931.1963, 2932.656, 2926.6328, 2913.9312, 2892.6936, 2866.235, 2844.3591, 2823.9265, 2806.711,
            2794.646, 2781.6836, 2767.1228, 2745.9714, 2718.743, 2689.2458, 2656.3586, 2617.0393, 2569.0032, 2518.3962,
            2462.7202, 2401.1904, 2338.104, 2267.7605, 2192.1794, 2111.591, 2025.737, 1939.427, 1853.4957, 1772.1136,
            1692.258, 1610.623, 1526.9841, 1443.2959, 1366.169, 1287.5541, 1207.6376, 1131.6368, 1052.598, 975.5851,
            900.54694, 825.7228, 754.9538, 681.07074, 608.27386, 541.21387, 475.56976, 416.8674, 363.58557, 308.99783,
            255.32938, 202.76422, 148.18742, 94.41359, 46.84757, 4.685952, -34.290234, -70.559235, -104.25989,
            -133.03651, -156.44568, -178.25288, -201.39255, -224.6504, -249.4279, -275.6822, -294.5834, -305.23853,
            -310.82166, -309.75256, -303.62228, -290.3406, -269.21198, -240.65431, -204.61427, -170.5873, -139.5041,
            -106.93396, -79.36879, -56.56495,
        ];
        post_filter.run(&info, 320, &mut freq_samples);
        #[rustfmt::skip]
        let freq_samples_expected = [
            -320.74072, -339.79187, -361.7587, -390.47235, -417.69794, -439.90604, -456.2488, -468.38815, -483.43298,
            -496.43082, -508.12448, -522.58386, -530.04175, -528.8088, -525.5168, -520.4961, -519.71844, -533.91736,
            -559.6075, -589.22473, -622.2184, -657.9132, -692.0026, -717.29065, -735.2929, -748.3628, -758.1971,
            -771.0237, -779.54724, -782.6649, -785.8477, -785.47546, -784.49164, -781.64215, -779.1591, -779.5778,
            -776.40576, -775.1644, -777.1515, -784.9843, -803.82434, -825.20935, -848.8351, -873.379, -893.7613,
            -908.0107, -913.8168, -919.348, -927.2191, -936.5585, -948.37036, -958.8877, -974.6736, -998.22687,
            -1028.1508, -1068.1948, -1112.0266, -1157.626, -1202.3196, -1237.7783, -1265.9574, -1281.5967, -1285.4142,
            -1289.7588, -1290.598, -1289.4849, -1291.6206, -1289.05, -1280.279, -1264.467, -1238.673, -1204.2544,
            -1167.9618, -1135.2483, -1098.1372, -1053.269, -1001.1264, -940.4044, -875.56415, -804.1852, -723.755,
            -633.38324, -533.0927, -429.58737, -326.2695, -226.23605, -130.0739, -37.691048, 51.62319, 140.08931,
            219.42056, 290.84152, 362.6493, 430.15048, 489.2416, 538.8814, 577.8921, 606.0157, 625.485, 638.906,
            646.931, 652.6079, 656.9983, 657.0007, 648.3366, 635.7316, 624.73096, 611.2572, 598.15356, 581.7396,
            553.80884, 519.1469, 476.70868, 426.4432, 373.05707, 313.4432, 246.13098, 174.29283, 101.04027, 27.201643,
            -47.036148, -115.21831, -181.18599, -248.4486, -317.69327, -384.86273, -452.0945, -521.3707, -591.94867,
            -666.6193, -744.74194, -827.2992, -917.48413, -1010.70575, -1104.5869, -1201.524, -1294.2238, -1379.6155,
            -1460.6113, -1528.9102, -1585.4523, -1631.1124, -1660.9009, -1684.4255, -1699.1821, -1701.4526, -1700.5947,
            -1697.61, -1696.6571, -1702.1078, -1714.439, -1733.531, -1754.5884, -1775.7247, -1799.2393, -1827.3556,
            -1856.2937, -1885.4236, -1918.1951, -1954.9684, -1996.948, -2042.28, -2093.781, -2146.6353, -2189.093,
            -2223.9863, -2251.689, -2273.4807, -2294.2654, -2315.8933, -2340.1543, -2367.7327, -2405.2227, -2447.4607,
            -2489.6123, -2535.0264, -2575.6077, -2611.8645, -2641.9124, -2666.7974, -2697.137, -2726.5056, -2755.8135,
            -2787.091, -2813.767, -2836.5327, -2852.9668, -2865.6873, -2873.5032, -2868.0408, -2850.715, -2822.4873,
            -2783.2732, -2736.566, -2685.327, -2627.6494, -2560.0261, -2486.4858, -2411.283, -2334.7515, -2260.1433,
            -2189.3125, -2119.3108, -2052.565, -1992.5256, -1937.0294, -1888.7611, -1848.9229, -1812.4441, -1772.8254,
            -1722.6503, -1662.5598, -1594.4043, -1515.1047, -1426.463, -1329.6918, -1224.3643, -1116.0164, -1009.5359,
            -905.4838, -802.9583, -701.79346, -605.7693, -516.5092, -436.81702, -375.58536, -334.2002, -310.5628,
            -303.5844, -308.35977, -323.23022, -346.89465, -375.84933, -406.03006, -432.0379, -450.78473, -462.28992,
            -467.39557, -466.08475, -462.6616, -459.1948, -449.0974, -431.15015, -411.08737, -392.9422, -377.49863,
            -368.67184, -370.1563, -378.75293, -392.79608, -410.79825, -435.5704, -468.88922, -502.39908, -535.69073,
            -569.6185, -602.27716, -633.3312, -658.2254, -678.0781, -689.1164, -687.72736, -678.77576, -657.07355,
            -624.56256, -587.8258, -539.33655, -477.26443, -404.74258, -320.6686, -227.36028, -126.90553, -18.687975,
            97.70469, 221.64476, 350.7937, 485.1836, 624.7636, 768.85706, 918.2838, 1067.7938, 1215.2661, 1362.7802,
            1506.2122, 1643.0493, 1769.5411, 1888.3374, 2007.5011, 2116.3704, 2204.1897, 2276.0493, 2334.2388,
            2375.0833, 2400.9397, 2416.7896, 2419.0356, 2409.31, 2391.3875, 2358.868, 2315.0203, 2262.3967, 2193.9539,
            2113.851, 2024.538, 1927.651, 1832.4061, 1739.7405, 1653.9604, 1578.7108, 1505.152, 1431.2009, 1357.3208,
            1283.5349, 1210.578, 1134.9974, 1056.7867, 974.1136, 891.709, 815.5397, 739.52954, 666.8109, 599.267,
            531.3117, 464.1205, 399.3873, 337.9558, 276.83328, 217.15677, 159.31384, 99.04352, 44.807182, -1.9447366,
            -42.968796, -72.37995, -97.51363, -120.01941, -135.05333, -145.92494, -151.81703, -153.28084, -150.64697,
            -142.95134, -131.53622, -112.060684, -84.213806, -48.56169, -4.389044, 39.323967, 82.23867, 131.60728,
            188.0253, 252.98865, 327.37976, 410.17245, 496.62167, 585.2411, 681.78894, 780.2194, 872.71906, 963.9527,
            1053.583, 1134.785, 1210.8744, 1286.9423, 1360.0349, 1431.304, 1497.583, 1556.1243, 1613.6188, 1669.1587,
            1725.1412, 1782.7856, 1837.7366, 1891.3519, 1938.9788, 1982.6478, 2025.6543, 2063.728, 2100.2334, 2135.076,
            2167.2615, 2200.1794, 2239.5994, 2288.5435, 2340.1338, 2390.3252, 2438.338, 2485.7942, 2532.0488, 2574.268,
            2615.8083, 2655.2146, 2692.5005, 2727.98, 2761.047, 2795.1404, 2825.9316, 2854.8765, 2885.123, 2909.283,
            2923.8887, 2931.1963, 2932.656, 2926.6328, 2913.9312, 2892.6936, 2866.235, 2844.3591, 2823.9265, 2806.711,
            2794.646, 2781.6836, 2767.1228, 2745.9714, 2718.743, 2689.2458, 2656.3586, 2617.0393, 2569.0032, 2518.3962,
            2462.7202, 2401.1904, 2338.104, 2267.7605, 2192.1794, 2111.591, 2025.737, 1939.427, 1853.4957, 1772.1136,
            1692.258, 1610.623, 1526.9841, 1443.2959, 1366.169, 1287.5541, 1207.6376, 1131.6368, 1052.598, 975.5851,
            900.54694, 825.7228, 754.9538, 681.07074, 608.27386, 541.21387, 475.56976, 416.8674, 363.58557, 308.99783,
            255.32938, 202.76422, 148.18742, 94.41359, 46.84757, 4.685952, -34.290234, -70.559235, -104.25989,
            -133.03651, -156.44568, -178.25288, -201.39255, -224.6504, -249.4279, -275.6822, -294.5834, -305.23853,
            -310.82166, -309.75256, -303.62228, -290.3406, -269.21198, -240.65431, -204.61427, -170.5873, -139.5041,
            -106.93396, -79.36879, -56.56495,
        ];
        assert_eq!(freq_samples, freq_samples_expected);
    }
}
