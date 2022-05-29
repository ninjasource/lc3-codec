// since we support both 64 and 32 bit constants (via the Scalar floating point type alias) clippy complains
// when we squash a 64 bit constant into a 32 bit one but this is perfectly fine
#![allow(clippy::excessive_precision)]

// checked against spec

use crate::{
    common::{
        complex::Scaler,
        config::{FrameDuration, Lc3Config},
    },
    tables::long_term_post_filter_coef::*,
};
use itertools::*;
#[allow(unused_imports)]
use num_traits::real::Real;

const NMEM_12P8D: usize = 232;
const K_MIN: usize = 17;
const K_MAX: usize = 114;

pub struct LongTermPostFilter<'a> {
    // constant
    config: Lc3Config,
    len12p8: usize,
    len6p4: usize,
    delay_ltpf: usize,
    upsampling_factor: usize,  // 4 for 48000
    resampling_factor: Scaler, // (0.5 or 1.0)

    // state
    t_prev: usize,
    mem_pitch: Scaler,
    mem_ltpf_active: bool,
    mem_nc: Scaler,
    mem_mem_nc: Scaler,
    x_s_extended: &'a mut [i16],
    x_tilde_12p8d_extended: &'a mut [Scaler],
    x_6p4_extended: &'a mut [Scaler],
    h50_minus1: Scaler,
    h50_minus2: Scaler,
}

pub struct LongTermPostFilterResult {
    pub pitch_index: usize,
    pub pitch_present: bool,
    pub ltpf_active: bool,
    pub nbits_ltpf: usize,
}

struct LongTermPostFilterTempFields {
    pub len12p8: usize,
    pub len6p4: usize,
    pub delay_ltpf: usize,
    pub upsampling_factor: usize,
    pub resampling_factor: Scaler,
    pub x_s_extended_length: usize,
    pub x_tilde_12p8d_extended_length: usize,
    pub x_6p4_extended_length: usize,
}

impl<'a> LongTermPostFilter<'a> {
    pub fn new(
        config: Lc3Config,
        scaler_buf: &'a mut [Scaler],
        integer_buf: &'a mut [i16],
    ) -> (Self, &'a mut [Scaler], &'a mut [i16]) {
        let tmp = Self::calc_temp_fields(&config);
        let (x_s_extended, integer_buf) = integer_buf.split_at_mut(tmp.x_s_extended_length);
        let (x_tilde_12p8d_extended, scaler_buf) = scaler_buf.split_at_mut(tmp.x_tilde_12p8d_extended_length);
        let (x_6p4_extended, scaler_buf) = scaler_buf.split_at_mut(tmp.x_6p4_extended_length);

        (
            Self {
                config,
                delay_ltpf: tmp.delay_ltpf,
                len12p8: tmp.len12p8,
                len6p4: tmp.len6p4,
                mem_ltpf_active: false,
                mem_mem_nc: 0.0,
                mem_nc: 0.0,
                mem_pitch: 0.0,
                upsampling_factor: tmp.upsampling_factor,
                resampling_factor: tmp.resampling_factor,
                t_prev: K_MIN,
                x_6p4_extended,
                x_s_extended,
                x_tilde_12p8d_extended,
                h50_minus1: 0.0,
                h50_minus2: 0.0,
            },
            scaler_buf,
            integer_buf,
        )
    }

    fn calc_temp_fields(config: &Lc3Config) -> LongTermPostFilterTempFields {
        let (len12p8, len6p4, delay_ltpf) = match config.n_ms {
            FrameDuration::TenMs => (128, 64, 24),
            FrameDuration::SevenPointFiveMs => (96, 48, 44),
        };

        let upsampling_factor = if config.fs == 44100 { 4 } else { 192000 / config.fs };
        let resampling_factor = if config.fs == 8000 { 0.5 } else { 1.0 };

        // e.g. 60 + 480
        let x_s_extended_length = 240 / upsampling_factor + config.nf;

        // 3 * len12p8 for 10ms
        let x_tilde_12p8d_extended_length = len12p8 + delay_ltpf + NMEM_12P8D;

        let x_6p4_extended_length = 64 + K_MAX;

        LongTermPostFilterTempFields {
            len12p8,
            len6p4,
            delay_ltpf,
            upsampling_factor,
            resampling_factor,
            x_s_extended_length,
            x_tilde_12p8d_extended_length,
            x_6p4_extended_length,
        }
    }

    // returns (integer_len, complex_len)
    pub fn calc_working_buffer_length(config: &Lc3Config) -> (usize, usize) {
        let tmp = Self::calc_temp_fields(config);
        (
            tmp.x_s_extended_length,
            tmp.x_tilde_12p8d_extended_length + tmp.x_6p4_extended_length,
        )
    }

    pub fn run(&mut self, x_s: &[i16], near_nyquist_flag: bool, nbits: usize) -> LongTermPostFilterResult {
        assert_eq!(x_s.len(), self.config.nf);

        let t_nbits = match self.config.n_ms {
            FrameDuration::SevenPointFiveMs => (nbits as f64 * 10.0 / 7.5).round() as usize,
            FrameDuration::TenMs => nbits,
        };
        let gain_ltpf_on = t_nbits < 560 + self.config.fs_ind * 80;

        // time domain signals
        self.shift_out_old_samples(x_s);

        // resampling (TODO: go over this again) - it is very slow
        let x_12p8 = &mut self.x_tilde_12p8d_extended[self.delay_ltpf + NMEM_12P8D..];
        for (n, x_12p8_n) in x_12p8[..self.len12p8].iter_mut().enumerate() {
            *x_12p8_n = 0.0;
            let p = self.upsampling_factor as i32;
            for k in (-120 / p as i32)..=(120 / p as i32) {
                let index_x_s = (15 * n as i32) / p + k - 120 / p;
                let index_h = p * k - ((15 * n as i32) % p);
                if index_h > -120 && index_h < 120 {
                    *x_12p8_n += self.x_s_extended[(240 / p + index_x_s) as usize] as Scaler
                        * TAB_RESAMP_FILTER[(119 + index_h) as usize];
                }
            }

            *x_12p8_n *= p as Scaler * self.resampling_factor;
        }

        // high-pass filtering
        for x_12p8_n in x_12p8[..self.len12p8].iter_mut() {
            // calculate high-pass 50hz filter
            let h50 = *x_12p8_n - -1.9652933726226904 * self.h50_minus1 - 0.9658854605688177 * self.h50_minus2;
            *x_12p8_n =
                0.9827947082978771 * h50 + -1.965589416595754 * self.h50_minus1 + 0.9827947082978771 * self.h50_minus2;

            self.h50_minus2 = self.h50_minus1;
            self.h50_minus1 = h50;
        }

        // pitch detection algorithm (mutates self state)
        let (t_current, pitch_present) = self.pitch_detection();

        // ltpf pitch-lag parameter
        let (mut pitch_index, pitch_int, pitch_fr) = self.pitch_lag_parameter(t_current);

        // ltpf activation bit
        let (ltpf_active, mut nc, pitch) = self.activation_bit(pitch_int, pitch_fr, near_nyquist_flag, gain_ltpf_on);

        // ltpf bitstream
        let nbits_ltpf = if pitch_present { 11 } else { 1 };

        if !pitch_present {
            pitch_index = 0;
            nc = 0.0;
        }

        // prepare states for next run
        self.t_prev = t_current;
        self.mem_mem_nc = self.mem_nc;
        if pitch_present {
            self.mem_pitch = pitch;
            self.mem_ltpf_active = ltpf_active;
            self.mem_nc = nc;
        } else {
            self.mem_pitch = 0.0;
            self.mem_ltpf_active = false;
            self.mem_nc = 0.0;
        }

        LongTermPostFilterResult {
            ltpf_active,
            nbits_ltpf,
            pitch_index,
            pitch_present,
        }
    }

    fn shift_out_old_samples(&mut self, x_s: &[i16]) {
        // copy last 60 samples to the front of the buffer
        let num_samples = 240 / self.upsampling_factor;
        self.x_s_extended
            .copy_within(self.x_s_extended.len() - num_samples.., 0);

        // copy in the new samples to fill up the remainder
        self.x_s_extended[num_samples..].copy_from_slice(x_s);

        // shift by one frame
        self.x_tilde_12p8d_extended.copy_within(self.len12p8.., 0);
        self.x_6p4_extended.copy_within(self.len6p4.., 0);
    }

    // mutates self.x_6p4_extended and r_w_6p4
    fn pitch_detection(&mut self) -> (usize, bool) {
        // downsample 12.8khz to 6.4khz signal using a moving window of 5 samples and jumping by 2 (2x downsample)
        for (x_6p4, (s0, s1, s2, s3, s4)) in self.x_6p4_extended[K_MAX..K_MAX + self.len6p4].iter_mut().zip(
            self.x_tilde_12p8d_extended[NMEM_12P8D - 3..]
                .iter()
                .tuple_windows()
                .step_by(2),
        ) {
            *x_6p4 = 0.1236796411180537 * *s0
                + 0.2353512128364889 * *s1
                + 0.2819382920909148 * *s2
                + 0.2353512128364889 * *s3
                + 0.1236796411180537 * *s4;
        }

        // autocorrelation
        let mut r_6p4 = [0.0; K_MAX + 1 - K_MIN];
        let mut r_w_6p4 = [0.0; K_MAX + 1 - K_MIN];
        for (k, (r_6p4, r_w_6p4)) in r_6p4.iter_mut().zip(r_w_6p4.iter_mut()).enumerate() {
            let from = K_MAX;
            let from_k = K_MAX - K_MIN - k;
            *r_6p4 = self.x_6p4_extended[from..from + self.len6p4]
                .iter()
                .zip(self.x_6p4_extended[from_k..from_k + self.len6p4].iter())
                .map(|(x0, x1)| *x0 * *x1)
                .sum();
            let weight = 1.0 - 0.5 * k as Scaler / (K_MAX - K_MIN) as Scaler;
            *r_w_6p4 = weight * *r_6p4;
        }

        // first estimate of pitch-lag by getting the max of the weighted autocorrelation
        let lag_t1 = Self::index_of_max_value(&r_w_6p4) + K_MIN;

        // second estimate of pitch-lag by getting the max of the non-weighted autocorrelation
        let k_from = K_MIN.max(self.t_prev - 4) - K_MIN;
        let k_to = K_MAX.min(self.t_prev + 4) - K_MIN + 1;
        let lag_t2 = Self::index_of_max_value(&r_6p4[k_from..k_to]) + k_from + K_MIN;

        // calculate normalized correlation for lag-t1
        let normvalue_nolag = self.compute_normalized_value(0);
        let normvalue_lag_t1 = self.compute_normalized_value(lag_t1);
        let normvalue1 = (normvalue_nolag * normvalue_lag_t1).sqrt();
        let normcorr1 = 0.0.max(r_6p4[lag_t1 - K_MIN] as Scaler / normvalue1);

        // calculate normalized correlation for lag-t2
        let normcorr2 = if lag_t1 == lag_t2 {
            normcorr1
        } else {
            let normvalue_lag_t2 = self.compute_normalized_value(lag_t2);
            let normvalue2 = (normvalue_nolag * normvalue_lag_t2).sqrt();
            0.0.max(r_6p4[lag_t2 - K_MIN] as Scaler / normvalue2)
        };

        if normcorr2 > 0.85 * normcorr1 {
            (lag_t2, normcorr2 > 0.6)
        } else {
            (lag_t1, normcorr1 > 0.6)
        }
    }

    fn pitch_lag_parameter(&self, t_curr: usize) -> (usize, usize, i8) {
        let k_min = 32.max(2 * t_curr - 4);
        let k_max = 228.min(2 * t_curr + 4);

        let mut r_12p8_buf = [0.0; 228 + 4 + 1];
        let r_12p8 = &mut r_12p8_buf[..k_max + 4 - (k_min - 4) + 1];
        let mut max_correlation_val = 0.0;
        let mut pitch_int = k_min;

        for k in (k_min - 4)..=(k_max + 4) {
            let mut correlation_val = 0.0;

            for (n, nk) in self.x_tilde_12p8d_extended[NMEM_12P8D..NMEM_12P8D + self.len12p8]
                .iter()
                .zip(&self.x_tilde_12p8d_extended[NMEM_12P8D - k..NMEM_12P8D + self.len12p8 - k])
            {
                correlation_val += *n * *nk;
            }

            r_12p8[k - (k_min - 4)] = correlation_val;
            if correlation_val > max_correlation_val && k >= k_min && k <= k_max {
                max_correlation_val = correlation_val;
                pitch_int = k;
            }
        }

        let pitch_int_rel = pitch_int - (k_min - 4);
        let mut pitch_fr = 0i8;
        if pitch_int == 32 {
            let mut interp_d_max = 0.0;
            for d in 0..=3 {
                let interp_d = Self::interpolate(r_12p8, pitch_int_rel, d);
                if interp_d > interp_d_max {
                    interp_d_max = interp_d;
                    pitch_fr = d;
                }
            }
        } else if pitch_int < 127 && pitch_int > 32 {
            let mut interp_d_max = 0.0;
            for d in -3..=3 {
                let interp_d = Self::interpolate(r_12p8, pitch_int_rel, d);
                if interp_d > interp_d_max {
                    interp_d_max = interp_d;
                    pitch_fr = d;
                }
            }
        } else if (127..157).contains(&pitch_int) {
            let mut interp_d_max = 0.0;
            for d in (-2..=2).step_by(2) {
                let interp_d = Self::interpolate(r_12p8, pitch_int_rel, d);
                if interp_d > interp_d_max {
                    interp_d_max = interp_d;
                    pitch_fr = d;
                }
            }
        }

        if pitch_fr < 0 {
            pitch_int -= 1;
            pitch_fr += 4;
        }

        let pitch_index = if pitch_int < 127 {
            (4 * pitch_int as i32 + pitch_fr as i32 - 128) as usize
        } else if (127..157).contains(&pitch_int) {
            (2 * pitch_int as i32 + (pitch_fr as i32) / 2 - 126) as usize
        } else {
            pitch_int as usize + 283
        };

        (pitch_index, pitch_int, pitch_fr)
    }

    fn activation_bit(
        &self,
        pitch_int: usize,
        pitch_fr: i8,
        near_nyquist_flag: bool,
        gain_ltpf_on: bool,
    ) -> (bool, Scaler, Scaler) {
        let mut nc_numerator = 0.0;
        let mut no_delay_total = 0.0;
        let mut shifted_total = 0.0;

        for n in 0..self.len12p8 {
            let no_delay = self.dot_product(n as i16, 0);
            let shifted = self.dot_product(n as i16 - pitch_int as i16, pitch_fr as i8);
            nc_numerator += no_delay * shifted;
            no_delay_total += no_delay * no_delay;
            shifted_total += shifted * shifted;
        }

        let nc_denominator = (no_delay_total * shifted_total).sqrt();

        let nc = if nc_denominator > 0.0 {
            nc_numerator / nc_denominator
        } else {
            0.0
        };

        let pitch = pitch_int as Scaler + pitch_fr as Scaler / 4.0;

        let ltpf_active = if gain_ltpf_on && !near_nyquist_flag {
            !self.mem_ltpf_active
                && (self.config.n_ms == FrameDuration::TenMs || self.mem_mem_nc > 0.94)
                && self.mem_nc > 0.94
                && nc > 0.94
                || self.mem_ltpf_active && nc > 0.9
                || self.mem_ltpf_active
                    && (pitch - self.mem_pitch).abs() < 2.0
                    && (nc - self.mem_nc) > -0.1
                    && nc > 0.84
        } else {
            false
        };

        (ltpf_active, nc, pitch)
    }

    // see x(i)(n, d) in spec (ltpf activation bit section)
    fn dot_product(&self, n: i16, d: i8) -> Scaler {
        let mut result = 0.0;

        for k in -2..=2 {
            let h_i_index = 4 * k - d as i16;
            if h_i_index > -8 && h_i_index < 8 {
                result += (self.x_tilde_12p8d_extended[(NMEM_12P8D as i16 + n - k) as usize]
                    * TAB_LTPF_INTERP_X12K8[(h_i_index + 7) as usize]) as Scaler
            }
        }

        result
    }

    // this function is good enough for our use case
    fn index_of_max_value(slice: &[Scaler]) -> usize {
        if slice.is_empty() {
            return 0;
        }

        let mut max = slice[0];
        let mut index = 0;

        for (n, value) in slice.iter().enumerate() {
            if *value > max {
                index = n;
                max = *value;
            }
        }

        index
    }

    fn compute_normalized_value(&self, lag_t: usize) -> Scaler {
        let mut normvalue: Scaler = 0.0;
        let from = K_MAX - lag_t;
        let to = from + self.len6p4;

        for x_6p4 in &self.x_6p4_extended[from..to] {
            normvalue += *x_6p4 * *x_6p4;
        }

        normvalue
    }

    fn interpolate(r_12p8: &[Scaler], pitch_int_rel: usize, d: i8) -> Scaler {
        let mut interpolated = 0.0;
        for m in -4..=4 {
            let n = 4 * m - d;
            if (n > -16) && (n < 16) {
                let impulse_response = TAB_LTPF_INTERP_R[(n + 15) as usize] as Scaler;
                let signal = r_12p8[(pitch_int_rel as i32 + m as i32) as usize];
                interpolated += signal * impulse_response;
            }
        }

        interpolated
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::common::config::SamplingFrequency;

    #[test]
    fn long_term_post_filter_run() {
        let config = Lc3Config::new(SamplingFrequency::Hz48000, FrameDuration::TenMs, 1);
        let mut scaler_buf = [0.; 562];
        let mut integer_buf = [0; 540];
        let (mut post, _, _) = LongTermPostFilter::new(config, &mut scaler_buf, &mut integer_buf);
        let x_s = [
            836, 739, 638, 510, 352, 200, 72, -56, -177, -297, -416, -520, -623, -709, -791, -911, -1062, -1199, -1298,
            -1375, -1433, -1484, -1548, -1603, -1648, -1687, -1724, -1744, -1730, -1699, -1650, -1613, -1594, -1563,
            -1532, -1501, -1473, -1441, -1409, -1393, -1355, -1280, -1201, -1118, -1032, -953, -860, -741, -613, -477,
            -355, -261, -168, -80, -8, 57, 127, 217, 296, 347, 385, 413, 456, 517, 575, 640, 718, 806, 888, 963, 1041,
            1080, 1081, 1083, 1072, 1062, 1067, 1056, 1035, 1019, 999, 964, 934, 909, 876, 854, 835, 813, 795, 781,
            783, 772, 750, 747, 728, 713, 726, 716, 680, 638, 580, 516, 451, 393, 351, 307, 244, 161, 79, 18, -45,
            -123, -215, -301, -389, -512, -644, -764, -888, -1006, -1126, -1253, -1378, -1500, -1614, -1716, -1813,
            -1926, -2051, -2176, -2301, -2416, -2514, -2595, -2680, -2783, -2883, -2977, -3068, -3163, -3262, -3341,
            -3381, -3392, -3392, -3379, -3368, -3353, -3318, -3292, -3244, -3169, -3109, -3049, -2989, -2922, -2844,
            -2790, -2743, -2672, -2588, -2490, -2371, -2222, -2046, -1861, -1695, -1546, -1384, -1214, -1058, -913,
            -761, -602, -441, -280, -124, 24, 169, 302, 421, 546, 661, 738, 796, 851, 924, 1055, 1227, 1412, 1588,
            1707, 1787, 1853, 1905, 1963, 2015, 2048, 2072, 2082, 2093, 2099, 2095, 2097, 2086, 2063, 2060, 2069, 2052,
            2012, 1977, 1956, 1948, 1918, 1843, 1748, 1641, 1533, 1435, 1342, 1252, 1163, 1081, 1024, 989, 962, 937,
            911, 879, 841, 769, 657, 541, 445, 365, 289, 202, 104, -4, -119, -245, -381, -523, -655, -770, -874, -957,
            -1017, -1069, -1118, -1173, -1256, -1370, -1497, -1629, -1745, -1827, -1882, -1934, -2021, -2115, -2165,
            -2196, -2230, -2258, -2282, -2302, -2320, -2332, -2340, -2344, -2338, -2313, -2269, -2215, -2152, -2072,
            -1978, -1885, -1793, -1704, -1621, -1528, -1419, -1310, -1213, -1116, -1014, -914, -820, -736, -656, -578,
            -514, -445, -358, -276, -206, -136, -62, 0, 56, 124, 190, 253, 316, 379, 458, 552, 630, 686, 725, 735, 709,
            661, 612, 572, 538, 507, 476, 453, 448, 453, 444, 415, 370, 316, 257, 203, 159, 125, 107, 114, 137, 162,
            181, 189, 186, 166, 145, 145, 154, 154, 161, 184, 200, 217, 254, 294, 325, 332, 320, 302, 286, 273, 260,
            266, 294, 297, 274, 251, 221, 170, 100, 29, -31, -82, -134, -187, -232, -278, -347, -426, -490, -548, -613,
            -677, -727, -755, -769, -770, -757, -741, -729, -713, -684, -659, -647, -631, -606, -588, -585, -577, -555,
            -534, -527, -528, -513, -480, -456, -440, -415, -382, -333, -244, -132, -32, 47, 130, 225, 308, 383, 460,
            533, 607, 687, 757, 817, 889, 977, 1038, 1064, 1100, 1165, 1250, 1349, 1456, 1563, 1665, 1755, 1829, 1890,
            1935, 1973, 2008, 2033, 2044, 2054, 2076, 2106, 2125, 2115, 2097, 2092, 2093, 2082, 2067, 2068, 2095, 2135,
            2169, 2193, 2213, 2219, 2202, 2163, 2101, 2033, 1992, 1985, 1990, 1986, 1978, 1977, 1976, 1969, 1959, 1956,
            1960, 1955, 1930, 1907, 1884, 1844, 1790, 1733, 1687, 1649, 1611, 1586,
        ];

        let result = post.run(&x_s, false, 1200);

        assert_eq!(result.nbits_ltpf, 11);
        assert_eq!(result.pitch_present, true);
        assert_eq!(result.ltpf_active, false);
        assert_eq!(result.pitch_index, 0);
    }

    #[test]
    fn long_term_post_filter_active() {
        let config = Lc3Config::new(SamplingFrequency::Hz48000, FrameDuration::TenMs, 1);
        let mut scaler_buf = [0.; 562];
        let mut integer_buf = [0; 540];
        let (mut post, _, _) = LongTermPostFilter::new(config, &mut scaler_buf, &mut integer_buf);

        // NOTE: we need to throw a couple of frames at the filter beforehand to prime the state

        let x_s = [
            962, 1491, 1891, 2165, 2368, 2594, 2941, 3379, 3801, 4136, 4341, 4428, 4482, 4569, 4674, 4769, 4842, 4857,
            4792, 4681, 4554, 4399, 4213, 4031, 3891, 3778, 3631, 3425, 3184, 2937, 2713, 2530, 2357, 2168, 1982, 1799,
            1575, 1288, 964, 643, 362, 140, -20, -131, -217, -276, -314, -378, -479, -564, -580, -561, -598, -735,
            -943, -1147, -1258, -1227, -1109, -1030, -1050, -1144, -1291, -1462, -1576, -1562, -1467, -1388, -1358,
            -1352, -1330, -1251, -1087, -864, -657, -488, -313, -106, 124, 367, 611, 827, 1012, 1196, 1392, 1620, 1903,
            2217, 2529, 2841, 3162, 3503, 3899, 4363, 4855, 5314, 5690, 5997, 6324, 6741, 7250, 7841, 8480, 9102, 9679,
            10219, 10716, 11182, 11652, 12135, 12596, 12998, 13350, 13712, 14118, 14505, 14809, 15047, 15246, 15408,
            15526, 15561, 15505, 15428, 15383, 15306, 15134, 14899, 14662, 14452, 14270, 14108, 13947, 13755, 13510,
            13243, 13021, 12862, 12711, 12505, 12212, 11855, 11493, 11150, 10811, 10485, 10191, 9906, 9572, 9140, 8604,
            8036, 7522, 7069, 6599, 6059, 5467, 4872, 4304, 3768, 3261, 2784, 2339, 1928, 1567, 1256, 947, 581, 189,
            -123, -276, -286, -229, -192, -212, -249, -254, -225, -175, -104, -23, 52, 117, 147, 122, 101, 180, 372,
            611, 838, 1034, 1196, 1341, 1520, 1776, 2084, 2394, 2691, 2959, 3157, 3291, 3437, 3655, 3952, 4300, 4635,
            4891, 5063, 5174, 5222, 5216, 5221, 5278, 5337, 5324, 5238, 5116, 4968, 4783, 4575, 4390, 4268, 4174, 4024,
            3789, 3521, 3245, 2959, 2705, 2517, 2352, 2171, 1967, 1712, 1395, 1103, 939, 872, 773, 589, 370, 145, -118,
            -412, -676, -893, -1088, -1265, -1440, -1629, -1793, -1914, -2047, -2212, -2353, -2449, -2518, -2533,
            -2466, -2349, -2226, -2103, -1965, -1793, -1565, -1279, -956, -615, -263, 93, 416, 669, 883, 1123, 1425,
            1782, 2161, 2527, 2870, 3213, 3597, 4050, 4537, 4959, 5260, 5487, 5712, 5972, 6265, 6547, 6758, 6909, 7069,
            7270, 7492, 7701, 7850, 7897, 7856, 7779, 7711, 7667, 7642, 7623, 7561, 7373, 7039, 6658, 6336, 6101, 5927,
            5726, 5364, 4806, 4198, 3703, 3359, 3103, 2837, 2476, 1989, 1402, 811, 358, 92, -98, -341, -688, -1163,
            -1744, -2337, -2850, -3253, -3553, -3797, -4055, -4351, -4680, -5048, -5452, -5870, -6286, -6681, -7035,
            -7371, -7723, -8086, -8432, -8735, -8964, -9107, -9181, -9221, -9296, -9476, -9739, -9991, -10177, -10305,
            -10369, -10331, -10206, -10086, -10033, -9968, -9749, -9327, -8760, -8150, -7585, -7088, -6609, -6106,
            -5604, -5138, -4678, -4171, -3627, -3115, -2675, -2279, -1866, -1384, -838, -311, 114, 440, 713, 973, 1253,
            1583, 1970, 2415, 2914, 3443, 3963, 4410, 4716, 4907, 5086, 5295, 5478, 5603, 5700, 5796, 5907, 6039, 6170,
            6255, 6275, 6243, 6176, 6085, 5970, 5808, 5564, 5246, 4892, 4511, 4099, 3698, 3356, 3050, 2724, 2382, 2079,
            1859, 1716, 1596, 1428, 1181, 919, 741, 681, 676, 654, 595, 521, 462, 430, 408, 370, 286, 126, -102, -335,
            -497, -548, -494, -401, -369, -446, -576, -644, -556, -321, -57, 117, 186, 226, 338, 583, 977, 1489, 2026,
            2460, 2732, 2912, 3133, 3460, 3830, 4128, 4310, 4419, 4499, 4569, 4627, 4670, 4752, 4935,
        ];

        let result = post.run(&x_s, false, 400);
        assert_eq!(result.ltpf_active, false);
        assert_eq!(result.pitch_present, false);
        assert_eq!(result.pitch_index, 0);
        assert_eq!(result.nbits_ltpf, 1);

        let x_s = [
            5149, 5313, 5401, 5440, 5457, 5495, 5583, 5675, 5720, 5707, 5626, 5453, 5209, 4969, 4825, 4827, 4950, 5109,
            5182, 5060, 4756, 4399, 4058, 3709, 3370, 3102, 2914, 2759, 2564, 2265, 1887, 1526, 1253, 1068, 936, 790,
            584, 329, 69, -146, -282, -335, -302, -183, -37, 47, 73, 131, 265, 454, 681, 931, 1160, 1326, 1437, 1539,
            1657, 1800, 1984, 2193, 2362, 2474, 2583, 2683, 2712, 2711, 2796, 3014, 3309, 3611, 3919, 4285, 4737, 5226,
            5682, 6111, 6565, 7064, 7596, 8153, 8721, 9245, 9682, 10067, 10461, 10917, 11459, 12049, 12607, 13089,
            13511, 13892, 14234, 14547, 14809, 14987, 15095, 15176, 15241, 15302, 15389, 15461, 15416, 15234, 15015,
            14811, 14552, 14163, 13654, 13079, 12529, 12098, 11788, 11514, 11194, 10750, 10137, 9428, 8740, 8125, 7609,
            7193, 6785, 6261, 5584, 4831, 4163, 3732, 3533, 3392, 3137, 2700, 2084, 1346, 586, -119, -744, -1320,
            -1921, -2593, -3308, -4016, -4643, -5074, -5255, -5283, -5341, -5573, -5994, -6499, -6972, -7384, -7745,
            -8064, -8383, -8746, -9171, -9653, -10145, -10542, -10764, -10831, -10791, -10671, -10530, -10446, -10445,
            -10522, -10657, -10775, -10775, -10640, -10446, -10272, -10189, -10235, -10325, -10295, -10045, -9616,
            -9124, -8663, -8257, -7870, -7448, -6931, -6314, -5682, -5147, -4783, -4612, -4590, -4614, -4597, -4522,
            -4400, -4248, -4099, -4017, -4056, -4181, -4270, -4219, -4015, -3714, -3377, -3081, -2913, -2885, -2917,
            -2889, -2734, -2500, -2315, -2252, -2296, -2432, -2662, -2972, -3344, -3753, -4131, -4382, -4471, -4431,
            -4316, -4158, -3951, -3681, -3369, -3041, -2687, -2294, -1894, -1515, -1175, -899, -678, -473, -290, -231,
            -401, -785, -1281, -1791, -2268, -2720, -3173, -3636, -4087, -4492, -4865, -5284, -5808, -6391, -6925,
            -7299, -7468, -7505, -7525, -7586, -7679, -7811, -7989, -8171, -8332, -8488, -8642, -8791, -8941, -9044,
            -8997, -8741, -8294, -7720, -7141, -6693, -6418, -6263, -6149, -6011, -5845, -5708, -5646, -5657, -5686,
            -5645, -5484, -5218, -4865, -4433, -3986, -3600, -3273, -2959, -2646, -2328, -2025, -1811, -1741, -1798,
            -1924, -2048, -2121, -2144, -2125, -2028, -1806, -1465, -1094, -835, -793, -957, -1248, -1602, -1979,
            -2343, -2696, -3088, -3576, -4207, -4981, -5778, -6425, -6851, -7087, -7171, -7142, -7046, -6883, -6631,
            -6307, -5953, -5618, -5346, -5128, -4915, -4679, -4407, -4072, -3669, -3191, -2626, -2026, -1468, -938,
            -367, 281, 1016, 1819, 2642, 3449, 4201, 4825, 5305, 5719, 6127, 6542, 6975, 7396, 7750, 8040, 8273, 8417,
            8482, 8529, 8614, 8777, 9037, 9401, 9894, 10519, 11186, 11774, 12269, 12737, 13206, 13675, 14150, 14614,
            15022, 15354, 15604, 15771, 15883, 15932, 15844, 15584, 15187, 14670, 14068, 13469, 12916, 12369, 11807,
            11219, 10546, 9788, 8984, 8109, 7176, 6292, 5534, 4922, 4478, 4137, 3752, 3262, 2713, 2132, 1516, 882, 227,
            -469, -1189, -1902, -2562, -3137, -3609, -3938, -4136, -4316, -4610, -5069, -5642, -6237, -6802, -7342,
            -7878, -8397, -8874, -9289, -9645, -9937, -10118, -10154, -10080, -9944, -9739, -9476, -9222, -9047, -8991,
            -9053, -9211, -9474, -9858, -10301, -10691, -10982, -11198, -11347, -11398, -11318, -11151, -11011, -10966,
            -10996, -11090, -11257, -11481, -11676, -11731, -11636, -11513, -11493, -11613, -11825, -12023, -12090,
            -11997, -11788, -11500, -11146, -10729, -10230, -9643, -8994, -8345, -7792,
        ];

        let result = post.run(&x_s, false, 400);
        assert_eq!(result.ltpf_active, false);
        assert_eq!(result.pitch_present, false);
        assert_eq!(result.pitch_index, 0);
        assert_eq!(result.nbits_ltpf, 1);

        let x_s = [
            -7361, -7081, -6851, -6564, -6160, -5635, -5054, -4475, -3907, -3320, -2712, -2106, -1512, -909, -313, 208,
            616, 917, 1093, 1167, 1210, 1243, 1266, 1337, 1498, 1683, 1795, 1796, 1698, 1539, 1364, 1220, 1153, 1171,
            1236, 1302, 1345, 1409, 1576, 1812, 1953, 1956, 1928, 1906, 1840, 1762, 1761, 1852, 1977, 2073, 2112, 2071,
            1898, 1561, 1126, 705, 341, 6, -347, -732, -1121, -1497, -1886, -2320, -2832, -3428, -4042, -4578, -4975,
            -5232, -5377, -5422, -5374, -5303, -5333, -5530, -5876, -6317, -6757, -7102, -7361, -7618, -7915, -8253,
            -8609, -8920, -9127, -9224, -9257, -9344, -9599, -10007, -10449, -10854, -11214, -11521, -11772, -11994,
            -12235, -12553, -12961, -13346, -13546, -13516, -13359, -13211, -13124, -13098, -13160, -13336, -13621,
            -13998, -14450, -14914, -15323, -15673, -15976, -16201, -16328, -16370, -16332, -16222, -16125, -16122,
            -16212, -16352, -16515, -16623, -16549, -16274, -15931, -15672, -15579, -15647, -15765, -15765, -15560,
            -15173, -14629, -13950, -13206, -12463, -11716, -10918, -10058, -9163, -8277, -7450, -6707, -6033, -5391,
            -4760, -4118, -3433, -2695, -1935, -1213, -612, -162, 201, 554, 873, 1089, 1186, 1205, 1205, 1234, 1299,
            1387, 1530, 1775, 2086, 2351, 2496, 2534, 2529, 2549, 2629, 2768, 2977, 3305, 3771, 4324, 4893, 5408, 5811,
            6103, 6353, 6619, 6878, 7075, 7213, 7342, 7482, 7595, 7678, 7771, 7852, 7817, 7585, 7192, 6795, 6542, 6433,
            6371, 6284, 6130, 5880, 5550, 5158, 4666, 4078, 3488, 3002, 2685, 2553, 2538, 2536, 2498, 2399, 2179, 1803,
            1324, 846, 453, 157, -130, -535, -1079, -1609, -1927, -1987, -1916, -1872, -1937, -2103, -2320, -2549,
            -2757, -2888, -2925, -2914, -2918, -2971, -3065, -3140, -3121, -2971, -2686, -2311, -1943, -1624, -1325,
            -1060, -881, -811, -855, -993, -1140, -1237, -1313, -1398, -1480, -1549, -1570, -1468, -1227, -942, -752,
            -725, -832, -989, -1082, -1012, -752, -366, 29, 285, 318, 224, 188, 285, 490, 795, 1184, 1611, 2054, 2536,
            3090, 3710, 4322, 4845, 5279, 5671, 6056, 6463, 6887, 7308, 7774, 8350, 8995, 9619, 10211, 10787, 11326,
            11812, 12256, 12647, 12977, 13287, 13604, 13923, 14275, 14710, 15192, 15643, 16048, 16389, 16606, 16699,
            16733, 16737, 16747, 16844, 17023, 17190, 17318, 17455, 17616, 17785, 17954, 18088, 18140, 18122, 18067,
            17982, 17869, 17743, 17630, 17577, 17608, 17661, 17605, 17372, 17021, 16660, 16335, 16016, 15668, 15304,
            14937, 14536, 14070, 13539, 12934, 12221, 11398, 10526, 9692, 8969, 8378, 7891, 7472, 7094, 6703, 6256,
            5772, 5288, 4794, 4287, 3828, 3454, 3093, 2654, 2148, 1660, 1253, 941, 710, 527, 359, 173, -82, -432, -838,
            -1225, -1535, -1765, -1963, -2146, -2296, -2409, -2473, -2436, -2301, -2159, -2053, -1955, -1857, -1775,
            -1720, -1727, -1824, -1997, -2238, -2580, -3002, -3417, -3762, -4031, -4237, -4387, -4511, -4660, -4881,
            -5187, -5543, -5867, -6117, -6316, -6455, -6514, -6568, -6691, -6839, -6971, -7112, -7226, -7241, -7181,
            -7082, -6900, -6587, -6130, -5536, -4893, -4328, -3850, -3373, -2884, -2428, -1990, -1508, -962, -368, 253,
            859, 1398, 1853, 2223, 2501, 2712, 2872, 2941, 2917, 2906, 2997, 3191, 3469, 3806, 4164, 4543, 4933, 5272,
            5523, 5701, 5819, 5923, 6094, 6343, 6609, 6834, 6975, 7044, 7120, 7245, 7355, 7373, 7290, 7170,
        ];

        let result = post.run(&x_s, false, 400);
        assert_eq!(result.ltpf_active, false);
        assert_eq!(result.pitch_present, true);
        assert_eq!(result.pitch_index, 180);
        assert_eq!(result.nbits_ltpf, 11);

        let x_s = [
            7031, 6895, 6703, 6508, 6426, 6408, 6321, 6143, 5908, 5615, 5296, 5033, 4835, 4653, 4488, 4364, 4262, 4146,
            3998, 3784, 3454, 2999, 2494, 2076, 1844, 1771, 1763, 1760, 1717, 1590, 1386, 1131, 840, 562, 339, 112,
            -173, -470, -761, -1072, -1361, -1576, -1754, -1915, -2024, -2114, -2236, -2377, -2527, -2694, -2841,
            -2936, -3021, -3100, -3114, -3062, -2985, -2884, -2749, -2603, -2446, -2247, -2027, -1841, -1725, -1692,
            -1731, -1796, -1822, -1815, -1852, -1970, -2099, -2128, -2033, -1859, -1670, -1557, -1585, -1730, -1911,
            -2067, -2147, -2115, -2021, -1915, -1784, -1642, -1537, -1463, -1395, -1352, -1304, -1150, -857, -483, -77,
            398, 975, 1633, 2347, 3063, 3730, 4385, 5069, 5745, 6372, 7014, 7751, 8539, 9300, 10039, 10783, 11522,
            12203, 12782, 13305, 13833, 14313, 14635, 14804, 14941, 15141, 15407, 15716, 16034, 16341, 16646, 16948,
            17193, 17328, 17377, 17411, 17459, 17520, 17608, 17722, 17825, 17888, 17920, 17908, 17851, 17795, 17727,
            17576, 17357, 17158, 17004, 16873, 16752, 16619, 16444, 16219, 15921, 15526, 15052, 14545, 14058, 13631,
            13266, 12939, 12634, 12359, 12099, 11774, 11331, 10789, 10150, 9403, 8637, 7978, 7476, 7119, 6823, 6451,
            5940, 5303, 4574, 3822, 3168, 2696, 2358, 2041, 1639, 1079, 393, -278, -754, -977, -1085, -1245, -1528,
            -1941, -2420, -2834, -3082, -3193, -3229, -3244, -3308, -3387, -3394, -3337, -3263, -3163, -2998, -2808,
            -2633, -2442, -2232, -2057, -1956, -1888, -1824, -1784, -1771, -1801, -1897, -2039, -2163, -2196, -2143,
            -2097, -2196, -2517, -2968, -3397, -3756, -4067, -4319, -4473, -4558, -4674, -4887, -5174, -5455, -5669,
            -5803, -5897, -5979, -6016, -6004, -5981, -5902, -5695, -5394, -5071, -4781, -4567, -4428, -4291, -4082,
            -3779, -3378, -2897, -2384, -1893, -1460, -1073, -707, -373, -111, 80, 204, 218, 129, 9, -109, -201, -264,
            -302, -272, -158, -69, -126, -286, -427, -516, -548, -488, -349, -200, -99, -66, -65, -32, 48, 121, 155,
            177, 177, 110, 11, -72, -161, -258, -300, -270, -251, -333, -521, -756, -956, -1062, -1081, -1061, -1059,
            -1103, -1161, -1219, -1320, -1519, -1834, -2203, -2537, -2835, -3166, -3523, -3809, -4022, -4273, -4655,
            -5173, -5751, -6293, -6766, -7185, -7572, -7965, -8410, -8941, -9575, -10280, -10949, -11477, -11878,
            -12247, -12664, -13162, -13714, -14243, -14691, -15031, -15234, -15290, -15255, -15187, -15087, -14961,
            -14823, -14679, -14533, -14390, -14261, -14150, -14067, -14029, -14047, -14133, -14262, -14371, -14466,
            -14609, -14787, -14945, -15051, -15078, -15003, -14850, -14733, -14769, -14950, -15155, -15234, -15153,
            -14983, -14754, -14507, -14336, -14261, -14213, -14132, -13983, -13757, -13507, -13264, -12960, -12517,
            -11971, -11390, -10770, -10118, -9492, -8902, -8330, -7806, -7361, -6951, -6483, -5906, -5230, -4499,
            -3763, -3084, -2518, -2061, -1682, -1392, -1225, -1152, -1101, -1036, -962, -910, -923, -1003, -1092,
            -1133, -1135, -1154, -1249, -1443, -1705, -1940, -2080, -2141, -2144, -2081, -1999, -1997, -2107, -2275,
            -2445, -2571, -2636, -2686, -2753, -2813, -2835, -2843, -2886, -2989, -3123, -3208, -3228, -3231, -3244,
            -3278, -3341, -3435, -3519, -3524, -3468, -3441, -3512, -3672, -3866, -4061, -4267, -4505, -4770, -5012,
            -5180, -5283, -5347, -5386, -5464, -5657, -5987, -6354, -6606, -6698, -6658, -6561, -6522, -6633, -6892,
            -7183, -7396, -7482,
        ];

        let result = post.run(&x_s, false, 400);
        assert_eq!(result.ltpf_active, false);
        assert_eq!(result.pitch_present, true);
        assert_eq!(result.pitch_index, 184);
        assert_eq!(result.nbits_ltpf, 11);

        let x_s = [
            -7458, -7341, -7199, -7111, -7155, -7327, -7511, -7582, -7515, -7319, -7004, -6648, -6329, -6104, -5970,
            -5821, -5567, -5220, -4869, -4591, -4390, -4237, -4107, -3961, -3784, -3616, -3509, -3476, -3497, -3541,
            -3595, -3654, -3690, -3668, -3610, -3620, -3787, -4113, -4515, -4882, -5154, -5371, -5611, -5888, -6162,
            -6392, -6527, -6575, -6623, -6696, -6723, -6638, -6420, -6069, -5616, -5094, -4526, -3965, -3451, -2993,
            -2567, -2125, -1644, -1129, -569, 54, 727, 1398, 2062, 2726, 3329, 3808, 4175, 4492, 4804, 5119, 5456,
            5839, 6242, 6647, 7052, 7415, 7697, 7911, 8097, 8283, 8472, 8681, 8939, 9257, 9619, 9999, 10412, 10859,
            11291, 11664, 11935, 12087, 12151, 12194, 12252, 12309, 12343, 12340, 12278, 12134, 11926, 11703, 11502,
            11317, 11095, 10807, 10489, 10181, 9953, 9848, 9815, 9759, 9618, 9395, 9112, 8816, 8557, 8319, 8057, 7701,
            7226, 6713, 6226, 5764, 5296, 4789, 4225, 3612, 2966, 2313, 1713, 1230, 864, 545, 207, -183, -665, -1232,
            -1825, -2398, -2850, -3105, -3239, -3401, -3693, -4111, -4559, -4906, -5063, -5041, -4935, -4869, -4924,
            -5067, -5209, -5285, -5267, -5162, -5023, -4934, -4943, -5047, -5238, -5491, -5738, -5933, -6088, -6209,
            -6327, -6514, -6787, -7073, -7293, -7458, -7614, -7781, -7972, -8156, -8288, -8365, -8414, -8476, -8547,
            -8572, -8555, -8514, -8436, -8317, -8176, -8015, -7799, -7486, -7057, -6545, -6052, -5625, -5187, -4706,
            -4220, -3713, -3152, -2562, -1988, -1453, -941, -424, 131, 755, 1442, 2125, 2732, 3233, 3594, 3788, 3884,
            3964, 4049, 4158, 4296, 4396, 4377, 4227, 4027, 3890, 3876, 3934, 3962, 3949, 3942, 3967, 4048, 4192, 4363,
            4550, 4756, 4913, 4943, 4841, 4661, 4466, 4310, 4236, 4192, 4082, 3913, 3762, 3654, 3555, 3430, 3302, 3226,
            3218, 3220, 3137, 2968, 2832, 2816, 2900, 3010, 3063, 3019, 2890, 2715, 2523, 2324, 2146, 2016, 1926, 1842,
            1696, 1488, 1278, 1095, 954, 825, 653, 464, 315, 189, 36, -142, -351, -653, -1032, -1389, -1648, -1787,
            -1866, -1994, -2229, -2529, -2798, -2947, -2956, -2863, -2753, -2713, -2741, -2770, -2744, -2645, -2480,
            -2271, -2060, -1884, -1773, -1760, -1824, -1870, -1851, -1792, -1734, -1719, -1768, -1871, -2000, -2095,
            -2110, -2041, -1921, -1812, -1766, -1775, -1787, -1775, -1779, -1819, -1871, -1931, -1971, -1949, -1887,
            -1813, -1718, -1589, -1422, -1216, -954, -635, -293, 38, 378, 787, 1275, 1831, 2477, 3181, 3864, 4466,
            4971, 5421, 5894, 6430, 6981, 7522, 8073, 8567, 8936, 9220, 9486, 9792, 10169, 10566, 10906, 11191, 11464,
            11708, 11906, 12095, 12309, 12542, 12793, 13085, 13426, 13833, 14306, 14804, 15288, 15732, 16117, 16444,
            16730, 17005, 17243, 17382, 17440, 17489, 17538, 17518, 17404, 17264, 17146, 17011, 16824, 16579, 16282,
            15968, 15663, 15353, 14978, 14560, 14216, 13983, 13785, 13544, 13225, 12871, 12557, 12285, 11997, 11654,
            11223, 10687, 10111, 9606, 9200, 8812, 8393, 7917, 7340, 6702, 6077, 5506, 5006, 4569, 4148, 3631, 2955,
            2196, 1461, 834, 370, 75, -103, -266, -491, -783, -1080, -1246, -1231, -1122, -974, -809, -713, -787, -997,
            -1189, -1257, -1202, -1073, -946, -859, -832, -860, -880, -850, -757, -635, -578, -630, -752, -917, -1111,
            -1268, -1324, -1317, -1337, -1411, -1517, -1660, -1832, -2059, -2367, -2657,
        ];

        let result = post.run(&x_s, false, 400);
        assert_eq!(result.ltpf_active, false);
        assert_eq!(result.pitch_present, true);
        assert_eq!(result.pitch_index, 477);
        assert_eq!(result.nbits_ltpf, 11);

        let x_s = [
            -2767, -2730, -2656, -2635, -2707, -2875, -3083, -3217, -3204, -3058, -2850, -2660, -2545, -2493, -2442,
            -2342, -2209, -2078, -1937, -1773, -1596, -1393, -1139, -821, -468, -122, 198, 485, 730, 905, 1023, 1148,
            1301, 1437, 1505, 1507, 1498, 1482, 1385, 1178, 953, 843, 895, 1063, 1297, 1567, 1829, 2049, 2245, 2472,
            2723, 2923, 3033, 3076, 3091, 3100, 3120, 3160, 3186, 3138, 2971, 2685, 2324, 1970, 1689, 1490, 1364, 1294,
            1246, 1181, 1070, 932, 808, 765, 846, 974, 1038, 1036, 1020, 988, 897, 759, 597, 382, 94, -257, -659,
            -1075, -1414, -1608, -1745, -1986, -2343, -2730, -3109, -3490, -3879, -4277, -4702, -5200, -5788, -6408,
            -6954, -7359, -7646, -7902, -8248, -8733, -9264, -9739, -10120, -10393, -10588, -10774, -10994, -11258,
            -11549, -11825, -12028, -12137, -12207, -12306, -12484, -12730, -12978, -13185, -13343, -13473, -13636,
            -13890, -14195, -14490, -14795, -15091, -15326, -15525, -15702, -15822, -15889, -15966, -16103, -16279,
            -16464, -16596, -16613, -16552, -16471, -16371, -16238, -16109, -16023, -15951, -15831, -15603, -15274,
            -14936, -14612, -14252, -13834, -13334, -12704, -11942, -11116, -10311, -9573, -8873, -8163, -7435, -6710,
            -6005, -5314, -4634, -3979, -3381, -2891, -2520, -2197, -1849, -1474, -1134, -864, -661, -524, -481, -547,
            -628, -603, -463, -273, -86, 84, 272, 513, 777, 1021, 1238, 1397, 1446, 1377, 1222, 1036, 888, 816, 769,
            682, 551, 368, 113, -192, -488, -696, -786, -813, -850, -929, -1054, -1237, -1469, -1671, -1755, -1708,
            -1611, -1564, -1585, -1627, -1658, -1696, -1760, -1860, -2026, -2263, -2525, -2761, -2930, -3006, -3001,
            -2998, -3102, -3326, -3574, -3751, -3858, -3962, -4144, -4464, -4905, -5411, -5899, -6242, -6372, -6365,
            -6381, -6500, -6663, -6766, -6746, -6620, -6416, -6138, -5847, -5632, -5501, -5403, -5296, -5141, -4924,
            -4712, -4548, -4413, -4290, -4158, -4004, -3877, -3835, -3854, -3872, -3900, -3988, -4126, -4263, -4350,
            -4353, -4288, -4210, -4141, -4104, -4140, -4275, -4497, -4745, -4967, -5121, -5169, -5118, -4996, -4871,
            -4825, -4814, -4700, -4421, -4061, -3725, -3420, -3104, -2758, -2388, -1982, -1521, -1014, -525, -118, 213,
            537, 936, 1442, 2003, 2558, 3105, 3651, 4156, 4589, 4986, 5400, 5848, 6279, 6630, 6904, 7125, 7290, 7378,
            7403, 7438, 7569, 7817, 8127, 8450, 8775, 9125, 9541, 9988, 10376, 10663, 10883, 11081, 11281, 11508,
            11732, 11905, 12057, 12225, 12378, 12465, 12469, 12385, 12239, 12126, 12111, 12182, 12286, 12369, 12402,
            12360, 12244, 12093, 11975, 11957, 12030, 12099, 12065, 11914, 11733, 11580, 11427, 11225, 10937, 10532,
            10033, 9515, 9070, 8731, 8433, 8095, 7676, 7148, 6516, 5858, 5311, 4913, 4573, 4175, 3635, 2981, 2314,
            1720, 1249, 915, 661, 350, -100, -628, -1108, -1452, -1679, -1858, -2018, -2185, -2399, -2671, -2958,
            -3210, -3379, -3437, -3427, -3413, -3418, -3469, -3589, -3721, -3794, -3836, -3931, -4107, -4335, -4589,
            -4862, -5127, -5324, -5446, -5541, -5654, -5838, -6122, -6442, -6699, -6870, -6978, -7013, -6968, -6876,
            -6747, -6579, -6386, -6204, -6057, -5940, -5816, -5650, -5452, -5249, -5032, -4771, -4452, -4091, -3712,
            -3361, -3085, -2873, -2671, -2413, -2051, -1612, -1191, -869, -629, -405, -185, 3, 125, 194, 250, 281, 233,
            98, -102, -356, -613, -780, -824, -738, -514,
        ];

        let result = post.run(&x_s, false, 400);
        assert_eq!(result.ltpf_active, false);
        assert_eq!(result.pitch_present, true);
        assert_eq!(result.pitch_index, 478);
        assert_eq!(result.nbits_ltpf, 11);

        let x_s = [
            -220, 83, 436, 924, 1513, 2089, 2567, 2913, 3134, 3257, 3300, 3295, 3307, 3350, 3370, 3329, 3234, 3107,
            2951, 2774, 2612, 2531, 2562, 2664, 2781, 2863, 2884, 2846, 2749, 2622, 2561, 2629, 2761, 2854, 2906, 2942,
            2956, 2924, 2790, 2551, 2265, 1938, 1557, 1202, 1016, 1019, 1055, 988, 823, 600, 351, 136, 45, 93, 153, 83,
            -163, -522, -836, -968, -895, -684, -443, -288, -272, -352, -404, -293, -9, 313, 544, 667, 734, 770, 789,
            834, 927, 1066, 1219, 1308, 1294, 1217, 1114, 999, 899, 816, 701, 488, 192, -110, -362, -524, -572, -518,
            -430, -387, -446, -662, -1001, -1337, -1593, -1774, -1887, -1908, -1850, -1758, -1638, -1490, -1318, -1120,
            -894, -635, -374, -136, 117, 427, 814, 1319, 1952, 2627, 3234, 3738, 4148, 4510, 4937, 5527, 6226, 6887,
            7427, 7858, 8254, 8670, 9085, 9448, 9769, 10096, 10427, 10715, 10900, 10955, 10928, 10924, 11063, 11395,
            11876, 12445, 13050, 13641, 14187, 14723, 15302, 15895, 16445, 16905, 17221, 17382, 17444, 17484, 17539,
            17613, 17680, 17680, 17590, 17439, 17248, 17027, 16816, 16681, 16626, 16587, 16497, 16329, 16072, 15725,
            15339, 14978, 14672, 14433, 14220, 13959, 13644, 13316, 12992, 12649, 12270, 11853, 11377, 10853, 10356,
            9966, 9698, 9508, 9302, 8966, 8470, 7891, 7330, 6877, 6572, 6378, 6175, 5840, 5329, 4717, 4191, 3888, 3774,
            3743, 3705, 3582, 3317, 2967, 2712, 2692, 2902, 3171, 3304, 3232, 3015, 2797, 2702, 2742, 2853, 2961, 2999,
            2922, 2747, 2529, 2316, 2148, 2057, 2006, 1908, 1740, 1523, 1284, 1088, 977, 932, 904, 859, 763, 578, 317,
            28, -213, -346, -393, -400, -389, -375, -361, -365, -397, -439, -452, -393, -275, -159, -98, -91, -80, -21,
            96, 283, 503, 667, 741, 776, 827, 909, 998, 1091, 1222, 1425, 1683, 1914, 2077, 2227, 2402, 2564, 2644,
            2582, 2331, 1911, 1416, 954, 616, 462, 499, 672, 909, 1144, 1342, 1515, 1694, 1899, 2137, 2393, 2589, 2665,
            2649, 2585, 2502, 2404, 2266, 2077, 1866, 1657, 1439, 1188, 893, 589, 354, 236, 176, 49, -205, -543, -865,
            -1064, -1087, -953, -747, -592, -574, -672, -786, -828, -800, -794, -888, -1067, -1305, -1578, -1779,
            -1842, -1858, -1981, -2258, -2648, -3112, -3597, -4013, -4299, -4502, -4770, -5197, -5755, -6378, -6978,
            -7454, -7770, -8012, -8305, -8717, -9225, -9721, -10086, -10317, -10503, -10696, -10933, -11265, -11675,
            -12051, -12278, -12323, -12257, -12216, -12295, -12481, -12712, -12973, -13284, -13612, -13893, -14143,
            -14431, -14769, -15107, -15391, -15588, -15704, -15741, -15708, -15643, -15573, -15538, -15568, -15622,
            -15636, -15565, -15403, -15146, -14811, -14469, -14194, -14006, -13845, -13641, -13395, -13114, -12809,
            -12505, -12202, -11889, -11572, -11259, -10895, -10431, -9912, -9405, -8947, -8546, -8208, -7901, -7579,
            -7235, -6843, -6395, -5966, -5635, -5417, -5256, -5088, -4931, -4860, -4927, -5121, -5353, -5515, -5554,
            -5489, -5335, -5070, -4712, -4330, -3976, -3660, -3343, -2969, -2547, -2159, -1888, -1747, -1697, -1665,
            -1627, -1623, -1660, -1745, -1890, -2047, -2150, -2188, -2230, -2318, -2399, -2413, -2354, -2284, -2285,
            -2392, -2581, -2804, -3024, -3181, -3242, -3247, -3261, -3324, -3456, -3664, -3864, -3940, -3923, -3928,
            -4015, -4194,
        ];

        let result = post.run(&x_s, false, 400);
        assert_eq!(result.ltpf_active, true);
        assert_eq!(result.pitch_present, true);
        assert_eq!(result.pitch_index, 478);
        assert_eq!(result.nbits_ltpf, 11);

        let x_s = [
            -4468, -4688, -4706, -4545, -4403, -4466, -4757, -5159, -5522, -5732, -5752, -5673, -5654, -5806, -6135,
            -6576, -6993, -7227, -7223, -7069, -6948, -6996, -7195, -7420, -7511, -7401, -7161, -6923, -6764, -6690,
            -6690, -6718, -6683, -6526, -6271, -5991, -5749, -5588, -5522, -5545, -5624, -5698, -5727, -5717, -5693,
            -5663, -5630, -5606, -5609, -5640, -5669, -5650, -5565, -5452, -5384, -5410, -5514, -5622, -5674, -5657,
            -5550, -5317, -5000, -4737, -4630, -4658, -4715, -4683, -4470, -4058, -3538, -3059, -2685, -2369, -2051,
            -1719, -1381, -1060, -780, -534, -331, -183, -48, 161, 495, 923, 1345, 1698, 2019, 2352, 2692, 3015, 3279,
            3418, 3450, 3482, 3589, 3781, 4042, 4324, 4614, 4999, 5529, 6120, 6668, 7128, 7500, 7824, 8147, 8467, 8739,
            8930, 9052, 9145, 9231, 9300, 9335, 9331, 9312, 9327, 9387, 9438, 9449, 9444, 9438, 9426, 9419, 9418, 9374,
            9247, 9071, 8905, 8754, 8608, 8477, 8336, 8115, 7777, 7362, 6951, 6631, 6400, 6158, 5830, 5413, 4884, 4249,
            3654, 3249, 3010, 2775, 2387, 1804, 1116, 473, -12, -300, -456, -632, -931, -1369, -1891, -2352, -2595,
            -2627, -2600, -2623, -2722, -2908, -3155, -3366, -3455, -3442, -3425, -3472, -3582, -3713, -3799, -3813,
            -3807, -3807, -3742, -3597, -3497, -3546, -3730, -3999, -4298, -4556, -4715, -4755, -4716, -4679, -4712,
            -4844, -5067, -5320, -5526, -5658, -5727, -5769, -5851, -6011, -6203, -6347, -6365, -6205, -5904, -5607,
            -5446, -5427, -5473, -5504, -5451, -5262, -4930, -4524, -4146, -3873, -3719, -3621, -3478, -3232, -2922,
            -2641, -2440, -2288, -2142, -1986, -1801, -1562, -1294, -1048, -840, -690, -623, -623, -689, -873, -1169,
            -1447, -1615, -1719, -1851, -2030, -2165, -2129, -1882, -1506, -1112, -763, -472, -240, -53, 126, 338, 577,
            797, 957, 1013, 941, 780, 588, 400, 251, 156, 93, 39, -30, -139, -256, -295, -237, -150, -97, -84, -73,
            -21, 76, 206, 369, 546, 699, 805, 866, 914, 1004, 1131, 1212, 1210, 1190, 1199, 1218, 1226, 1238, 1261,
            1251, 1153, 977, 778, 597, 481, 472, 533, 556, 479, 320, 135, -11, -54, 6, 88, 116, 79, -7, -131, -244,
            -256, -159, -47, 4, 7, 1, -11, -34, -63, -96, -123, -160, -273, -511, -824, -1117, -1344, -1485, -1548,
            -1591, -1664, -1750, -1842, -1967, -2110, -2223, -2284, -2286, -2236, -2168, -2115, -2084, -2055, -1972,
            -1792, -1551, -1323, -1165, -1119, -1156, -1156, -1028, -754, -341, 178, 719, 1216, 1641, 1984, 2292, 2671,
            3197, 3832, 4413, 4780, 4955, 5131, 5445, 5900, 6440, 6974, 7400, 7693, 7884, 7997, 8057, 8108, 8194, 8336,
            8538, 8757, 8938, 9081, 9263, 9573, 10047, 10640, 11263, 11841, 12334, 12725, 13041, 13347, 13691, 14048,
            14342, 14514, 14579, 14600, 14598, 14554, 14483, 14423, 14360, 14258, 14123, 14006, 13944, 13930, 13921,
            13908, 13913, 13883, 13721, 13440, 13145, 12880, 12636, 12416, 12208, 11966, 11645, 11222, 10728, 10281,
            9954, 9673, 9313, 8847, 8335, 7854, 7450, 7111, 6784, 6431, 6040, 5601, 5102, 4556, 4040, 3647, 3377, 3103,
            2714, 2267, 1921, 1759, 1739, 1754, 1703, 1563, 1420, 1352, 1315, 1250, 1185, 1157, 1148, 1126, 1087, 1045,
            1026, 1038, 1059, 1064, 1026, 917, 744, 563, 420, 319, 246, 174,
        ];

        let result = post.run(&x_s, false, 400);
        assert_eq!(result.ltpf_active, true);
        assert_eq!(result.pitch_present, true);
        assert_eq!(result.pitch_index, 478);
        assert_eq!(result.nbits_ltpf, 11);
    }
}
