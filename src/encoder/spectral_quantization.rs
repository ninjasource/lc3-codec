use crate::{
    common::{complex::Scaler, constants::MAX_LEN_SPECTRAL},
    encoder::spectral_noise_shaping::NBITS_SNS,
    tables::spectral_data_tables::*,
};

use itertools::Itertools;
#[allow(unused_imports)]
use num_traits::real::Real;

struct GlobalGainEstimationParams {
    nbits_offset: Scaler,
    nbits_spec_adj: usize,
    gg_off: i16,
}

struct BitConsumption {
    rate_flag: usize,
    lastnz: usize,
    nbits_lsb: usize,
    lastnz_trunc: usize,
    nbits_est: usize,
    nbits_trunc: usize,
    mode_flag: bool,
}

struct GlobalGainEstimation {
    gg_ind: i16,
    gg_min: i16,
    est_params: GlobalGainEstimationParams,
    reset_offset: bool,
}

struct QuantizeResult {
    lsb_mode: bool,
    bit_consumption: BitConsumption,
    gg: Scaler,
}

pub struct SpectralQuantizationResult {
    pub gg_ind: i16,
    pub nbits_spec: usize,
    pub nbits_lsb: usize,
    pub nbits_trunc: usize,
    pub lsb_mode: bool,
    pub rate_flag: usize,
    pub lastnz_trunc: usize,
    pub gg: f32,
}

pub struct SpectralQuantization {
    // constant
    ne: usize,
    fs_ind: usize,

    // state
    reset_offset_old: bool,
    nbits_offset_old: f32,
    nbits_spec_old: usize,
    nbits_est_old: usize,
}

impl SpectralQuantization {
    pub fn new(ne: usize, fs_ind: usize) -> Self {
        Self {
            fs_ind,
            ne,
            nbits_est_old: 0,
            nbits_offset_old: 0.0,
            nbits_spec_old: 0,
            reset_offset_old: false,
        }
    }

    pub fn run<'a>(
        &mut self,
        x_f: &[Scaler],
        x_q: &'a mut [i16],
        nbits: usize,
        nbits_bandwidth: usize,
        nbits_tns: usize,
        nbits_ltpf: usize,
    ) -> SpectralQuantizationResult {
        assert_eq!(x_f.len(), self.ne);
        assert_eq!(x_q.len(), self.ne);

        // bit budget
        let nbits_spec = self.calc_bit_budget(nbits, nbits_bandwidth, nbits_tns, nbits_ltpf);

        // first global gain estimation
        let mut gg = self.first_global_gain_estimation(x_f, nbits, nbits_spec);

        // quantization
        let mut quant = self.quantize_spectrum(x_f, x_q, nbits, gg.est_params.gg_off, gg.gg_ind, nbits_spec);

        // save state for next frame
        self.nbits_offset_old = gg.est_params.nbits_offset;
        self.nbits_est_old = quant.bit_consumption.nbits_est;
        self.reset_offset_old = gg.reset_offset;
        self.nbits_offset_old = gg.est_params.nbits_offset;

        // global gain adjustment
        let is_adjusted = self.global_gain_adjustment(&mut gg, nbits_spec, quant.bit_consumption.nbits_est);
        if is_adjusted {
            // requantize spectrum
            quant = self.quantize_spectrum(x_f, x_q, nbits, gg.est_params.gg_off, gg.gg_ind, nbits_spec);
        }

        SpectralQuantizationResult {
            //  output: x_q,
            gg_ind: gg.gg_ind,
            nbits_spec,
            nbits_lsb: quant.bit_consumption.nbits_lsb,
            lsb_mode: quant.lsb_mode,
            nbits_trunc: quant.bit_consumption.nbits_trunc,
            rate_flag: quant.bit_consumption.rate_flag,
            lastnz_trunc: quant.bit_consumption.lastnz_trunc,
            gg: quant.gg,
        }
    }

    fn calc_bit_budget(&self, nbits: usize, nbits_bandwidth: usize, nbits_tns: usize, nbits_ltpf: usize) -> usize {
        let nbits_ari = (self.ne as Scaler / 2.0).log2().ceil() as usize
            + if nbits <= 1280 {
                3
            } else if nbits <= 2560 {
                4
            } else {
                5
            };
        const NBITS_GAIN: usize = 8;
        const NBITS_NF: usize = 3;
        nbits - (nbits_bandwidth + nbits_tns + nbits_ltpf + NBITS_SNS + NBITS_GAIN + NBITS_NF + nbits_ari)
    }

    fn first_global_gain_estimation(&self, x_f: &[Scaler], nbits: usize, nbits_spec: usize) -> GlobalGainEstimation {
        let est_params = self.get_global_gain_estimation_parameter(nbits, nbits_spec);

        let mut e_buf = [0.0; MAX_LEN_SPECTRAL / 4];
        let e = &mut e_buf[..self.ne / 4];
        Self::compute_spectral_energy(e, x_f);

        let gg_ind = self.global_gain_estimation(e, &est_params);

        // finally, the quantized gain index shall be limited such that the quantized spectrum stays within the range [-32,768, 32,767]
        let (reset_offset, gg_min, gg_ind) = Self::global_gain_limitation(x_f, est_params.gg_off, gg_ind);

        GlobalGainEstimation {
            gg_ind,
            gg_min,
            est_params,
            reset_offset,
        }
    }

    fn get_global_gain_estimation_parameter(&self, nbits: usize, nbits_spec: usize) -> GlobalGainEstimationParams {
        let nbits_offset = if self.reset_offset_old {
            0.0
        } else {
            let prev_nbits = self.nbits_offset_old + self.nbits_spec_old as f32 - self.nbits_est_old as f32;
            0.8 * self.nbits_offset_old + 0.2 * (40.0).min((-40.0).max(prev_nbits))
        };

        let nbits_spec_adj = (nbits_spec as f32 + nbits_offset + 0.5) as u16 as usize;
        let gg_off = -(115.min(nbits as i16 / (10 * (self.fs_ind as i16 + 1)))) - 105 - 5 * (self.fs_ind as i16 + 1);

        GlobalGainEstimationParams {
            nbits_offset,
            nbits_spec_adj,
            gg_off,
        }
    }

    fn global_gain_estimation(&self, e: &[Scaler], gg_est_params: &GlobalGainEstimationParams) -> i16 {
        let mut fac: i16 = 256;
        let mut gg_ind = 255;

        for _ in 0..8 {
            fac >>= 1;
            gg_ind -= fac;
            let mut tmp = 0.0;
            let mut is_zero = true;

            for e_item in e.iter().rev() {
                if *e_item * 28.0 / 20.0 < (gg_ind as Scaler + gg_est_params.gg_off as Scaler) {
                    if !is_zero {
                        tmp += 2.7 * 28.0 / 20.0;
                    }
                } else {
                    if (gg_ind as Scaler + gg_est_params.gg_off as Scaler)
                        < (*e_item * 28.0 / 20.0 - 43.0 * 28.0 / 20.0)
                    {
                        tmp += 2.0 * *e_item * 28.0 / 20.0
                            - 2.0 * (gg_ind as Scaler + gg_est_params.gg_off as Scaler)
                            - 36.0 * 28.0 / 20.0;
                    } else {
                        tmp += *e_item * 28.0 / 20.0 - (gg_ind as Scaler + gg_est_params.gg_off as Scaler)
                            + 7.0 * 28.0 / 20.0;
                    }
                    is_zero = false;
                }
            }
            if (tmp > gg_est_params.nbits_spec_adj as Scaler * 1.4 * 28.0 / 20.0) && !is_zero {
                gg_ind += fac;
            }
        }

        gg_ind
    }

    // returns (reset_offset, gg_min, gg_ind)
    fn global_gain_limitation(x_f: &[Scaler], gg_off: i16, gg_ind: i16) -> (bool, i16, i16) {
        let mut x_f_max = 0.0;
        for x in x_f {
            x_f_max = x_f_max.max((*x).abs());
        }
        let gg_min = if x_f_max > 0.0 {
            (28.0 * (x_f_max / (32768.0 - 0.375)).log10()).ceil() as i16 - gg_off
        } else {
            0
        };

        if gg_ind < gg_min || x_f_max == 0.0 {
            (true, gg_min, gg_min)
        } else {
            (false, gg_min, gg_ind)
        }
    }

    fn quantize_spectrum(
        &self,
        x_f: &[Scaler],
        x_q: &mut [i16],
        nbits: usize,
        gg_off: i16,
        gg_ind: i16,
        nbits_spec: usize,
    ) -> QuantizeResult {
        let gg = (10.0).powf((gg_ind as Scaler + gg_off as Scaler) / 28.0) as f32;
        for (output, input) in x_q.iter_mut().zip(x_f) {
            *output = if *input >= 0.0 {
                (*input / gg + 0.375) as i16
            } else {
                (*input / gg - 0.375) as i16
            };
        }

        // bit consumption
        let bit_consumption = self.compute_bit_consumption(x_q, nbits, nbits_spec);

        // truncation
        for item in x_q[bit_consumption.lastnz_trunc..bit_consumption.lastnz].iter_mut() {
            *item = 0;
        }

        let lsb_mode = bit_consumption.mode_flag && bit_consumption.nbits_est > nbits_spec;

        QuantizeResult {
            lsb_mode,
            bit_consumption,
            gg,
        }
    }

    fn compute_bit_consumption(&self, x_q: &mut [i16], nbits: usize, nbits_spec: usize) -> BitConsumption {
        let rate_flag = if nbits > (160 + self.fs_ind * 160) { 512 } else { 0 };

        let mode_flag = nbits >= (480 + self.fs_ind * 160);

        let mut lastnz = self.ne;
        while lastnz > 2 && x_q[lastnz - 1] == 0 && x_q[lastnz - 2] == 0 {
            lastnz -= 2;
        }

        let mut nbits_est_local: u32 = 0;
        let mut nbits_trunc_local: u32 = 0;
        let mut nbits_lsb = 0;
        let mut lastnz_trunc = 2;
        let mut c = 0;

        for n in (0..lastnz).step_by(2) {
            let mut t = c + rate_flag;
            if n > self.ne / 2 {
                t += 256;
            }

            let mut a = x_q[n].unsigned_abs();
            let mut a_lsb = a;
            let mut b = x_q[n + 1].unsigned_abs();
            let mut b_lsb = b;
            let mut lev = 0;
            while a.max(b) >= 4 {
                let pki = AC_SPEC_LOOKUP[t + lev * 1024];
                nbits_est_local += AC_SPEC_BITS[pki as usize][16] as u32;
                if lev == 0 && mode_flag {
                    nbits_lsb += 2;
                } else {
                    nbits_est_local += 2 * 2048;
                }
                a >>= 1;
                b >>= 1;
                lev = 3.min(lev + 1);
            }
            let pki = AC_SPEC_LOOKUP[t + lev * 1024];
            let sym = a + 4 * b;
            nbits_est_local += AC_SPEC_BITS[pki as usize][sym as usize] as u32;

            // alternative implementation (more clear, more performant?)
            if a_lsb > 0 {
                nbits_est_local += 2048;
            }
            if b_lsb > 0 {
                nbits_est_local += 2048;
            }
            if lev > 0 && mode_flag {
                a_lsb >>= 1;
                b_lsb >>= 1;
                if a_lsb == 0 && x_q[n] != 0 {
                    nbits_lsb += 1;
                }
                if b_lsb == 0 && x_q[n + 1] != 0 {
                    nbits_lsb += 1;
                }
            }

            if (x_q[n] != 0 || x_q[n + 1] != 0) && (nbits_est_local as Scaler / 2048.0).ceil() as usize <= nbits_spec {
                lastnz_trunc = n + 2;
                nbits_trunc_local = nbits_est_local;
            }
            t = if lev <= 1 {
                1 + (a + b) as usize * (lev + 1)
            } else {
                12 + lev
            };
            c = (c & 15) * 16 + t;
        }
        let nbits_est = (nbits_est_local as Scaler / 2048.0).ceil() as usize + nbits_lsb;
        let nbits_trunc = (nbits_trunc_local as Scaler / 2048.0).ceil() as usize;
        BitConsumption {
            lastnz,
            lastnz_trunc,
            nbits_est,
            mode_flag,
            nbits_lsb,
            nbits_trunc,
            rate_flag,
        }
    }

    fn global_gain_adjustment(&self, gg: &mut GlobalGainEstimation, nbits_spec: usize, nbits_est: usize) -> bool {
        const T1: [usize; 5] = [80, 230, 380, 530, 680];
        const T2: [usize; 5] = [500, 1025, 1550, 2075, 2600];
        const T3: [usize; 5] = [850, 1700, 2550, 3400, 4250];

        let t1 = T1[self.fs_ind];
        let t2 = T2[self.fs_ind];
        let t3 = T3[self.fs_ind];

        let delta = if nbits_est < t1 {
            (nbits_est as Scaler + 48.0) / 16.0
        } else if nbits_est < t2 {
            let tmp1 = t1 as Scaler / 16.0 + 3.0;
            let tmp2 = t2 as Scaler / 48.0;
            (nbits_est as Scaler - t1 as Scaler) * (tmp2 - tmp1) / (t2 as Scaler - t1 as Scaler) + tmp1
        } else if nbits_est < t3 {
            nbits_est as Scaler / 48.0
        } else {
            t3 as Scaler / 48.0
        };
        let delta = (delta + 0.5).floor();
        let delta2 = delta + 2.0;

        let gg_ind_origin = gg.gg_ind;
        if gg.gg_ind < 255 && nbits_est > nbits_spec
            || gg.gg_ind > 0 && (nbits_est as Scaler) < (nbits_spec as Scaler - delta2)
        {
            if (nbits_est as Scaler) < (nbits_spec as Scaler - delta2) {
                gg.gg_ind -= 1;
            } else if gg.gg_ind == 254 || (nbits_est as Scaler) < (nbits_spec as Scaler + delta) {
                gg.gg_ind += 1;
            } else {
                gg.gg_ind += 2;
            }
            gg.gg_ind = gg.gg_ind.max(gg.gg_min);
        }

        gg_ind_origin != gg.gg_ind
    }

    fn compute_spectral_energy(e: &mut [Scaler], x_f: &[Scaler]) {
        for (e, (x0, x1, x2, x3)) in e.iter_mut().zip(x_f.iter().tuples()) {
            let total = *x0 * *x0 + *x1 * *x1 + *x2 * *x2 + *x3 * *x3;
            *e = 10.0 * (Scaler::EPSILON + total).log10();
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn spectral_quantization_run() {
        let mut quant = SpectralQuantization::new(400, 4);
        #[rustfmt::skip]
        let x_f = [
            2511.287, -3606.8093, -453.28122, -360.71924, -2574.9756, -3166.2068, 6525.6, 6284.0137, -10303.951,
            -4442.8755, 2318.4038, -691.36865, 509.12134, -938.29266, 525.9005, -1542.4482, -3091.5273, 262.27206,
            -379.65656, -98.486786, 668.16437, -213.70082, -114.14174, 80.94827, 83.418945, -16.794994, 6.197174,
            -332.66284, -48.91777, 0.084522665, -281.5821, 78.69863, 133.78897, 102.63185, -259.9021, -450.69684,
            -538.17615, 102.73462, 256.75674, -122.24531, 414.3129, 90.33753, -141.62321, 64.58323, 158.11197,
            264.04486, -70.996445, -74.847595, 715.87695, 398.03842, -383.14874, 81.798744, 100.61626, 183.76709,
            107.91918, -74.72749, 137.34314, -12.11866, 145.58879, -8.319928, -146.90816, 21.44834, -14.692991,
            -26.504921, -44.41256, 169.60251, 157.3852, 57.295708, -223.72159, -105.66721, 85.917145, -124.74416,
            67.12806, 148.09708, 108.51118, -38.312347, 85.37175, 124.25812, -19.708088, -252.26158, -396.83145,
            30.49828, 24.158318, 4.4514937, -100.692856, -19.743237, 174.73466, -129.2712, -94.76638, -48.278133,
            -12.916355, -108.09598, 15.809202, 102.17793, -36.756264, -59.4923, -168.1815, 18.80262, -43.62355,
            23.15091, 16.152151, -187.13437, 30.643784, 90.79638, -16.21843, -39.05421, 62.11714, -2.155285, -19.06319,
            -42.414265, 35.171066, 84.379585, -116.602325, -33.51855, -7.659307, 44.801598, -11.07798, -14.886959,
            62.95822, 14.218942, 31.104227, -83.76815, 17.044928, 55.428146, -13.889921, -126.15556, -27.248165,
            120.640366, -19.761774, 14.046745, -73.71858, 0.10803318, 10.083524, -16.70562, 4.7014112, -62.691025,
            50.462833, -84.346176, -13.629369, 36.10631, -46.619267, -21.598942, -48.490704, -6.3323555, 15.65649,
            48.896317, -56.442543, -6.640568, -17.761715, -70.73248, -40.4366, -36.27153, 74.86966, -46.19746,
            -44.013317, 2.0889094, 26.809942, -6.282827, -65.310585, 33.484646, -9.667151, 15.079562, -16.305983,
            9.046168, 29.939863, 5.129214, 16.5475, -50.21096, 31.017387, 38.323303, -3.0681198, -36.80033, -1.2009478,
            11.490404, 26.555166, 44.298775, -31.610435, 4.04779, -32.927948, 29.660675, 11.896004, -18.109625,
            25.200207, -23.730797, 15.500507, -25.358208, 6.163124, -7.3798866, -7.921206, -16.156162, -36.149162,
            -1.5045524, -43.494595, 29.947332, -15.511134, -17.982704, -28.039505, -30.445019, -3.5225277, -46.805386,
            -1.0227482, -23.363768, -17.676548, -3.6520846, -12.88875, 4.756609, 31.271141, -6.629322, -37.832882,
            -0.21950912, 38.885174, 21.138603, 6.2617035, -9.60021, -11.284341, -19.242826, -1.6980145, -9.359415,
            -41.126484, -29.448069, -5.9372683, 25.94433, 1.3254867, -14.705631, 15.379487, 16.574158, -27.095804,
            -19.93113, 20.497425, 0.79107094, -7.7554317, -36.76988, -52.092567, -38.030884, -20.167278, -0.28380156,
            -7.2491307, 8.2373905, 7.9681587, 12.655432, 23.266579, 13.15513, 5.5212536, 12.821712, 10.402097, -1.8669,
            17.029139, 7.877034, -3.6022367, -13.437171, -10.018736, 6.7936516, 12.0322485, 17.809977, -13.797862,
            -19.886257, -20.896944, -13.391824, -3.7870193, -6.042081, 9.495218, -8.35246, -16.302475, -16.089418,
            -2.2239032, 5.133191, -1.9499176, -12.571083, -26.08479, -8.472019, -4.010655, 5.987412, -1.6527638,
            -5.525652, -1.8339038, -6.3098893, -2.546278, -14.999996, -4.8673024, -21.751396, -23.044847, 0.98999345,
            3.2206738, 1.6813838, -5.9552116, -2.9898171, -6.6439576, 10.739187, -0.41604716, -6.192164, 10.029629,
            -8.77803, -3.0170813, -3.2569091, -16.77877, -12.368547, 2.8054588, -1.341449, 4.0487995, 9.832774,
            9.444449, 19.458578, -5.653375, -5.7151184, -1.1136432, 2.3793151, 5.6741295, -4.6841993, -0.53913784,
            -5.0413313, -4.085096, -17.153347, -1.8630176, 11.785563, 1.1223254, -0.71278524, -4.503395, 1.1829545,
            -10.829809, -1.1153163, 3.7169714, -0.4367964, -0.591923, -2.8435392, 1.75674, -4.178983, 4.8031635,
            -5.574907, -0.26360065, -4.317787, -11.108944, 12.20769, 3.0794377, -0.6653844, -2.5413795, -2.403656,
            -0.8925853, 13.288289, 6.883606, -15.778875, -1.6005001, 5.696593, 8.6179, -7.1294184, -5.413874,
            5.2202096, 2.9486847, -2.3101602, -2.2582812, 15.61281, -0.8072282, -0.08025418, -4.076133, -9.068343,
            4.1847715, -2.2272792, -1.0373013, -4.7336984, 1.2461259, -0.5397187, 0.18704754, -0.1272373, 0.41515422,
            4.11998, -5.550728, 0.5320622, 3.4917614, 8.215734, -10.053921, -7.784162, 3.4075887, -8.748142,
            0.87312615, -1.6497533, 1.9920977, -0.20973617, -1.9895163, -3.3932712, 1.6009337, 7.0630236, -3.209106,
            5.2722173, -5.2088814, -1.2551317, 2.55086, -3.7561512, -1.9274446, -2.2996173, 5.709345, -2.3815656,
            -0.94840765, 0.60572165, 4.09223, -2.2723556, -4.731332, 2.7107785, -2.495332, 3.3961606, -2.640641,
            -1.3273795,
        ];
        let mut x_q = [0; 400];

        let result = quant.run(&x_f, &mut x_q, 1200, 3, 42, 11);

        let x_q_expected = [
            102, -146, -18, -14, -104, -128, 264, 254, -417, -180, 94, -28, 20, -38, 21, -62, -125, 10, -15, -4, 27,
            -9, -4, 3, 3, -1, 0, -13, -2, 0, -11, 3, 5, 4, -10, -18, -22, 4, 10, -5, 17, 4, -6, 2, 6, 11, -3, -3, 29,
            16, -15, 3, 4, 7, 4, -3, 5, 0, 6, 0, -6, 1, 0, -1, -2, 7, 6, 2, -9, -4, 3, -5, 3, 6, 4, -1, 3, 5, -1, -10,
            -16, 1, 1, 0, -4, -1, 7, -5, -4, -2, 0, -4, 1, 4, -1, -2, -7, 1, -2, 1, 1, -7, 1, 4, -1, -1, 2, 0, -1, -2,
            1, 3, -5, -1, 0, 2, 0, 0, 2, 0, 1, -3, 1, 2, 0, -5, -1, 5, -1, 0, -3, 0, 0, -1, 0, -2, 2, -3, 0, 1, -2, -1,
            -2, 0, 1, 2, -2, 0, -1, -3, -2, -1, 3, -2, -2, 0, 1, 0, -3, 1, 0, 0, -1, 0, 1, 0, 1, -2, 1, 1, 0, -1, 0, 0,
            1, 2, -1, 0, -1, 1, 0, -1, 1, -1, 1, -1, 0, 0, 0, -1, -1, 0, -2, 1, -1, -1, -1, -1, 0, -2, 0, -1, -1, 0, 0,
            0, 1, 0, -1, 0, 1, 1, 0, 0, 0, -1, 0, 0, -2, -1, 0, 1, 0, 0, 0, 1, -1, -1, 1, 0, 0, -1, -2, -1, -1, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, -1, -1, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(x_q, x_q_expected);
        assert_eq!(result.gg, 24.7091141);
        assert_eq!(result.lastnz_trunc, 350);
        assert_eq!(result.lsb_mode, false);
        assert_eq!(result.gg_ind, 193);
        assert_eq!(result.rate_flag, 512);
        assert_eq!(result.nbits_lsb, 107);
    }
}
