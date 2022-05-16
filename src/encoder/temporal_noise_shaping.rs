#![allow(clippy::excessive_precision)]

use crate::{
    common::{
        complex::Scaler,
        config::{FrameDuration, Lc3Config},
    },
    tables::temporal_noise_shaping_tables::*,
};
use core::{f64::consts::PI, panic};
#[allow(unused_imports)]
use num_traits::real::Real;

pub struct TemporalNoiseShaping {
    config: Lc3Config,
}

pub struct TnsResult {
    pub nbits_tns: usize,
    pub lpc_weighting: u8,
    pub num_tns_filters: usize,
    pub rc_order: [usize; 2], // The order of quantized reflection coefficients
    pub rc_i: [usize; 16],    // integer reflection coefficients
    pub rc_q: [Scaler; 16],   // quantized reflection coefficients
}

struct TnsParams {
    pub num_tns_filters: usize,
    pub start_freq: [usize; 2],
    pub stop_freq: [usize; 2],
    pub sub_start: [[usize; 3]; 2],
    pub sub_stop: [[usize; 3]; 2],
}

impl TemporalNoiseShaping {
    pub fn new(config: Lc3Config) -> Self {
        Self { config }
    }

    pub fn run(&self, x_s: &mut [Scaler], p_bw: usize, nbits: usize, near_nyquist_flag: bool) -> TnsResult {
        assert_eq!(x_s.len(), self.config.ne);
        let tns = self.get_tns_params(p_bw);
        let mut rc_order = [0; 2];
        let mut rc_i = [0; 16];
        let mut rc_q = [0.0; 16];
        let num_tns_filters = tns.num_tns_filters;

        // tns analysis
        let lpc_weighting = match self.config.n_ms {
            FrameDuration::TenMs if nbits < 480 => 1,
            FrameDuration::SevenPointFiveMs if nbits < 360 => 1,
            _ => 0,
        };

        for (f, (sub_start, sub_stop)) in tns.sub_start[..num_tns_filters].iter().zip(&tns.sub_stop).enumerate() {
            let r = Self::compute_normalized_autocorrelation(sub_start, sub_stop, x_s);
            Self::tns_analysis(&r, f, near_nyquist_flag, lpc_weighting, &mut rc_q);
        }

        // quantization
        Self::apply_quantization(num_tns_filters, &mut rc_q, &mut rc_i, &mut rc_order);

        // bit budget
        let nbits_tns = Self::calc_bit_budget(num_tns_filters, lpc_weighting, &rc_i, &rc_order);

        // filtering
        Self::apply_filtering(&tns, x_s, &mut rc_q, &mut rc_order);

        TnsResult {
            //output: x_s,
            nbits_tns,
            num_tns_filters,
            rc_i,
            rc_q,
            rc_order,
            lpc_weighting,
        }
    }

    fn compute_normalized_autocorrelation(sub_start: &[usize], sub_stop: &[usize], x_s: &[Scaler]) -> [Scaler; 9] {
        let lag_window = [
            1.0,
            0.9980280260203829,
            0.9921354055113971,
            0.9823915844707989,
            0.9689107911912967,
            0.9518498073692735,
            0.9314049334023056,
            0.9078082299969592,
            0.8813231366694713,
        ];

        let mut r = [0.0; 9];
        for (k, (r, lag_window)) in r.iter_mut().zip(&lag_window).enumerate() {
            let r0 = if k == 0 { 3.0 } else { 0.0 };
            let mut rk = 0.0;
            let mut e_prod = 1.0;

            for (start, stop) in sub_start.iter().zip(sub_stop.iter()) {
                let es: Scaler = x_s[*start..*stop].iter().map(|x| *x * *x).sum();

                // ac = sum(x[n] * x[n+k]) for n = start..stop
                let k_from = *start + k;
                let ac: Scaler = if k_from < x_s.len() && k_from < *stop {
                    x_s[k_from..*stop]
                        .iter()
                        .zip(x_s[*start..*stop].iter())
                        .map(|(kx, x)| *x * *kx)
                        .sum()
                } else {
                    0.0
                };

                e_prod *= es;
                rk += ac / es;
            }

            *r = if e_prod == 0.0 { r0 } else { rk } * *lag_window;
        }

        r
    }

    fn get_tns_params(&self, p_bw: usize) -> TnsParams {
        match self.config.n_ms {
            FrameDuration::TenMs => match p_bw {
                0 => TnsParams {
                    num_tns_filters: 1,
                    start_freq: [12, 160],
                    stop_freq: [80, 0],
                    sub_start: [[12, 34, 57], [0, 0, 0]],
                    sub_stop: [[34, 57, 80], [0, 0, 0]],
                },
                1 => TnsParams {
                    num_tns_filters: 1,
                    start_freq: [12, 160],
                    stop_freq: [160, 0],
                    sub_start: [[12, 61, 110], [0, 0, 0]],
                    sub_stop: [[61, 110, 160], [0, 0, 0]],
                },
                2 => TnsParams {
                    num_tns_filters: 1,
                    start_freq: [12, 160],
                    stop_freq: [200, 0],
                    sub_start: [[12, 88, 164], [0, 0, 0]],
                    sub_stop: [[88, 164, 240], [0, 0, 0]],
                },
                3 => TnsParams {
                    num_tns_filters: 2,
                    start_freq: [12, 160],
                    stop_freq: [160, 320],
                    sub_start: [[12, 61, 110], [160, 213, 266]],
                    sub_stop: [[61, 110, 160], [213, 266, 320]],
                },
                4 => TnsParams {
                    num_tns_filters: 2,
                    start_freq: [12, 200],
                    stop_freq: [200, 400],
                    sub_start: [[12, 74, 137], [200, 266, 333]],
                    sub_stop: [[74, 137, 200], [266, 333, 400]],
                },
                _ => panic!(
                    "Cannot create valid start and stop freq indexes because of an invalid p_bw: {}",
                    p_bw
                ),
            },
            FrameDuration::SevenPointFiveMs => match p_bw {
                0 => TnsParams {
                    num_tns_filters: 1,
                    start_freq: [9, 120],
                    stop_freq: [60, 0],
                    sub_start: [[9, 26, 43], [0, 0, 0]],
                    sub_stop: [[26, 43, 60], [0, 0, 0]],
                },
                1 => TnsParams {
                    num_tns_filters: 1,
                    start_freq: [9, 120],
                    stop_freq: [120, 0],
                    sub_start: [[9, 46, 83], [0, 0, 0]],
                    sub_stop: [[46, 83, 120], [0, 0, 0]],
                },
                2 => TnsParams {
                    num_tns_filters: 1,
                    start_freq: [9, 120],
                    stop_freq: [180, 0],
                    sub_start: [[9, 66, 123], [0, 0, 0]],
                    sub_stop: [[66, 123, 180], [0, 0, 0]],
                },
                3 => TnsParams {
                    num_tns_filters: 2,
                    start_freq: [9, 120],
                    stop_freq: [120, 240],
                    sub_start: [[9, 46, 82], [120, 159, 200]],
                    sub_stop: [[46, 82, 120], [159, 200, 240]],
                },
                4 => TnsParams {
                    num_tns_filters: 2,
                    start_freq: [9, 150],
                    stop_freq: [150, 300],
                    sub_start: [[9, 56, 103], [150, 200, 250]],
                    sub_stop: [[56, 103, 150], [200, 250, 300]],
                },
                _ => panic!(
                    "Cannot create valid start and stop freq indexes because of an invalid p_bw: {}",
                    p_bw
                ),
            },
        }
    }

    fn tns_analysis(r: &[Scaler], f: usize, near_nyquist_flag: bool, lpc_weighting: u8, rc_q: &mut [Scaler]) {
        // use Levinson-Durbin recursion to obtain  LPC (Linear Predictive Coding) coefficients and a predictor error
        let mut a_memory = [[0.0; 9]; 2];
        let (a, a_last) = a_memory.split_at_mut(1);
        let mut a = &mut a[0];
        let mut a_last = &mut a_last[0];

        let mut e = r[0];
        a[0] = 1.0;
        for k in 1..9 {
            core::mem::swap(&mut a_last, &mut a);

            // calculate reflection coefficients
            let mut rc = 0.0;
            for n in 0..k {
                rc -= a_last[n] * r[k - n];
            }
            if e != 0.0 {
                rc /= e;
            }

            // calculate error
            a[0] = 1.0;
            for n in 1..k {
                a[n] = a_last[n] + rc * a_last[k - n];
            }
            a[k] = rc;
            e *= 1.0 - rc * rc;
        }

        let pred_gain = if e == 0.0 { r[0] } else { r[0] / e };
        const THRESH: Scaler = 1.5;
        if pred_gain > THRESH && !near_nyquist_flag {
            // turn on the tns filter
            let mut gamma = 1.0;
            const THRESH2: Scaler = 2.0;
            if lpc_weighting > 0 && pred_gain < THRESH2 {
                gamma -= (1.0 - 0.85) * (THRESH2 - pred_gain) / (THRESH2 - THRESH);
            }
            for (k, a_k) in a.iter_mut().enumerate() {
                *a_k *= gamma.powi(k as i32);
            }
            let rc = &mut rc_q[f * 8..];
            let mut a_k = a;
            let mut a_km1 = a_last;
            for k in (1..9).rev() {
                rc[k - 1] = a_k[k];
                let e = 1.0 - rc[k - 1] * rc[k - 1];
                for n in 1..k {
                    a_km1[n] = a_k[n] - rc[k - 1] * a_k[k - n];
                    a_km1[n] /= e;
                }
                core::mem::swap(&mut a_k, &mut a_km1);
            }
        } else {
            // turn off the tns filter
            let rc = &mut rc_q[f * 8..];
            for rc_n in rc[..8].iter_mut() {
                *rc_n = 0.0;
            }
        }
    }

    fn apply_quantization(num_tns_filters: usize, rc_q: &mut [Scaler], rc_i: &mut [usize], rc_order: &mut [usize]) {
        const QUANTIZER_STEPSIZE: Scaler = PI as Scaler / 17.0;
        for f in 0..num_tns_filters {
            let rc = &mut rc_q[f * 8..];
            for (rc_quant, rc_int) in rc[..8].iter_mut().zip(rc_i[(f * 8)..(f * 8 + 8)].iter_mut()) {
                *rc_int = (to_int((*rc_quant).asin() / QUANTIZER_STEPSIZE) + 8) as usize;
                *rc_quant = (QUANTIZER_STEPSIZE * (*rc_int as Scaler - 8.0)).sin();
            }

            // calculate the order of the quantized reflection coefficients
            let mut k: i8 = 7;
            while k >= 0 && rc_i[f * 8 + k as usize] == 8 {
                k -= 1;
            }
            rc_order[f] = (k + 1) as usize;
        }

        // TODO: double check against spec
        for f in num_tns_filters..2 {
            for k in 0..8 {
                rc_i[f * 8 + k] = 8;
                rc_q[f * 8 + k] = 0.0;
            }
            rc_order[f] = 0;
        }
    }

    fn calc_bit_budget(num_tns_filters: usize, tns_lpc_weighting: u8, rc_i: &[usize], rc_order: &[usize]) -> usize {
        let mut nbits_tns = 0;
        for f in 0..num_tns_filters {
            let nbits_tns_order = if rc_order[f] != 0 {
                AC_TNS_ORDER_BITS[tns_lpc_weighting as usize][rc_order[f] - 1]
            } else {
                0
            };
            let mut nbits_tns_coef = 0;
            for (k, coef_bits) in AC_TNS_COEF_BITS[..rc_order[f]].iter().enumerate() {
                nbits_tns_coef += coef_bits[rc_i[f * 8 + k]];
            }

            nbits_tns += ((2048.0 + nbits_tns_order as Scaler + nbits_tns_coef as Scaler) / 2048.0).ceil() as usize;
        }

        nbits_tns
    }

    fn apply_filtering(tns: &TnsParams, x_s: &mut [Scaler], rc_q: &mut [Scaler], rc_order: &mut [usize]) {
        let mut st = [0.0; 8];
        for f in 0..tns.num_tns_filters {
            if rc_order[f] != 0 {
                let from = tns.start_freq[f];
                let to = tns.stop_freq[f];

                // TODO: check this assertion below
                // we can safely modify x_s in place because ther is no overlap with from and to
                for x_f in &mut x_s[from..to] {
                    let mut t = *x_f;
                    let mut st_save = t;
                    let prev_order = rc_order[f] - 1;

                    for (st, rcq) in st[..prev_order].iter_mut().zip(&rc_q[f * 8..f * 8 + prev_order]) {
                        let st_tmp = *rcq * t + *st;
                        t += *rcq * *st;
                        *st = st_save;
                        st_save = st_tmp;
                    }

                    t += rc_q[f * 8 + prev_order] * st[prev_order];
                    st[prev_order] = st_save;
                    *x_f = t;
                }
            }
        }
    }
}

fn to_int(x: Scaler) -> i8 {
    if x >= 0.0 {
        (x + 0.5) as i8
    } else {
        -(-x + 0.5) as i8
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::common::config::SamplingFrequency;

    #[rustfmt::skip]
    #[test]
    fn temporal_noise_shaping_run() {
        let config = Lc3Config::new(SamplingFrequency::Hz48000, FrameDuration::TenMs, 1);
        let tns = TemporalNoiseShaping::new(config);
        let mut x_s = [2511.287, -3606.8093, -453.28122, -360.71924, -2574.9756, -3166.2068, 6525.6, 6284.0137, -10303.951, -4442.8755, 2318.4038, -691.36865, 509.12134, -1054.0342, 852.0186, -1959.2595, -2463.084, 583.8584, -1045.9277, 886.3083, 94.29779, 106.0552, 262.7902, 195.86151, 748.0028, -369.72824, 326.68167, -706.81616, 83.90518, -101.80473, -425.53433, 248.85109, -246.2868, 415.67813, -428.0742, -195.10165, -497.12534, 148.11542, 351.7376, -259.38834, 639.9149, -232.2877, 182.4239, 30.73989, 139.50827, 315.7314, -314.8985, 111.07428, 464.2896, 352.44342, -380.097, 82.403694, -41.359577, 195.84523, 114.65242, -207.70195, 156.48032, -178.95581, 313.9437, -130.90295, -105.89414, 22.376598, -114.98715, 120.20845, -164.51794, 261.76236, 37.331238, 106.96828, -198.03798, -108.76448, 134.22601, -216.93547, 195.3998, -21.63419, 156.55684, -51.93392, 101.626854, 149.68706, -94.93349, -155.7282, -464.9618, 119.154236, -82.80184, 72.36793, -122.200264, -39.244514, 270.96143, -173.43079, 107.70619, -164.70389, 55.223305, -111.1405, 17.96329, 125.23528, -135.59274, 60.439613, -251.46301, 131.2461, -101.01771, 69.59965, 16.21791, -239.07864, 166.54358, -27.770905, 100.91532, -84.49127, 69.28591, 2.6135557, -31.626968, 20.677122, -36.27791, 119.84301, -173.36378, 35.84611, -57.688297, 60.183567, 4.423961, -44.209984, 105.70867, -50.634827, 109.67776, -134.32838, 61.955124, 21.708387, -28.346958, -83.66344, -70.2042, 154.10416, -77.64532, 88.29541, -130.808, 36.89496, 16.645363, -25.276045, 51.67908, -137.08151, 124.83691, -152.0996, 65.79087, 0.93463665, -60.837196, 39.439693, -114.28214, 81.32954, -45.443783, 96.88175, -80.02808, 6.91597, -0.9586373, -92.0962, 23.475348, -94.339935, 125.18178, -98.72728, 9.358742, -2.886684, 14.301312, 43.602825, -103.58884, 91.13613, -73.82291, 71.59755, -40.91015, 9.037108, 43.62777, -32.685516, 69.14355, -105.05149, 82.54336, -8.614124, 11.804503, -27.901173, -29.564074, 41.35118, -8.069561, 77.09149, -70.76394, 31.398144, -49.828518, 43.814983, 11.879899, -39.018723, 49.713158, -68.79033, 65.760086, -60.97088, 33.07043, -20.903938, -14.867276, 9.887002, -65.46376, 41.143112, -80.06619, 70.94139, -43.88703, 2.1660235, -17.16438, -44.083027, 38.160618, -81.36653, 70.78228, -76.33006, 38.737568, -12.454603, -29.338528, 56.428005, -39.599926, 39.25371, -73.31344, 69.79237, -17.97577, 12.852926, 12.684623, -32.4752, 36.446888, -65.407234, 61.598774, -65.757484, 10.238938, -20.029032, -2.825231, 42.594162, -56.822716, 53.156845, -30.462795, 38.866444, -56.68353, 24.660025, 14.751921, -35.350494, 37.21626, -76.86587, 29.546429, -62.394695, 27.275826, -14.183131, -10.919811, 38.614643, -38.49674, 59.394817, -34.20599, 41.272846, -23.526806, 22.555021, 1.7496673, -19.42429, 46.749092, -46.406094, 42.399597, -50.664745, 32.1371, -13.392963, 7.2786326, 24.364573, -48.878193, 39.617836, -60.437885, 40.904007, -37.622654, 16.281404, 10.157282, -32.828197, 30.448824, -50.824665, 49.72176, -41.46506, 27.87757, -24.912022, -13.79381, 20.7759, -37.632587, 48.469193, -50.989594, 42.78355, -32.603626, 12.9430275, -0.48920912, -29.010874, 36.240032, -65.120476, 43.169476, -36.05779, 27.190653, -12.357571, -6.6507573, 20.675322, -37.217205, 53.199875, -54.718746, 43.378906, -21.585281, -1.6675593, 13.522394, -29.79583, 22.414072, -39.135113, 40.94145, -37.682613, 31.857054, -8.821981, 6.641381, 25.134476, -40.58887, 42.64201, -44.198082, 40.230164, -27.744131, 10.078336, 4.423329, -22.164833, 27.893787, -50.982807, 50.552082, -34.822014, 24.829428, -12.062506, -3.7417355, 18.681189, -40.84422, 45.581814, -42.326923, 35.923992, -27.111525, 13.07634, 2.8001645, -20.283352, 35.183113, -46.604267, 47.304718, -47.010338, 26.623617, -1.2559637, -11.157539, 23.359474, -34.244343, 38.091057, -39.234665, 46.234547, -30.573185, 2.6562088, 11.293563, -17.823277, 33.29007, -46.73799, 43.705185, -32.237366, 23.87085, -14.893113, 0.8721875, 27.640835, -39.741753, 45.260857, -47.98558, 33.93777, -20.93668, 4.6949544, 7.7584434, -24.253351, 35.06494, -40.09654, 39.286926, -33.447765, 24.822977, -8.966054, -9.330206, 21.831121, -26.851707, 38.898933, -49.516945, 38.540756, -25.502691, 3.2083294, 10.038194, -23.048763, 34.32631, -39.45431, 37.582066, -34.85849, 27.25857, -8.987542, -7.2581387, 24.746119, -38.36987, 41.85204, -39.183273, 29.537968, -22.075731, 8.153215, 10.045177, -24.236364, 31.76943, -34.252445, 37.48698, -35.33923, 22.365273, -7.9304223, -7.034074, 20.73084, -31.455378, 35.02656];

        let result = tns.run(&mut x_s, 4, 1200, false);

        let x_f_expected = [2511.287, -3606.8093, -453.28122, -360.71924, -2574.9756, -3166.2068, 6525.6, 6284.0137, -10303.951, -4442.8755, 2318.4038, -691.36865, 509.12134, -938.29266, 525.9005, -1542.4482, -3091.5273, 262.27206, -379.65656, -98.486786, 668.16437, -213.70082, -114.14174, 80.94827, 83.418945, -16.794994, 6.197174, -332.66284, -48.91777, 0.084522665, -281.5821, 78.69863, 133.78897, 102.63185, -259.9021, -450.69684, -538.17615, 102.73462, 256.75674, -122.24531, 414.3129, 90.33753, -141.62321, 64.58323, 158.11197, 264.04486, -70.996445, -74.847595, 715.87695, 398.03842, -383.14874, 81.798744, 100.61626, 183.76709, 107.91918, -74.72749, 137.34314, -12.11866, 145.58879, -8.319928, -146.90816, 21.44834, -14.692991, -26.504921, -44.41256, 169.60251, 157.3852, 57.295708, -223.72159, -105.66721, 85.917145, -124.74416, 67.12806, 148.09708, 108.51118, -38.312347, 85.37175, 124.25812, -19.708088, -252.26158, -396.83145, 30.49828, 24.158318, 4.4514937, -100.692856, -19.743237, 174.73466, -129.2712, -94.76638, -48.278133, -12.916355, -108.09598, 15.809202, 102.17793, -36.756264, -59.4923, -168.1815, 18.80262, -43.62355, 23.15091, 16.152151, -187.13437, 30.643784, 90.79638, -16.21843, -39.05421, 62.11714, -2.155285, -19.06319, -42.414265, 35.171066, 84.379585, -116.602325, -33.51855, -7.659307, 44.801598, -11.07798, -14.886959, 62.95822, 14.218942, 31.104227, -83.76815, 17.044928, 55.428146, -13.889921, -126.15556, -27.248165, 120.640366, -19.761774, 14.046745, -73.71858, 0.10803318, 10.083524, -16.70562, 4.7014112, -62.691025, 50.462833, -84.346176, -13.629369, 36.10631, -46.619267, -21.598942, -48.490704, -6.3323555, 15.65649, 48.896317, -56.442543, -6.640568, -17.761715, -70.73248, -40.4366, -36.27153, 74.86966, -46.19746, -44.013317, 2.0889094, 26.809942, -6.282827, -65.310585, 33.484646, -9.667151, 15.079562, -16.305983, 9.046168, 29.939863, 5.129214, 16.5475, -50.21096, 31.017387, 38.323303, -3.0681198, -36.80033, -1.2009478, 11.490404, 26.555166, 44.298775, -31.610435, 4.04779, -32.927948, 29.660675, 11.896004, -18.109625, 25.200207, -23.730797, 15.500507, -25.358208, 6.163124, -7.3798866, -7.921206, -16.156162, -36.149162, -1.5045524, -43.494595, 29.947332, -15.511134, -17.982704, -28.039505, -30.445019, -3.5225277, -46.805386, -1.0227482, -23.363768, -17.676548, -3.6520846, -12.88875, 4.756609, 31.271141, -6.629322, -37.832882, -0.21950912, 38.885174, 21.138603, 6.2617035, -9.60021, -11.284341, -19.242826, -1.6980145, -9.359415, -41.126484, -29.448069, -5.9372683, 25.94433, 1.3254867, -14.705631, 15.379487, 16.574158, -27.095804, -19.93113, 20.497425, 0.79107094, -7.7554317, -36.76988, -52.092567, -38.030884, -20.167278, -0.28380156, -7.2491307, 8.2373905, 7.9681587, 12.655432, 23.266579, 13.15513, 5.5212536, 12.821712, 10.402097, -1.8669, 17.029139, 7.877034, -3.6022367, -13.437171, -10.018736, 6.7936516, 12.0322485, 17.809977, -13.797862, -19.886257, -20.896944, -13.391824, -3.7870193, -6.042081, 9.495218, -8.35246, -16.302475, -16.089418, -2.2239032, 5.133191, -1.9499176, -12.571083, -26.08479, -8.472019, -4.010655, 5.987412, -1.6527638, -5.525652, -1.8339038, -6.3098893, -2.546278, -14.999996, -4.8673024, -21.751396, -23.044847, 0.98999345, 3.2206738, 1.6813838, -5.9552116, -2.9898171, -6.6439576, 10.739187, -0.41604716, -6.192164, 10.029629, -8.77803, -3.0170813, -3.2569091, -16.77877, -12.368547, 2.8054588, -1.341449, 4.0487995, 9.832774, 9.444449, 19.458578, -5.653375, -5.7151184, -1.1136432, 2.3793151, 5.6741295, -4.6841993, -0.53913784, -5.0413313, -4.085096, -17.153347, -1.8630176, 11.785563, 1.1223254, -0.71278524, -4.503395, 1.1829545, -10.829809, -1.1153163, 3.7169714, -0.4367964, -0.591923, -2.8435392, 1.75674, -4.178983, 4.8031635, -5.574907, -0.26360065, -4.317787, -11.108944, 12.20769, 3.0794377, -0.6653844, -2.5413795, -2.403656, -0.8925853, 13.288289, 6.883606, -15.778875, -1.6005001, 5.696593, 8.6179, -7.1294184, -5.413874, 5.2202096, 2.9486847, -2.3101602, -2.2582812, 15.61281, -0.8072282, -0.08025418, -4.076133, -9.068343, 4.1847715, -2.2272792, -1.0373013, -4.7336984, 1.2461259, -0.5397187, 0.18704754, -0.1272373, 0.41515422, 4.11998, -5.550728, 0.5320622, 3.4917614, 8.215734, -10.053921, -7.784162, 3.4075887, -8.748142, 0.87312615, -1.6497533, 1.9920977, -0.20973617, -1.9895163, -3.3932712, 1.6009337, 7.0630236, -3.209106, 5.2722173, -5.2088814, -1.2551317, 2.55086, -3.7561512, -1.9274446, -2.2996173, 5.709345, -2.3815656, -0.94840765, 0.60572165, 4.09223, -2.2723556, -4.731332, 2.7107785, -2.495332, 3.3961606, -2.640641, -1.3273795];
        assert_eq!(x_s, x_f_expected);
        assert_eq!(result.rc_i, [10, 7, 8, 9, 7, 9, 8, 9, 14, 11, 6, 9, 7, 9, 8, 8] );
        assert_eq!(result.rc_q, [0.36124167, -0.18374951, 0.0, 0.18374951, -0.18374951, 0.18374951, 0.0, 0.18374951, 0.8951633, 0.52643216, -0.36124167, 0.18374951, -0.18374951, 0.18374951, 0.0, 0.0]);
        assert_eq!(result.lpc_weighting, 0);
        assert_eq!(result.num_tns_filters, 2);
        assert_eq!(result.rc_order, [8, 6]);
        assert_eq!(result.nbits_tns, 42)
    }
}
