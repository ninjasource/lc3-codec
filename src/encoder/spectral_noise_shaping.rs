use crate::common::{
    complex::Scaler,
    config::{FrameDuration, Lc3Config},
};
use crate::tables::spec_noise_shape_quant_tables::{
    D, HFCB, LFCB, MPVQ_OFFSETS, SNS_VQ_FAR_ADJ_GAINS, SNS_VQ_NEAR_ADJ_GAINS, SNS_VQ_REG_ADJ_GAINS,
    SNS_VQ_REG_LF_ADJ_GAINS,
};
use core::{cmp::Ordering, panic};
use itertools::izip;
use itertools::Itertools;
#[allow(unused_imports)]
use num_traits::real::Real;

// checked against spec

const N_SCALES: usize = 16;

pub struct SnsResult {
    pub ind_lf: usize,
    pub ind_hf: usize,
    pub shape_j: usize,
    pub gind: usize,
    pub ls_inda: i32, // TODO: convert this to a usize
    pub ls_indb: i32,
    pub index_joint_j: usize,
}

struct SnsStage1Result {
    pub ind_lf: usize,
    pub ind_hf: usize,
}

struct SnsStage2Result {
    pub shape_j: usize,
    pub gind: usize,
    pub ls_inda: i32, // TODO: convert this to a usize
    pub ls_indb: i32,
    pub index_joint_j: usize,
}

// checked against spec

pub struct SpectralNoiseShaping {
    // constant
    config: Lc3Config,
    g_tilt: usize, // used for pre-emphasis?
    mdct_band_indices: &'static [usize],
}

const G_TILT: [usize; 5] = [
    14, // fs_ind 8000
    18, // fs_ind 16000
    22, // fs_ind 24000
    26, // fs_ind 32000
    30, // fs_ind 44100 or 48000
];

const WEIGHTING: [Scaler; 6] = [1.0 / 12.0, 2.0 / 12.0, 3.0 / 12.0, 3.0 / 12.0, 2.0 / 12.0, 1.0 / 12.0];
const MAX_NUM_BANDS: usize = 64;
const NUM_SCALES: usize = 16;
pub const NBITS_SNS: usize = 38;

impl SpectralNoiseShaping {
    pub fn new(config: Lc3Config, mdct_band_indices: &'static [usize]) -> Self {
        let g_tilt = G_TILT[config.fs_ind];

        Self {
            config,
            g_tilt,
            mdct_band_indices,
        }
    }

    fn apply_padding_for_narrow_band(num_bands: usize, input: &[Scaler], output: &mut [Scaler]) {
        let diff = MAX_NUM_BANDS - num_bands;

        // padding is only applied for 8khz 7.5ms audio (60 bands instead of 64)
        if diff > 0 {
            for i in 0..diff {
                output[i * 2] = input[i];
                output[i * 2 + 1] = input[i];
            }
            for i in 0..num_bands {
                output[2 * diff + i] = input[diff + i];
            }
        } else {
            output.copy_from_slice(input);
        };
    }

    fn energy_band_smoothing(input: &[Scaler], output: &mut [Scaler]) {
        output[0] = 0.75 * input[0] + 0.25 * input[1];
        for b in 1..(MAX_NUM_BANDS - 1) {
            output[b] = 0.25 * input[b - 1] + 0.5 * input[b] + 0.25 * input[b + 1];
        }
        output[MAX_NUM_BANDS - 1] = 0.25 * input[MAX_NUM_BANDS - 2] + 0.75 * input[MAX_NUM_BANDS - 1];
    }

    // band energy grouping
    fn downsample(energy_bands: &[Scaler]) -> [Scaler; NUM_SCALES] {
        let mut downsampled = [0.0; NUM_SCALES];
        // downsampled[0]
        downsampled[0] = WEIGHTING[0] * energy_bands[0];
        for (k, weight) in WEIGHTING.iter().enumerate().take(6).skip(1) {
            downsampled[0] += *weight * energy_bands[k - 1];
        }

        // downsampled[1..15]
        for (b2, downsampled) in downsampled.iter_mut().enumerate().take(15).skip(1) {
            *downsampled = 0.0;
            let from = 4 * b2 - 1;
            for (weight, energy_band) in WEIGHTING.iter().zip(&energy_bands[from..from + WEIGHTING.len()]) {
                *downsampled += *weight * *energy_band;
            }
        }

        // downsampled[15]
        downsampled[15] = WEIGHTING[5] * energy_bands[63];
        for (k, weight) in WEIGHTING.iter().enumerate().take(5) {
            downsampled[15] += *weight * energy_bands[60 + k - 1];
        }

        downsampled
    }

    fn mean_removal_and_scaling(downsampled: &mut [Scaler]) {
        let downsampled_total: Scaler = downsampled.iter().sum();
        let downsampled_avg = downsampled_total / downsampled.len() as Scaler;
        for item in downsampled.iter_mut() {
            *item = 0.85 * (*item - downsampled_avg);
        }
    }

    fn attack_handling(attack_detected: bool, frame_duration: FrameDuration, input: &[Scaler], output: &mut [Scaler]) {
        assert_eq!(input.len(), NUM_SCALES);
        assert_eq!(output.len(), NUM_SCALES);

        if attack_detected {
            output[0] = (input[0] + input[1] + input[2]) / 3.0;
            output[1] = (input[0] + input[1] + input[2] + input[3]) / 4.0;
            for (n, scf) in output.iter_mut().enumerate().take(14).skip(2) {
                let window_total: Scaler = input[n - 2..n + 3].iter().sum();
                *scf = window_total / 5.0;
            }
            output[14] = (input[12] + input[13] + input[14] + input[15]) / 4.0;
            output[15] = (input[13] + input[14] + input[15]) / 3.0;

            let scale_factor_total: Scaler = output.iter().sum();
            let scale_factor_avg = scale_factor_total / output.len() as Scaler;
            let attenuation_factor = match frame_duration {
                FrameDuration::TenMs => 0.5,
                FrameDuration::SevenPointFiveMs => 0.3,
            };
            for item in output.iter_mut() {
                *item = attenuation_factor * (*item - scale_factor_avg);
            }
        } else {
            output.copy_from_slice(input);
        }
    }

    fn apply_scale_factor_interpolation(input: &[Scaler], output: &mut [Scaler]) {
        assert_eq!(input.len(), NUM_SCALES);
        assert_eq!(output.len(), MAX_NUM_BANDS);

        output[0] = input[0];
        output[1] = input[0];

        // iterate through 60 items in groups of 4 and apply them to a sliding window (of length 2) of 16 floats
        for ((out0, out1, out2, out3), (in0, in1)) in
            output[2..62].iter_mut().tuples().zip(input.iter().tuple_windows())
        {
            let diff = *in1 - *in0;
            *out0 = *in0 + (0.125 * diff);
            *out1 = *in0 + (0.375 * diff);
            *out2 = *in0 + (0.625 * diff);
            *out3 = *in0 + (0.875 * diff);
        }

        output[62] = input[15] + (0.125 * (input[15] - input[14]));
        output[63] = input[15] + (0.375 * (input[15] - input[14]));
    }

    fn reduce_scale_factors_for_narrow_band(num_bands: usize, energy_bands: &mut [Scaler]) {
        let diff = MAX_NUM_BANDS - num_bands;

        // scale factors only need to be reduced for 8khz 7.5ms audio (60 bands instead of 64)
        if diff > 0 {
            // take the average of the next two bands and save it into the one before it
            // energy_bands[..4]
            for i in 0..diff {
                energy_bands[i] = (energy_bands[2 * i] + energy_bands[2 * i + 1]) / 2.0;
            }

            // energy_bands[4..64]
            for i in diff..num_bands {
                energy_bands[i] = energy_bands[diff + 1];
            }
        }
    }

    pub fn run<'a>(&self, x: &'a mut [Scaler], e_b: &[Scaler], attack_detected: bool) -> SnsResult {
        assert_eq!(x.len(), self.config.ne);
        let mut padded = [0.0; MAX_NUM_BANDS];
        let mut smoothed = [0.0; MAX_NUM_BANDS];

        // padding
        SpectralNoiseShaping::apply_padding_for_narrow_band(self.config.nb, e_b, &mut padded);

        // smoothing
        SpectralNoiseShaping::energy_band_smoothing(&padded, &mut smoothed);

        // pre-emphasis
        let energy_bands = &mut smoothed;
        let exponent = self.g_tilt as Scaler / 630.0;
        for (b, item) in energy_bands.iter_mut().enumerate() {
            *item *= (10.0).powf(b as Scaler * exponent);
        }

        // noise floor
        let mut total_energy: Scaler = energy_bands.iter().sum();
        total_energy = (total_energy / 64.0) * (10.0).powi(-4);
        let noise_floor: Scaler = (2.0).powi(-32);
        let noise_floor = noise_floor.max(total_energy);
        for item in energy_bands.iter_mut() {
            *item = (*item).max(noise_floor);
        }

        // logarithm
        for item in energy_bands.iter_mut() {
            *item = (Scaler::EPSILON + *item).log2() / 2.0;
        }

        // band energy grouping
        let mut downsampled = SpectralNoiseShaping::downsample(energy_bands);

        // mean removal and scaling, attack handling
        Self::mean_removal_and_scaling(&mut downsampled);
        let mut scale_factors = [0.0; NUM_SCALES];
        SpectralNoiseShaping::attack_handling(attack_detected, self.config.n_ms, &downsampled, &mut scale_factors);

        // sns quantization
        let mut scfq = [0.0; NUM_SCALES];
        let (stage1, stage2) = Self::run_quant(&scale_factors, &mut scfq);

        // sns scale factors interpolation
        let mut interpolated = [0.0; MAX_NUM_BANDS];
        SpectralNoiseShaping::apply_scale_factor_interpolation(&scfq, &mut interpolated);

        // reduce scale factors if required
        SpectralNoiseShaping::reduce_scale_factors_for_narrow_band(self.config.nb, &mut interpolated);

        // scale factors are then transformed back into the linear domain
        for item in &mut interpolated {
            *item = (-*item).exp2() as Scaler;

            // use an approxmation function rater than slow libm exp2 above
            // TODO: check that we can use the raw version (range criteria) and put it behind a feature flag
            // *scale_factor = fast_math::exp2_raw(-*interpolated);
        }

        // spectral shaping
        // apply sns scale fators to mdct frequency coefficients for each band separately to generate the shaped spectrum.
        // this is descructive change to x (which we wont need again anyway) is ok because from and to never overlap
        for (scale_factor, (from, to)) in interpolated.iter().zip(self.mdct_band_indices.iter().tuple_windows()) {
            for x in &mut x[*from..*to] {
                *x *= *scale_factor;
            }
        }

        SnsResult {
            gind: stage2.gind,
            ind_hf: stage1.ind_hf,
            ind_lf: stage1.ind_lf,
            index_joint_j: stage2.index_joint_j,
            ls_inda: stage2.ls_inda,
            ls_indb: stage2.ls_indb,
            shape_j: stage2.shape_j,
            //    output: &mut x[..self.config.ne],
        }
    }

    // k is num_pulses
    fn add_unit_pulse(
        abs_x: &[Scaler],
        n_max: usize,
        k: usize,
        k_max: usize,
        candidate: &mut [i32],
        corr_xy: &mut Scaler,
        energy_y: &mut Scaler,
    ) {
        let mut corr_xy_last = *corr_xy;
        let mut energy_y_last = *energy_y;
        for _ in k..k_max {
            let n_c = 0;
            let mut n_best = 0;
            *corr_xy = corr_xy_last + abs_x[n_c];
            let mut best_corr_sq = *corr_xy * *corr_xy;
            let mut best_en = energy_y_last + 2.0 * candidate[n_c] as Scaler + 1.0;

            for (n_c, abs_x_n_c) in abs_x.iter().enumerate().take(n_max).skip(n_c + 1) {
                *corr_xy = corr_xy_last + *abs_x_n_c;
                *energy_y = energy_y_last + 2.0 * candidate[n_c] as Scaler + 1.0;
                if *corr_xy * *corr_xy * best_en > best_corr_sq * *energy_y {
                    n_best = n_c;
                    best_corr_sq = *corr_xy * *corr_xy;
                    best_en = *energy_y;
                }
            }
            corr_xy_last += abs_x[n_best];
            energy_y_last += 2.0 * candidate[n_best] as Scaler + 1.0;
            candidate[n_best] += 1;
        }
    }

    fn sns_quant_stage1(scf: &[Scaler], st1: &mut [Scaler], r1: &mut [Scaler]) -> SnsStage1Result {
        assert_eq!(scf.len(), 16);
        assert_eq!(st1.len(), 16);
        assert_eq!(r1.len(), 16);

        // stage 1
        let mut dmse_lf_min = Scaler::INFINITY;
        let mut dmse_hf_min = Scaler::INFINITY;
        let mut ind_lf = 0;
        let mut ind_hf = 0;

        // iterate through low and high frequency codebooks
        for (i, (lfcb, hfcb)) in LFCB.iter().zip(&HFCB).enumerate() {
            let mut dmse_lf = 0.0;
            let mut dmse_hf = 0.0;

            // calculate mean squared error distortions
            for (scf_lf, scf_hf, lf_code, hf_code) in izip!(&scf[..8], &scf[8..16], lfcb, hfcb) {
                dmse_lf += (*scf_lf - *lf_code) * (*scf_lf - *lf_code);
                dmse_hf += (*scf_hf - *hf_code) * (*scf_hf - *hf_code);
            }

            // remember the index of the smallest value
            if dmse_lf < dmse_lf_min {
                ind_lf = i;
                dmse_lf_min = dmse_lf;
            }

            if dmse_hf < dmse_hf_min {
                ind_hf = i;
                dmse_hf_min = dmse_hf;
            }
        }

        // first stage vector
        st1[..8].copy_from_slice(&LFCB[ind_lf][..8]);
        st1[8..16].copy_from_slice(&HFCB[ind_hf][..8]);

        for (r1, scf, st1) in izip!(r1, scf, st1) {
            *r1 = *scf - *st1;
        }

        SnsStage1Result { ind_lf, ind_hf }
    }

    fn sns_quant_stage2(r1: &mut [Scaler], st1: &[Scaler], scfq: &mut [Scaler]) -> SnsStage2Result {
        assert_eq!(st1.len(), N_SCALES);
        assert_eq!(r1.len(), N_SCALES);
        assert_eq!(scfq.len(), N_SCALES);

        let mut t2rot = [0.0; N_SCALES];
        let mut sns_y0 = [0; N_SCALES];
        let mut sns_y1 = [0; 10];
        let mut sns_y2 = [0; N_SCALES];
        let mut sns_y3 = [0; N_SCALES];
        let mut sns_xq0 = [0.0; N_SCALES];
        let mut sns_xq1 = [0.0; N_SCALES];
        let mut sns_xq2 = [0.0; N_SCALES];
        let mut sns_xq3 = [0.0; N_SCALES];

        // stage 2 target preparation - forward dct-16 transformation
        t2rot.fill(0.0);
        for (r1_row, d_row) in r1.iter().zip(&D) {
            for (t2rot, d_row_n) in t2rot.iter_mut().zip(d_row) {
                *t2rot += *r1_row * *d_row_n
            }
        }

        // step 1, shape 3 - project to or below pyramid N=16, K=6,
        let mut k = 0;
        let k_max = 6;
        let n_max = 16;
        let mut abs_sum = 0.0;
        let mut abs_x = [0.0; 16];
        for (abs_x_n, t2rot_n) in abs_x[..n_max].iter_mut().zip(t2rot.iter()) {
            *abs_x_n = (*t2rot_n).abs();
            abs_sum += *abs_x_n;
        }
        let projection_factor = (k_max as Scaler - 1.0) / abs_sum;
        let mut corr_xy = 0.0;
        let mut energy_y = 0.0;
        for (sns_y3_n, abs_x_n) in sns_y3[..n_max].iter_mut().zip(abs_x.iter()) {
            *sns_y3_n = (*abs_x_n * projection_factor).floor() as i32;
            if *sns_y3_n != 0 {
                k += *sns_y3_n as usize;
                corr_xy += *sns_y3_n as Scaler * *abs_x_n;
                energy_y += *sns_y3_n as Scaler * *sns_y3_n as Scaler;
            }
        }

        // step 2, shape 3 - add unit pulses until you reach K=6 over N=16 samples
        Self::add_unit_pulse(&abs_x, n_max, k, k_max, &mut sns_y3, &mut corr_xy, &mut energy_y);

        // step 3, shape 2 - add unit pulses until you reach K=8 over N=16 samples
        let n_max = 16;
        let k = 6;
        let k_max = 8;
        sns_y2[..n_max].copy_from_slice(&sns_y3[..n_max]);
        Self::add_unit_pulse(&abs_x, n_max, k, k_max, &mut sns_y2, &mut corr_xy, &mut energy_y);

        // step 4, shape 1 - remove any unit pulses in y1 pre-start that are not part of set A to yield y1 start
        let n_max = 10;
        sns_y1[..n_max].copy_from_slice(&sns_y2[..n_max]);
        sns_y1[n_max..].fill(0);

        // step 5, shape 1 - update energy energyy and correlation corrxy terms to reflect the pulses present in y1 start
        let mut k = 8;
        let k_max = 10;
        for (sns_y2_n, abs_x_n) in sns_y2[n_max..16].iter().zip(abs_x[n_max..16].iter()) {
            if *sns_y2_n != 0 {
                k -= *sns_y2_n;
                corr_xy -= *sns_y2_n as Scaler * *abs_x_n;
                energy_y -= *sns_y2_n as Scaler * *sns_y2_n as Scaler;
            }
        }

        // step 6, shape 1 - add unit pulses until you reach K=10 over N=10 samples (in set A)
        let k = k as usize;
        Self::add_unit_pulse(&abs_x, n_max, k, k_max, &mut sns_y1, &mut corr_xy, &mut energy_y);

        // step 7, shape 0 - add unit pulses to y0,start until you reach K=1 over N=6 samples (in set B)
        let n_max = 10;
        sns_y0[..n_max].copy_from_slice(&sns_y1[..n_max]);
        let n_max = 6;
        let mut max_abs_x = 0.0;
        let mut n_best = 0;
        for (n_c, abs_x_n_c) in abs_x.iter().enumerate().take(n_max + 10).skip(10) {
            sns_y0[n_c] = 0;
            if abs_x[n_c] > max_abs_x {
                max_abs_x = *abs_x_n_c;
                n_best = n_c;
            }
        }
        sns_y0[n_best] = 1;

        // step 8, shapes 3,2,1,0 - Add signs to non-zero positions
        for (t2rot, y0, y1, y2, y3) in izip!(
            &t2rot[..10],
            sns_y0[..10].iter_mut(),
            sns_y1[..10].iter_mut(),
            sns_y2[..10].iter_mut(),
            sns_y3[..10].iter_mut(),
        ) {
            if *t2rot < 0.0 {
                *y0 *= -1;
                *y1 *= -1;
                *y2 *= -1;
                *y3 *= -1;
            }
        }
        for (t2rot, y0, y2, y3) in izip!(
            &t2rot[10..16],
            sns_y0[10..16].iter_mut(),
            sns_y2[10..16].iter_mut(),
            sns_y3[10..16].iter_mut(),
        ) {
            if *t2rot < 0.0 {
                *y0 *= -1;
                *y2 *= -1;
                *y3 *= -1;
            }
        }

        // step 9, shapes 3,2,1,0 - unit energy normalize each yj vector to candidate vector xq_j
        Self::normalize_candidate(&sns_y0, &mut sns_xq0, 16);
        Self::normalize_candidate(&sns_y1, &mut sns_xq1, 10);
        Self::normalize_candidate(&sns_y2, &mut sns_xq2, 16);
        Self::normalize_candidate(&sns_y3, &mut sns_xq3, 16);

        // shape and gain combination determination
        let mut shape_j = 0;
        let mut gind = 0;
        let mut g_gain_i_shape_j = 0.0;
        let mut sns_xq_shape_j = &sns_xq0; // placeholder
        let mut d_mse_min = Scaler::INFINITY;
        for j in 0..4 {
            // adjustment gain candidates
            let (g_maxind_j, sns_vq_gains, sns_xq) = match j {
                0 => (1, SNS_VQ_REG_ADJ_GAINS.as_slice(), &sns_xq0),
                1 => (3, SNS_VQ_REG_LF_ADJ_GAINS.as_slice(), &sns_xq1),
                2 => (3, SNS_VQ_NEAR_ADJ_GAINS.as_slice(), &sns_xq2),
                3 => (7, SNS_VQ_FAR_ADJ_GAINS.as_slice(), &sns_xq3),
                _ => panic!("This cannot happen because of max value of j"),
            };

            for (i, sns_vq_gains) in sns_vq_gains[..g_maxind_j].iter().enumerate() {
                let mut d_mse = 0.0;
                for (sns_xq_n, t2rot_n) in sns_xq[..N_SCALES].iter().zip(t2rot.iter()) {
                    let diff = *t2rot_n - *sns_vq_gains * *sns_xq_n;
                    d_mse += diff * diff;
                }

                if d_mse < d_mse_min {
                    shape_j = j;
                    gind = i;
                    d_mse_min = d_mse;
                    g_gain_i_shape_j = *sns_vq_gains;
                    sns_xq_shape_j = sns_xq;
                }
            }
        }

        let lsb_gain = gind & 1;
        let mut idxa = 0;
        let mut idxb = 0;
        let mut ls_inda = 0;
        let mut ls_indb = 0;

        // enumeration of the selected PVQ pulse configurations
        let index_joint_j = match shape_j {
            0 => {
                Self::mvpq_enum(&mut idxa, &mut ls_inda, 10, &sns_y0);
                Self::mvpq_enum(&mut idxb, &mut ls_indb, 6, &sns_y0[10..]);
                const SZ_SHAPEA_0: usize = 2390004;
                (2 * idxb + ls_indb as usize + 2) * SZ_SHAPEA_0 + idxa
            }
            1 => {
                Self::mvpq_enum(&mut idxa, &mut ls_inda, 10, &sns_y1);
                const SZ_SHAPEA_1: usize = 2390004;
                lsb_gain * SZ_SHAPEA_1 + idxa
            }
            2 => {
                Self::mvpq_enum(&mut idxa, &mut ls_inda, 16, &sns_y2);
                idxa
            }
            3 => {
                Self::mvpq_enum(&mut idxa, &mut ls_inda, 16, &sns_y3);
                const SZ_SHAPEA_2: usize = 15158272;
                SZ_SHAPEA_2 + lsb_gain + (2 * idxa)
            }
            _ => unreachable!("shape_j can never be more than j above"),
        };

        // synthesis of the Quantized SNS scale factor vector
        for (scfq_n, st1_n, d_n) in izip!(scfq, st1, D) {
            let mut factor = 0.0;
            for (sns_xq_shape_j_col, d_n_col) in sns_xq_shape_j[..N_SCALES].iter().zip(d_n.iter()) {
                factor += *sns_xq_shape_j_col * *d_n_col;
            }

            *scfq_n = *st1_n + g_gain_i_shape_j * factor;
        }

        SnsStage2Result {
            shape_j,
            gind,
            ls_inda,
            ls_indb,
            index_joint_j,
        }
    }

    fn run_quant(scf: &[Scaler], scfq: &mut [Scaler]) -> (SnsStage1Result, SnsStage2Result) {
        // working buffers
        let mut st1 = [0.0; N_SCALES];
        let mut r1 = [0.0; N_SCALES];

        // stage 1
        let stage1 = Self::sns_quant_stage1(scf, &mut st1, &mut r1);

        // stage 2
        let stage2 = Self::sns_quant_stage2(&mut r1, &st1, scfq);

        (stage1, stage2)
    }

    // TODO: figure out a way to make lead_sign_ind a usize
    fn mvpq_enum(index: &mut usize, lead_sign_ind: &mut i32, dim_in: usize, vec_in: &[i32]) {
        // init
        let mut next_sign_ind = i32::MIN; // sentinel for first sign
        let mut k_val_acc = 0;
        *index = 0;
        let mut n = 0;

        // MPVQ-index composition loop
        let mut tmp_h_row = MPVQ_OFFSETS[n][0];
        for pos in (0..dim_in).rev() {
            let tmp_val = vec_in[pos] as i8;

            (*index, next_sign_ind) = Self::enc_push_sign(tmp_val, next_sign_ind, *index);
            *index += tmp_h_row;
            k_val_acc += if tmp_val < 0 { -tmp_val } else { tmp_val };
            if pos != 0 {
                n += 1;
            }

            tmp_h_row = if k_val_acc >= 11 {
                MPVQ_OFFSETS[n + 1][(k_val_acc % 11) as usize]
            } else {
                MPVQ_OFFSETS[n][k_val_acc as usize]
            };
        }

        *lead_sign_ind = next_sign_ind;
    }

    fn enc_push_sign(val: i8, next_sign_ind_in: i32, index_in: usize) -> (usize, i32) {
        let mut index = index_in;
        if (next_sign_ind_in & i32::MIN) == 0 && val != 0 {
            index = 2 * index_in + next_sign_ind_in as usize;
        }

        let next_sign_ind = match val.cmp(&0) {
            Ordering::Less => 1,
            Ordering::Greater => 0,
            Ordering::Equal => next_sign_ind_in,
        };

        (index, next_sign_ind)
    }

    fn normalize_candidate(y: &[i32], xq: &mut [Scaler], n_max: usize) {
        let mut norm_value: Scaler = 0.0;
        for y_n in y[..n_max].iter() {
            if *y_n != 0 {
                norm_value += *y_n as Scaler * *y_n as Scaler;
            }
        }

        norm_value = norm_value.sqrt();

        for (xq, y) in xq[..n_max].iter_mut().zip(y) {
            *xq = *y as Scaler;
            if *y != 0 {
                *xq /= norm_value;
            }
        }

        // ensure trailing zero values for N < N_SCALES
        xq[n_max..N_SCALES].fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use crate::encoder::spectral_noise_shaping::SpectralNoiseShaping;
    extern crate std;

    #[rustfmt::skip]
    #[test]
    fn sns_run() {
        use super::SpectralNoiseShaping;
        use crate::common::config::{FrameDuration, Lc3Config, SamplingFrequency};

        let config = Lc3Config::new(SamplingFrequency::Hz48000, FrameDuration::TenMs, 1);
        const I_FS: [usize; 65] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 39,
            42, 45, 48, 51, 55, 59, 63, 67, 71, 76, 81, 86, 92, 98, 105, 112, 119, 127, 135, 144, 154, 164, 175, 186,
            198, 211, 225, 240, 256, 273, 291, 310, 330, 352, 375, 400,
        ];
        let sns = SpectralNoiseShaping::new(config, &I_FS);
        let mut x = [
            4333.862, -6224.463, -784.96075, -629.00256, -4521.2563, -5597.94, 11444.9, 10770.378, -17258.375,
            -7272.1436, 3731.2341, -1100.781, 801.9373, -1642.486, 1293.8563, -2856.1506, -3446.8354, 784.334,
            -1342.4543, 1137.5818, 115.09482, 129.44528, 305.01404, 227.33159, 825.6013, -408.0842, 345.62503,
            -747.8025, 85.77078, -104.06832, -420.29608, 245.78778, -235.03473, 396.6871, -394.27603, -179.6976,
            -441.42596, 131.52014, 312.3279, -222.0512, 547.8036, -198.8515, 150.55495, 25.369715, 115.13656,
            262.71463, -262.02158, 92.42298, 407.3344, 309.2086, -333.46985, 76.226234, -38.259026, 181.16354,
            106.05741, -202.57883, 152.6206, -174.54173, 306.20004, -127.961975, -103.51504, 21.873867, -112.403755,
            111.95071, -153.21635, 243.78055, 34.76676, 94.90896, -175.7117, -96.50266, 119.09373, -183.37627,
            165.17209, -18.28745, 132.33801, -43.899914, 82.3806, 121.33909, -76.954834, -126.23615, -376.90662,
            93.23381, -64.7894, 56.625244, -95.61722, -30.707392, 204.65337, -130.98984, 81.348976, -124.39854,
            41.70939, -83.94286, 13.096184, 91.303116, -98.85425, 44.06366, -183.32977, 95.68531, -72.24291, 49.774258,
            11.598255, -170.97733, 119.103806, -19.86039, 72.16969, -60.23373, 49.393845, 1.8632008, -22.54683,
            14.740696, -25.862478, 85.43594, -123.201645, 25.474178, -40.99641, 42.769684, 3.1439052, -31.417995,
            75.12228, -35.870518, 77.69747, -95.16037, 43.889996, 15.378567, -20.081436, -59.26851, -49.733772,
            109.29692, -55.06921, 62.622684, -92.77434, 26.167402, 11.805566, -17.926796, 36.6529, -97.75811, 89.02601,
            -108.46809, 46.918003, 0.6665254, -43.38535, 28.125965, -81.499, 57.999226, -32.585823, 69.469826,
            -57.38477, 4.959151, -0.68739843, -66.03831, 16.833183, -67.647194, 89.76258, -70.793175, 6.7476416,
            -2.0812957, 10.31123, 31.437586, -74.68743, 65.709045, -53.226227, 51.62175, -29.49617, 6.5157447,
            31.459711, -23.56932, 49.85898, -75.75197, 59.5215, -6.211591, 8.512154, -20.11936, -21.318468, 29.818077,
            -5.8189096, 55.300896, -50.761887, 22.52318, -35.744045, 31.43029, 8.52194, -27.989735, 35.661293,
            -49.346134, 47.172413, -43.73692, 23.599325, -14.917218, -10.609407, 7.0554442, -46.715466, 29.360056,
            -57.135876, 50.624348, -31.318138, 1.5456918, -12.248639, -31.458002, 27.090008, -57.761642, 50.247944,
            -54.186283, 27.49958, -8.841453, -20.827255, 40.057922, -28.111763, 27.865988, -52.044796, 49.54521,
            -12.760899, 9.174177, 9.054046, -23.18019, 26.015104, -46.686455, 43.96805, -46.936455, 7.308362,
            -14.296347, -2.0165968, 30.402912, -40.558987, 37.942356, -21.74377, 28.193476, -41.117878, 17.888227,
            10.700952, -25.643028, 26.996443, -55.758022, 21.432795, -45.260723, 19.785713, -10.288355, -7.921163,
            28.01082, -27.925295, 43.08463, -25.216494, 30.426146, -17.34385, 16.627455, 1.2898464, -14.319494,
            34.463207, -34.21035, 31.25678, -37.349808, 23.691319, -9.873229, 5.365773, 17.961447, -36.032772,
            29.206081, -45.279385, 30.644823, -28.186468, 12.197845, 7.609721, -24.594517, 22.811916, -38.077267,
            37.250984, -31.065157, 20.885561, -18.663807, -10.3341675, 15.56507, -28.19391, 36.312572, -38.20083,
            32.260784, -24.58465, 9.7596445, -0.3688864, -21.875547, 27.32667, -49.103867, 32.551792, -27.189249,
            20.503016, -9.318183, -5.0149803, 15.590154, -28.063501, 40.115177, -41.26048, 32.709713, -16.276306,
            -1.2533846, 10.163813, -22.395386, 16.847048, -29.415052, 30.772745, -28.323315, 23.94466, -6.63085,
            4.991849, 18.89178, -30.507738, 32.050938, -33.220524, 30.238125, -20.853271, 7.5751615, 3.3246987,
            -16.659714, 20.89852, -38.197224, 37.874516, -26.08927, 18.602648, -9.037443, -2.8033745, 13.996278,
            -30.601213, 34.150703, -31.712082, 26.91489, -20.312435, 9.797026, 2.0979328, -15.196647, 26.359812,
            -34.916744, 35.441532, -35.220978, 19.882914, -0.93797237, -8.332617, 17.445204, -25.574184, 28.446968,
            -29.301031, 34.52865, -22.83251, 1.9836963, 8.434201, -13.310688, 24.86152, -34.90463, 32.639683,
            -24.075344, 17.827106, -11.122398, 0.6513626, 20.642588, -29.679733, 33.801483, -35.721416, 25.26395,
            -15.585679, 3.4950173, 5.775539, -18.054676, 26.103037, -29.84866, 29.245966, -24.89918, 18.478716,
            -6.674508, -6.94559, 16.25152, -19.988943, 28.957138, -36.861397, 28.690504, -18.984709, 2.388344,
            7.4726305, -17.157957, 25.553185, -29.27638, 27.887112, -25.86613, 20.226744, -6.669048, -5.3857746,
            18.362423, -28.47169, 31.05557, -29.07526, 21.91813, -16.380909, 6.0499506, 7.453848, -17.984169,
            23.573948, -25.416426, 27.816557, -26.222857, 16.595758, -5.8846307, -5.219511, 15.3829565, -23.340912,
            25.990845,
        ];
        let e_b = [
            18782358.0, 38743940.0, 616163.4, 395644.22, 20441758.0, 31336932.0, 130985740.0, 116001040.0, 297851520.0,
            52884070.0, 13922108.0, 1211718.9, 643103.44, 2697760.3, 1674064.1, 8157596.5, 11880675.0, 615179.8,
            1548138.0, 15001.449, 72356.61, 424075.13, 339332.63, 9093.422, 118530.22, 106300.99, 93872.414, 103234.38,
            129645.81, 12188.947, 48738.766, 124244.47, 12835.649, 47138.63, 10050.675, 24161.475, 15844.599, 16136.73,
            37085.188, 5236.445, 14986.677, 10497.837, 8121.843, 2109.306, 3711.1233, 3116.423, 3749.5027, 4903.189,
            3149.5522, 1745.0712, 1382.3269, 1555.3384, 994.6934, 1484.393, 888.5528, 926.9374, 639.82434, 801.4557,
            743.6313, 487.39868, 681.486, 519.567, 481.0444, 454.6319,
        ];
        let attack_detected = true;

        sns.run(&mut x, &e_b, attack_detected);

        let x_s_expected = [
            2511.287, -3606.8093, -453.28122, -360.71924, -2574.9756, -3166.2068, 6525.6, 6284.0137, -10303.951,
            -4442.8755, 2318.4038, -691.36865, 509.12134, -1054.0342, 852.0186, -1959.2595, -2463.084, 583.8584,
            -1045.9277, 886.3083, 94.29779, 106.0552, 262.7902, 195.86151, 748.0028, -369.72824, 326.68167, -706.81616,
            83.90518, -101.80473, -425.53433, 248.85109, -246.2868, 415.67813, -428.0742, -195.10165, -497.12534,
            148.11542, 351.7376, -259.38834, 639.9149, -232.2877, 182.4239, 30.73989, 139.50827, 315.7314, -314.8985,
            111.07428, 464.2896, 352.44342, -380.097, 82.403694, -41.359577, 195.84523, 114.65242, -207.70195,
            156.48032, -178.95581, 313.9437, -130.90295, -105.89414, 22.376598, -114.98715, 120.20845, -164.51794,
            261.76236, 37.331238, 106.96828, -198.03798, -108.76448, 134.22601, -216.93547, 195.3998, -21.63419,
            156.55684, -51.93392, 101.626854, 149.68706, -94.93349, -155.7282, -464.9618, 119.154236, -82.80184,
            72.36793, -122.200264, -39.244514, 270.96143, -173.43079, 107.70619, -164.70389, 55.223305, -111.1405,
            17.96329, 125.23528, -135.59274, 60.439613, -251.46301, 131.2461, -101.01771, 69.59965, 16.21791,
            -239.07864, 166.54358, -27.770905, 100.91532, -84.49127, 69.28591, 2.6135557, -31.626968, 20.677122,
            -36.27791, 119.84301, -173.36378, 35.84611, -57.688297, 60.183567, 4.423961, -44.209984, 105.70867,
            -50.634827, 109.67776, -134.32838, 61.955124, 21.708387, -28.346958, -83.66344, -70.2042, 154.10416,
            -77.64532, 88.29541, -130.808, 36.89496, 16.645363, -25.276045, 51.67908, -137.08151, 124.83691, -152.0996,
            65.79087, 0.93463665, -60.837196, 39.439693, -114.28214, 81.32954, -45.443783, 96.88175, -80.02808,
            6.91597, -0.9586373, -92.0962, 23.475348, -94.339935, 125.18178, -98.72728, 9.358742, -2.886684, 14.301312,
            43.602825, -103.58884, 91.13613, -73.82291, 71.59755, -40.91015, 9.037108, 43.62777, -32.685516, 69.14355,
            -105.05149, 82.54336, -8.614124, 11.804503, -27.901173, -29.564074, 41.35118, -8.069561, 77.09149,
            -70.76394, 31.398144, -49.828518, 43.814983, 11.879899, -39.018723, 49.713158, -68.79033, 65.760086,
            -60.97088, 33.07043, -20.903938, -14.867276, 9.887002, -65.46376, 41.143112, -80.06619, 70.94139,
            -43.88703, 2.1660235, -17.16438, -44.083027, 38.160618, -81.36653, 70.78228, -76.33006, 38.737568,
            -12.454603, -29.338528, 56.428005, -39.599926, 39.25371, -73.31344, 69.79237, -17.97577, 12.852926,
            12.684623, -32.4752, 36.446888, -65.407234, 61.598774, -65.757484, 10.238938, -20.029032, -2.825231,
            42.594162, -56.822716, 53.156845, -30.462795, 38.866444, -56.68353, 24.660025, 14.751921, -35.350494,
            37.21626, -76.86587, 29.546429, -62.394695, 27.275826, -14.183131, -10.919811, 38.614643, -38.49674,
            59.394817, -34.20599, 41.272846, -23.526806, 22.555021, 1.7496673, -19.42429, 46.749092, -46.406094,
            42.399597, -50.664745, 32.1371, -13.392963, 7.2786326, 24.364573, -48.878193, 39.617836, -60.437885,
            40.904007, -37.622654, 16.281404, 10.157282, -32.828197, 30.448824, -50.824665, 49.72176, -41.46506,
            27.87757, -24.912022, -13.79381, 20.7759, -37.632587, 48.469193, -50.989594, 42.78355, -32.603626,
            12.9430275, -0.48920912, -29.010874, 36.240032, -65.120476, 43.169476, -36.05779, 27.190653, -12.357571,
            -6.6507573, 20.675322, -37.217205, 53.199875, -54.718746, 43.378906, -21.585281, -1.6675593, 13.522394,
            -29.79583, 22.414072, -39.135113, 40.94145, -37.682613, 31.857054, -8.821981, 6.641381, 25.134476,
            -40.58887, 42.64201, -44.198082, 40.230164, -27.744131, 10.078336, 4.423329, -22.164833, 27.893787,
            -50.982807, 50.552082, -34.822014, 24.829428, -12.062506, -3.7417355, 18.681189, -40.84422, 45.581814,
            -42.326923, 35.923992, -27.111525, 13.07634, 2.8001645, -20.283352, 35.183113, -46.604267, 47.304718,
            -47.010338, 26.623617, -1.2559637, -11.157539, 23.359474, -34.244343, 38.091057, -39.234665, 46.234547,
            -30.573185, 2.6562088, 11.293563, -17.823277, 33.29007, -46.73799, 43.705185, -32.237366, 23.87085,
            -14.893113, 0.8721875, 27.640835, -39.741753, 45.260857, -47.98558, 33.93777, -20.93668, 4.6949544,
            7.7584434, -24.253351, 35.06494, -40.09654, 39.286926, -33.447765, 24.822977, -8.966054, -9.330206,
            21.831121, -26.851707, 38.898933, -49.516945, 38.540756, -25.502691, 3.2083294, 10.038194, -23.048763,
            34.32631, -39.45431, 37.582066, -34.85849, 27.25857, -8.987542, -7.2581387, 24.746119, -38.36987, 41.85204,
            -39.183273, 29.537968, -22.075731, 8.153215, 10.045177, -24.236364, 31.76943, -34.252445, 37.48698,
            -35.33923, 22.365273, -7.9304223, -7.034074, 20.73084, -31.455378, 35.02656,
        ];
        assert_eq!(x[..400], x_s_expected);
    }

    #[rustfmt::skip]
    #[test]
    fn sns_quant_run() {
        let scf = [
            0.9451774, 0.83912355, 0.7632116, 0.63639724, 0.38121527, 0.13590612, -0.017586362, -0.15777636,
            -0.22507168, -0.29468405, -0.3766759, -0.4441675, -0.5119835, -0.5442837, -0.5604709, -0.5683312,
        ];
        let mut scfq = [0.0; 16];
        let scfq_expected = [
            0.78722626, 0.8271283, 0.6942812, 0.63219905, 0.39637116, 0.10612016, -0.092263274, -0.30339628,
            0.0021636784, -0.27740508, -0.48140508, -0.49960667, -0.46797758, -0.49808747, -0.4049576, -0.42349446,
        ];

        let (stage1, stage2) = SpectralNoiseShaping::run_quant(&scf, &mut scfq);

        assert_eq!(scfq, scfq_expected);
        assert_eq!(stage2.gind, 0);
        assert_eq!(stage1.ind_hf, 17);
        assert_eq!(stage1.ind_lf, 8);
        assert_eq!(stage2.index_joint_j, 15253432);
        assert_eq!(stage2.shape_j, 3);
        assert_eq!(stage2.ls_inda, 0);
        assert_eq!(stage2.ls_indb, 0);
    }
}
