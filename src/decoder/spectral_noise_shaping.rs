use super::side_info::SnsVq;
use crate::tables::{
    band_index_tables::*,
    spec_noise_shape_quant_tables::{
        D, HFCB, LFCB, SNS_VQ_FAR_ADJ_GAINS, SNS_VQ_NEAR_ADJ_GAINS, SNS_VQ_REG_ADJ_GAINS, SNS_VQ_REG_LF_ADJ_GAINS,
    },
};
use crate::{
    common::{
        complex::Scaler,
        config::{FrameDuration, Lc3Config},
    },
    tables::spec_noise_shape_quant_tables::MPVQ_OFFSETS,
};

// checked against spec

#[allow(unused_imports)]
use num_traits::real::Real;

pub fn decode(config: &Lc3Config, sns: &SnsVq, spec_lines: &mut [Scaler]) {
    let mut quant_1st_stage_vec = [0.0; 16];
    quant_1st_stage_vec[..8].copy_from_slice(&LFCB[sns.ind_lf]);
    quant_1st_stage_vec[8..].copy_from_slice(&HFCB[sns.ind_hf]);

    let shape_j = (sns.submode_msb << 1) + sns.submode_lsb;
    let gain_i = sns.g_ind;

    let mut y = [0; 16];
    let mut z = [0; 16];

    // mpvq de-enumeration calls made for the demultiplexed shape_j
    match shape_j {
        0 => {
            mpvq_deenum(10, 10, sns.ls_inda, sns.idx_a, &mut y);
            mpvq_deenum(6, 1, sns.ls_indb, sns.idx_b, &mut z);
            y[10..16].copy_from_slice(&z[..6]);
        }
        1 => {
            mpvq_deenum(10, 10, sns.ls_inda, sns.idx_a, &mut y);
            for y_n in &mut y[10..16] {
                *y_n = 0;
            }
        }
        2 => mpvq_deenum(16, 8, sns.ls_inda, sns.idx_a, &mut y),
        3 => mpvq_deenum(16, 6, sns.ls_inda, sns.idx_a, &mut y),
        _ => panic!("invalid shape_j when making mpvq denum calls: {}", shape_j),
    };

    // unit energy normalization of the received shape
    let mut y_norm = 0.0;
    for val in y.iter().take(16) {
        y_norm += *val as Scaler * *val as Scaler;
    }
    y_norm = y_norm.sqrt();

    // reconstruction of the quantized sns scale factors - lookup adjustment gain candidates
    let mut quant_adj_gain = match shape_j {
        0 => SNS_VQ_REG_ADJ_GAINS[gain_i],
        1 => SNS_VQ_REG_LF_ADJ_GAINS[gain_i],
        2 => SNS_VQ_NEAR_ADJ_GAINS[gain_i],
        3 => SNS_VQ_FAR_ADJ_GAINS[gain_i],
        _ => panic!(
            "invalid shape_j when loopking up adjustment gain candidates: {}",
            shape_j
        ),
    };

    if y_norm != 0.0 {
        quant_adj_gain /= y_norm;
    }

    // synthesis of the quantized scale factors
    let mut quant_scale_factor = [0.0; 16];
    for n in 0..16 {
        let mut factor = 0.0;
        for (col, y_col) in y[..16].iter().enumerate() {
            factor += *y_col as Scaler * D[n][col];
        }

        quant_scale_factor[n] = quant_1st_stage_vec[n] + quant_adj_gain * factor;
    }

    // sns scale factors interpolation
    let mut scale_factor_interp = [0.0; 64];
    scale_factor_interp[0] = quant_scale_factor[0];
    scale_factor_interp[1] = quant_scale_factor[0];

    for n in 0..=14 {
        let factor_n = quant_scale_factor[n];
        let diff_next = quant_scale_factor[n + 1] - factor_n;
        scale_factor_interp[4 * n + 2] = factor_n + (1.0 / 8.0 * diff_next);
        scale_factor_interp[4 * n + 3] = factor_n + (3.0 / 8.0 * diff_next);
        scale_factor_interp[4 * n + 4] = factor_n + (5.0 / 8.0 * diff_next);
        scale_factor_interp[4 * n + 5] = factor_n + (7.0 / 8.0 * diff_next);
    }
    scale_factor_interp[62] = quant_scale_factor[15] + 1.0 / 8.0 * (quant_scale_factor[15] - quant_scale_factor[14]);
    scale_factor_interp[63] = quant_scale_factor[15] + 3.0 / 8.0 * (quant_scale_factor[15] - quant_scale_factor[14]);

    // if nb < 64 then we need to reduce the number of scale factors
    // TODO: check this again
    let nb = config.nb;
    let n2 = 64 - nb;
    if n2 != 0 {
        for i in 0..n2 {
            scale_factor_interp[i] = (scale_factor_interp[2 * i] + scale_factor_interp[2 * i + 1]) / 2.0;
        }
        for i in n2..nb {
            scale_factor_interp[i] = scale_factor_interp[i + n2];
        }
    }

    let mut g_sns = [0.0; 64];

    // transform scale factors back into the linear domain
    for b in 0..nb {
        // this is very slow (2.43 ms - but probably more accurate than the fast_math version below)
        // g_sns[b] = (scf_quint[b]).exp2();

        // use an approxmation function rater than slow libm exp2 (2.43 ms -> 0.12 ms)
        // TODO: check that we can use the raw version (range criteria)
        g_sns[b] = fast_math::exp2_raw(scale_factor_interp[b]);
    }

    let i_fs: &[usize] = match config.n_ms {
        FrameDuration::SevenPointFiveMs => match config.fs_ind {
            0 => &I_8000_7P5MS,
            1 => &I_16000_7P5MS,
            2 => &I_24000_7P5MS,
            3 => &I_32000_7P5MS,
            4 => &I_48000_7P5MS,
            _ => panic!("invalid fs_ind"),
        },
        FrameDuration::TenMs => match config.fs_ind {
            0 => &I_8000_10MS,
            1 => &I_16000_10MS,
            2 => &I_24000_10MS,
            3 => &I_32000_10MS,
            4 => &I_48000_10MS,
            _ => panic!("invalid fs_ind"),
        },
    };

    // Spectral Shaping
    for (b, g_sns_b) in g_sns[..nb].iter().enumerate() {
        // since i_fs has no overlapping indices it is safe to mutate x_hat in place
        for spec_line in spec_lines[i_fs[b]..i_fs[b + 1]].iter_mut() {
            *spec_line *= *g_sns_b;
        }
    }
}

// The logic in the spec is so mysterious that I decided to keep variable and function names as well as the business
// logic is close as possible to the spec rather than to make sense of it.
fn mpvq_deenum(dim_in: usize, k_val_in: usize, ls_ind: usize, mpvq_ind: usize, vec_out: &mut [i32]) {
    for x in vec_out.iter_mut().take(dim_in) {
        *x = 0;
    }

    let mut leading_sign = if ls_ind == 0 { 1 } else { -1 };
    let mut k_max_local = k_val_in;
    let mut ind = mpvq_ind;

    // init
    let mut k_acc;

    // loop over positions
    for pos in 0..dim_in {
        let h_row_ptr = &MPVQ_OFFSETS[dim_in - 1 - pos];
        let k_delta;
        if ind != 0 {
            k_acc = k_max_local;
            let ul_tmp_offset = h_row_ptr[k_acc];
            let mut wrap_flag = ind < ul_tmp_offset;
            let mut ul_diff = 0;
            if !wrap_flag {
                // TODO: check this, looks unnecessary
                ul_diff = ind - ul_tmp_offset;
            }

            while wrap_flag {
                k_acc -= 1;
                wrap_flag = ind < h_row_ptr[k_acc];
                if !wrap_flag {
                    // TODO: check this, looks unnecessary
                    ul_diff = ind - h_row_ptr[k_acc];
                }
            }

            ind = ul_diff;
            k_delta = k_max_local - k_acc;
        } else {
            vec_out[pos] = mind2vec_one(k_max_local, leading_sign);
            break;
        }

        k_max_local = setval_update_sign(k_delta, k_max_local, &mut leading_sign, &mut ind, &mut vec_out[pos])
    }
}

fn setval_update_sign(
    k_delta: usize,
    k_max_local_in: usize,
    leading_sign: &mut i32,
    ind_in: &mut usize,
    vec_out_item: &mut i32,
) -> usize {
    let mut k_max_local_out = k_max_local_in;
    if k_delta != 0 {
        *vec_out_item = mind2vec_one(k_delta, *leading_sign);
        *leading_sign = get_lead_sign(ind_in);
        k_max_local_out -= k_delta;
    }

    k_max_local_out
}

fn get_lead_sign(ind_in: &mut usize) -> i32 {
    let mut leading_sign = 1;

    if *ind_in & 0x1 != 0 {
        leading_sign = -1;
    }

    *ind_in >>= 1;
    leading_sign
}

fn mind2vec_one(k_val_in: usize, leading_sign: i32) -> i32 {
    if leading_sign < 0 {
        -(k_val_in as i32)
    } else {
        k_val_in as i32
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::common::config::FrameDuration;

    #[test]
    fn spectral_noise_shaping_decode() {
        let config = Lc3Config {
            fs_ind: 4,
            n_ms: FrameDuration::TenMs,
            nb: 64,
            // not used below here
            fs: 0,
            nc: 0,
            ne: 0,
            nf: 0,
            z: 0,
        };
        let sns = SnsVq {
            ind_lf: 13,
            ind_hf: 4,
            ls_inda: 1,
            ls_indb: 0,
            idx_a: 1718290,
            idx_b: 2,
            submode_lsb: 0,
            submode_msb: 0,
            g_ind: 0,
        };
        #[rustfmt::skip]
        let mut spec_lines = [-568.56555, 1972.808, 3987.5906, 2461.2402, -263.29547, -2949.6724, -2217.0242, 141.18742, 1270.6868, -385.4035, 1667.538, 568.56555, -538.0386, -1924.9376, -330.6227, 358.01404, -103.25958, 640.72064, -380.6663, -565.7268, 1425.4789, -1620.2815, -2171.777, 65.41626, -630.4777, -241.61359, 70.17501, 648.6986, 913.1724, 377.05695, 1248.8955, 447.55414, 122.224335, 387.0873, 670.9608, 202.11314, 470.52597, 9.777176, 250.14848, -869.4124, 705.27844, 764.594, -196.6836, 162.10555, 32.383907, 212.68384, -200.68219, -385.05658, 74.749176, 244.4491, 75.39133, 71.7427, -370.58853, -284.1499, -221.86317, -378.81683, -78.224205, 120.7565, 49.39573, -431.70422, -639.2284, -863.50916, -762.39795, -557.4247, -210.2326, 207.37712, 667.06, 268.28708, 266.76526, 180.27934, 266.1345, 18.510933, -71.35217, -18.678001, 61.719135, 78.072945, 41.07525, 31.682747, 65.34431, 39.83117, 50.713173, 265.0006, 87.029144, -105.01486, -92.9501, 109.07149, 213.51758, 237.69977, 244.04578, 169.18098, -60.653255, -202.20692, -5.709038, 418.32336, 390.42206, -52.63867, -409.54736, -371.99884, -123.08209, 15.460239, -13.179712, 10.579021, 10.312677, 49.179047, -104.04299, -108.42853, 45.232265, 106.2404, -18.412956, -252.0934, -189.36148, 85.68147, 260.1948, 66.20917, -89.14935, 37.372047, 196.89316, 4.2800827, -186.45068, -10.664055, 251.7307, 283.97842, 87.595795, 65.05641, 91.597, -64.72161, -260.67474, -120.33, 275.7829, 360.36444, 76.64067, -85.21214, -19.708641, 59.083878, 71.03325, 37.62242, 41.2656, 33.881153, 25.94694, -3.6607666, -15.336594, -46.333282, -68.63583, 51.637062, 181.11745, 75.242294, -25.19682, -26.834858, 23.40418, 38.82212, -11.055281, -14.305193, -22.998611, 129.22333, 87.56438, -91.969635, -113.919846, -21.006641, 165.25201, 93.82325, 70.90962, 56.982315, 15.586489, -109.50122, -156.43475, -32.77449, 127.61873, 86.515625, -71.82623, -60.272907, 74.61493, 174.22997, 35.206467, -130.87024, -37.255043, 125.10912, 72.04756, 28.948765, 1.5288572, 89.77242, 28.662163, -122.471275, -64.80139, 35.938797, 56.74036, 19.603413, -92.18097, -95.45599, 4.2908516, 4.6752024, 39.53695, 19.507534, -69.66742, -24.112461, -4.6465445, -19.89853, -2.2964888, 51.27441, 41.12713, 64.92945, 0.0, 0.0, 0.0, 61.05402, 0.0, 61.05402, 0.0, 61.05402, 0.0, 0.0, 0.0, -61.05402, 61.05402, 122.10804, -61.05402, 0.0, 122.10804, 0.0, 61.05402, -61.05402, 0.0, 0.0, -61.05402, 0.0, 0.0, 122.10804, -61.05402, -61.05402, 0.0, 0.0, 0.0, 0.0, -61.05402, 61.05402, -61.05402, 61.05402, 61.05402, 61.05402, 0.0, 0.0, 61.05402, 0.0, 0.0, 0.0, 61.05402, 61.05402, 61.05402, 0.0, 0.0, 61.05402, 0.0, 0.0, 61.05402, 122.10804, -122.10804, 0.0, -61.05402, -61.05402, 0.0, 0.0, 0.0, 0.0, -61.05402, 0.0, 0.0, -61.05402, 61.05402, 61.05402, 61.05402, -122.10804, -61.05402, -61.05402, 0.0, 61.05402, -61.05402, 0.0, 0.0, 61.05402, 0.0, 0.0, 0.0, -61.05402, 61.05402, -61.05402, 0.0, 0.0, -61.05402, 61.05402, 0.0, 0.0, 0.0, 61.05402, 61.05402, 0.0, -61.05402, -61.05402, 0.0, -61.05402, 0.0, -61.05402, -61.05402, 0.0, 61.05402, 0.0, 0.0, 0.0, 0.0, -61.05402, 0.0, 61.05402, 61.05402, 0.0, 0.0, 0.0, 0.0, -61.05402, -61.05402, -61.05402, 0.0, 0.0, 0.0, -61.05402, 0.0, 0.0, 61.05402, 0.0, -61.05402, 0.0, 61.05402, 0.0, 0.0, 0.0, -61.05402, 0.0, 0.0, 0.0, -61.05402, 0.0, 0.0, -61.05402, 0.0, 0.0, 0.0, 0.0, -61.05402, 61.05402, 0.0, 0.0, 0.0, 0.0, 61.05402, 0.0, 0.0, 0.0, 0.0, -61.05402, -61.05402, 61.05402, 0.0, 0.0, 0.0, -19.079382, -19.079382, -19.079382, -19.079382, 19.079382, -19.079382, 19.079382, 0.0, 0.0, 0.0, -61.05402, -61.05402, 0.0, 0.0, 61.05402, -61.05402, -61.05402, 0.0, 0.0, 61.05402, 0.0, 0.0, 0.0, 0.0, -61.05402, 0.0, 61.05402, 0.0, 61.05402, 0.0, 0.0, 0.0, 0.0, 0.0, -61.05402, 0.0, 0.0, 61.05402, 61.05402];

        decode(&config, &sns, &mut spec_lines);

        #[rustfmt::skip]
        let spec_lines_expected = [-2268.137, 7869.9785, 15884.984, 9776.979, -1042.96, -11651.202, -8543.342, 519.08997, 4455.2827, -1288.14, 5093.401, 1520.3356, -1260.0343, -3961.906, -641.0432, 703.6705, -205.72899, 1298.229, -831.222, -1235.3197, 3581.536, -4070.9802, -6290.9507, 189.4902, -2103.3403, -806.0485, 248.62497, 2298.2922, 3174.625, 1310.8307, 4260.057, 1526.6339, 409.04425, 1295.4526, 2116.6775, 637.6055, 1343.3187, 27.913153, 714.1564, -2245.3208, 1821.4329, 1974.6196, -459.77234, 378.94186, 75.70141, 449.14334, -423.7984, -813.15814, 141.91173, 464.08798, 143.13086, 122.60409, -633.3142, -485.5956, -379.15125, -581.63947, -120.10629, 185.41084, 75.84274, -589.5617, -872.96936, -1179.2609, -1041.1772, -671.12726, -253.11552, 249.67758, 803.12585, 285.46893, 283.84964, 191.82494, 283.1785, 17.406265, -67.094124, -17.563364, 58.03595, 73.413826, 35.959316, 27.736652, 57.205654, 34.870186, 44.39683, 227.21794, 74.620895, -90.04228, -79.69767, 93.52053, 179.29234, 199.59831, 204.92711, 142.06258, -50.931004, -169.79471, -4.6945424, 343.98734, 321.0441, -43.284782, -336.7708, -305.89468, -90.73599, 11.397272, -9.71607, 7.7988434, 7.6024947, 36.25474, -76.700386, -65.53449, 27.338501, 64.21198, -11.1288395, -152.36592, -114.450584, 51.78611, 129.27042, 32.89415, -44.291332, 18.567244, 97.82079, 2.126438, -92.632744, -4.3691993, 103.137276, 116.34959, 35.889114, 26.654442, 37.52846, -26.51727, -106.801765, -40.06586, 91.82647, 119.98929, 25.518776, -28.37279, -6.5623174, 19.672953, 23.651693, 10.104294, 11.0827465, 9.099498, 6.9685974, -0.98317605, -4.1189656, -12.44378, -18.4336, 13.868221, 39.34896, 16.346886, -5.4741755, -5.83005, 5.084713, 8.434362, -2.4018328, -3.1078978, -4.9966, 28.074621, 15.33394, -16.105371, -19.94921, -3.678603, 28.938301, 16.42997, 12.417422, 9.978525, 2.7294464, -19.175436, -23.296207, -4.8807654, 19.004936, 12.883876, -10.696337, -8.97582, 11.111629, 25.946264, 5.242934, -19.48915, -5.5480075, 16.77535, 9.660551, 3.8816166, 0.20499796, 12.037202, 3.8431873, -16.421652, -8.688943, 4.8188806, 7.608073, 2.6285381, -11.124037, -11.519255, 0.51780313, 0.564185, 4.7711635, 2.3540926, -8.40719, -2.909797, -0.5607267, -2.4012764, -0.27713123, 6.1875944, 4.4743004, 7.0638013, 0.0, 0.0, 0.0, 6.6421857, 0.0, 6.6421857, 0.0, 6.6421857, 0.0, 0.0, 0.0, -6.4278693, 6.4278693, 12.855739, -6.4278693, 0.0, 12.855739, 0.0, 6.4278693, -6.4278693, 0.0, 0.0, -6.4278693, 0.0, 0.0, 13.372511, -6.6862555, -6.6862555, 0.0, 0.0, 0.0, 0.0, -6.6862555, 6.6862555, -6.6862555, 6.6862555, 6.6862555, 6.6862555, 0.0, 0.0, 6.9530754, 0.0, 0.0, 0.0, 6.9530754, 6.9530754, 6.9530754, 0.0, 0.0, 6.9530754, 0.0, 0.0, 6.9530754, 13.906151, -13.906151, 0.0, -7.2283335, -7.2283335, 0.0, 0.0, 0.0, 0.0, -7.2283335, 0.0, 0.0, -7.2283335, 7.2283335, 7.2283335, 7.2283335, -14.456667, -7.2283335, -7.2283335, 0.0, 8.2442, -8.2442, 0.0, 0.0, 8.2442, 0.0, 0.0, 0.0, -8.2442, 8.2442, -8.2442, 0.0, 0.0, -8.2442, 8.2442, 0.0, 0.0, 0.0, 10.337095, 10.337095, 0.0, -10.337095, -10.337095, 0.0, -10.337095, 0.0, -10.337095, -10.337095, 0.0, 10.337095, 0.0, 0.0, 0.0, 0.0, -10.337095, 0.0, 10.337095, 12.974136, 0.0, 0.0, 0.0, 0.0, -12.974136, -12.974136, -12.974136, 0.0, 0.0, 0.0, -12.974136, 0.0, 0.0, 12.974136, 0.0, -12.974136, 0.0, 12.974136, 0.0, 0.0, 0.0, -16.216764, 0.0, 0.0, 0.0, -16.216764, 0.0, 0.0, -16.216764, 0.0, 0.0, 0.0, 0.0, -16.216764, 16.216764, 0.0, 0.0, 0.0, 0.0, 16.216764, 0.0, 0.0, 0.0, 0.0, -20.320456, -20.320456, 20.320456, 0.0, 0.0, 0.0, -6.3501425, -6.3501425, -6.3501425, -6.3501425, 6.3501425, -6.3501425, 6.3501425, 0.0, 0.0, 0.0, -20.320456, -20.320456, 0.0, 0.0, 25.512432, -25.512432, -25.512432, 0.0, 0.0, 25.512432, 0.0, 0.0, 0.0, 0.0, -25.512432, 0.0, 25.512432, 0.0, 25.512432, 0.0, 0.0, 0.0, 0.0, 0.0, -25.512432, 0.0, 0.0, 25.512432, 25.512432];
        assert_eq!(spec_lines, spec_lines_expected);
    }

    #[test]
    fn mpvq_deenum_test1() {
        let mut vec_out = [0; 16];

        mpvq_deenum(10, 10, 1, 1718290, &mut vec_out);

        assert_eq!(vec_out, [0, -2, 0, 0, 1, 1, 3, -2, 1, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn mpvq_deenum_test2() {
        let mut vec_out = [0; 16];

        mpvq_deenum(6, 1, 0, 2, &mut vec_out);

        assert_eq!(vec_out, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }
}
