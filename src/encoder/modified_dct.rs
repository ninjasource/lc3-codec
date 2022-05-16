// Modified Discrete Cosine Transform Encoder

use crate::common::complex::{Complex, Scaler};
use crate::common::config::Lc3Config;
pub use crate::common::dct_iv::DiscreteCosTransformIv;
use crate::tables::band_index_tables::I_8000_7P5MS;
use crate::{
    common::config::FrameDuration,
    tables::{band_index_tables::*, mdct_windows::*},
};
use itertools::izip;

// checked against spec

#[allow(unused_imports)]
use num_traits::real::Real;

pub struct ModDiscreteCosTrans<'a> {
    config: Lc3Config,
    dct_iv: DiscreteCosTransformIv<'a>,
    freq: &'a mut [i16],
    window: &'static [Scaler],
    band_indices: &'static [usize],
}

impl<'a> ModDiscreteCosTrans<'a> {
    pub fn new(
        config: Lc3Config,
        integer_buf: &'a mut [i16],
        complex_buf: &'a mut [Complex],
    ) -> (Self, &'a mut [i16], &'a mut [Complex]) {
        let (freq, integer_buf) = integer_buf.split_at_mut(config.nf * 2);
        let (dct_iv, complex_buf) = DiscreteCosTransformIv::new(config.nf, complex_buf);

        let (window, i_fs) = match config.n_ms {
            FrameDuration::SevenPointFiveMs => match config.fs_ind {
                0 => (W_N60_7P5MS.as_slice(), I_8000_7P5MS.as_slice()),
                1 => (W_N120_7P5MS.as_slice(), I_16000_7P5MS.as_slice()),
                2 => (W_N180_7P5MS.as_slice(), I_24000_7P5MS.as_slice()),
                3 => (W_N240_7P5MS.as_slice(), I_32000_7P5MS.as_slice()),
                4 => (W_N360_7P5MS.as_slice(), I_48000_7P5MS.as_slice()),
                _ => panic!("Cannot lookup wn for 7.5ms, invalid fs_ind: {}", config.fs_ind),
            },
            FrameDuration::TenMs => match config.fs_ind {
                0 => (W_N80_10MS.as_slice(), I_8000_10MS.as_slice()),
                1 => (W_N160_10MS.as_slice(), I_16000_10MS.as_slice()),
                2 => (W_N240_10MS.as_slice(), I_24000_10MS.as_slice()),
                3 => (W_N320_10MS.as_slice(), I_32000_10MS.as_slice()),
                4 => (W_N480_10MS.as_slice(), I_48000_10MS.as_slice()),
                _ => panic!("Cannot lookup wn for 10ms, invalid fs_ind: {}", config.fs_ind),
            },
        };

        (
            Self {
                config,
                freq,
                dct_iv,
                window,
                band_indices: i_fs,
            },
            integer_buf,
            complex_buf,
        )
    }

    pub fn calc_working_buffer_lengths(config: &Lc3Config) -> (usize, usize) {
        let complex_len = DiscreteCosTransformIv::calc_working_buffer_length(config);
        let integer_len = config.nf * 2;
        (integer_len, complex_len)
    }

    fn apply_mdct(&mut self, output: &mut [Scaler]) {
        let nf = self.config.nf;
        assert_eq!(output.len(), nf);

        // apply twiddle factors to first half of output
        let half_nf = nf / 2;
        let mid = 3 * half_nf;
        let t1 = self.freq[mid - half_nf..mid].iter().rev();
        let wn1 = self.window[mid - half_nf..mid].iter().rev();
        let t2 = self.freq[mid..mid + half_nf].iter();
        let wn2 = self.window[mid..mid + half_nf].iter();

        for (output, tw1, wn1, tw2, wn2) in izip!(output[..half_nf].iter_mut(), t1, wn1, t2, wn2) {
            *output = -(*tw1 as Scaler * *wn1) - (*tw2 as Scaler * *wn2);
        }

        // apply twiddle factors to second half of output
        let t1 = self.freq[..half_nf].iter();
        let wn1 = self.window[..half_nf].iter();
        let t2 = self.freq[half_nf..nf].iter().rev();
        let wn2 = self.window[half_nf..nf].iter().rev();

        for (output, tw1, wn1, tw2, wn2) in izip!(output[half_nf..nf].iter_mut(), t1, wn1, t2, wn2) {
            *output = (*tw1 as Scaler * *wn1) - (*tw2 as Scaler * *wn2);
        }

        self.dct_iv.run(output);

        let gain = 1.0 / (2.0 * nf as Scaler).sqrt();
        for output in output.iter_mut() {
            *output *= gain;
        }
    }

    /// output and energy_bands are only written to
    pub fn run(&mut self, input: &[i16], output: &mut [Scaler], energy_bands: &mut [Scaler]) -> bool {
        assert_eq!(input.len(), self.config.nf);
        assert_eq!(output.len(), self.config.nf);
        assert_eq!(energy_bands.len(), self.config.nb);

        // update time buffer because the low delay mdct uses data from the previous frame
        self.update_time_buffer(input);

        // apply modified descrete cosine transform
        self.apply_mdct(output);

        // apply energy estimation per band (uses output to calculate energy_bands)
        self.apply_energy_estimation(output, energy_bands);

        // detect near nyquist signals
        self.is_near_nyquist(energy_bands)
    }

    fn update_time_buffer(&mut self, input: &[i16]) {
        let nf = self.config.nf;
        let z = self.config.z;

        {
            // shift time buffer by one frame
            let (left_t, right_t) = self.freq.split_at_mut(nf);
            left_t[..nf - z].copy_from_slice(&right_t[..nf - z]);
        }

        // copy input data into the middle of the time buffer
        self.freq[(nf - z)..(2 * nf - z)].copy_from_slice(input);
    }

    fn apply_energy_estimation(&self, output: &[Scaler], energy_bands: &mut [Scaler]) {
        for (b, energy_band) in energy_bands.iter_mut().enumerate() {
            *energy_band = 0.0;

            let from = self.band_indices[b];
            let to = self.band_indices[b + 1];
            let width = to - from;

            for output in &output[from..to] {
                *energy_band += output * output / width as Scaler;
            }
        }
    }

    fn is_near_nyquist(&self, energy_bands: &[Scaler]) -> bool {
        if self.config.fs <= 32000 {
            let nn_idx = match self.config.n_ms {
                FrameDuration::SevenPointFiveMs => self.config.nb - 4,
                FrameDuration::TenMs => self.config.nb - 2,
            };

            let mut lower_bands_energy = 0.0;
            let mut upper_bands_energy = 0.0;

            // e_b will have config.nb number of elements
            for (n, eb) in energy_bands.iter().enumerate() {
                if n < nn_idx {
                    lower_bands_energy += *eb;
                } else {
                    upper_bands_energy += *eb;
                }
            }

            upper_bands_energy > 30.0 * lower_bands_energy
        } else {
            false
        }
    }

    pub fn get_i_fs(&self) -> &'static [usize] {
        self.band_indices
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::common::config::SamplingFrequency;

    #[test]
    fn modified_dct_encode() {
        let config = Lc3Config::new(SamplingFrequency::Hz48000, FrameDuration::TenMs, 1);
        let mut integer_buf = [0; 960];
        let mut complex_buf = [Complex { r: 0.0, i: 0.0 }; 960];
        let mut output = [0.0; 480];
        let mut energy_bands = [0.0; 64];
        let (mut mdct, _, _) = ModDiscreteCosTrans::new(config, &mut integer_buf, &mut complex_buf);

        // first run is used to prime the time buffer for the second run
        let samples_in = [
            2360, 1327, -1757, -3008, 1624, 7974, 9860, 7697, 5603, 5346, 3010, -2665, -4220, -1339, -1748, -3510,
            -1137, 1767, 2173, 3341, 5683, 6115, 4470, 4568, 6662, 5240, -111, -4451, -4833, -1404, 2175, 856, -4067,
            -3973, -10, -1228, -3896, -1901, 876, 952, -955, -3503, -4187, -1806, -323, -1487, -3684, -5577, -3855,
            -627, -968, -4604, -8408, -7294, -2234, -704, -3477, -4680, -2539, -2668, -6826, -7878, -3208, 52, -2380,
            -5630, -4713, -1813, -1454, -2666, -3090, -4387, -6342, -5039, -249, 1197, -2666, -4296, -1770, 699, 1161,
            -384, -1553, -728, -524, -2519, -1915, 1525, 873, -1152, -59, -57, -1473, -1465, -1191, -2163, -2733,
            -1135, 812, 934, -727, -2905, -3647, -770, 4502, 6035, 2498, -282, -500, -1032, -1762, -2021, -4514, -7341,
            -5565, -3163, -3648, -2744, -1061, -1553, -1151, 1528, 2760, 226, -3213, -3780, -2144, -715, -426, -1314,
            -2667, -3675, -3445, -2001, -1727, -3510, -4339, -2796, -1615, -2267, -2846, -2037, -779, -582, -2280,
            -4601, -4394, -1337, 774, -527, -3187, -2979, -414, -1368, -4939, -4037, 95, 1033, -1870, -4529, -3350,
            -629, -1393, -3635, -2800, -205, 1604, 2218, 1052, -496, -122, 545, 443, 671, -432, -2259, -1504, 932,
            2546, 2037, 636, 395, -167, -287, 537, -1297, -2357, 1078, 3346, 2507, 2368, 2785, 2315, 2329, 2252, 61,
            -2158, -2058, -314, 364, -1998, -3891, -3311, -2271, -87, 1415, -760, -3042, -1612, 409, -1406, -3309,
            -975, 1176, -119, -2495, -3259, -2538, -3592, -5777, -4606, -1813, -3106, -6603, -6680, -5281, -5006,
            -3501, -1569, -2274, -4540, -4316, -2494, -4189, -6097, -2479, 712, -2264, -5589, -5180, -4292, -3681,
            -2458, -2192, -4226, -6151, -3777, -703, -1912, -3539, -4084, -4894, -3436, -1451, -2134, -2737, -2098,
            -1461, -436, -1077, -3375, -3160, -1316, -1216, -2777, -3554, -1481, 836, 1172, 2498, 4174, 2613, 859,
            1171, 583, -26, 808, 1668, 1497, -126, -288, 1556, 1279, -161, 64, 49, -939, 145, 3034, 3783, 1693, -416,
            623, 3683, 3412, -199, -1411, 133, -577, -2362, -645, 2330, 2027, -1349, -3288, -1588, -155, -606, -465,
            -1145, -3162, -2308, -465, -1038, -413, 382, -2235, -4405, -3483, -2256, -1726, -1314, -2721, -4380, -4169,
            -4922, -4523, -835, 514, -2242, -4940, -4670, -3074, -2593, -2210, -2239, -3775, -5406, -6113, -5787,
            -4334, -2250, -1799, -4693, -6643, -3154, 795, -313, -3347, -4675, -4992, -5028, -3183, -366, -1419, -5543,
            -6672, -3812, -1420, -1747, -2988, -4958, -7066, -5931, -2029, 323, -931, -3138, -3057, -1867, -1868,
            -2303, -1871, -1707, -3010, -4260, -2851, 30, 111, -1806, -2898, -3747, -3452, -2573, -3081, -2408, -449,
            -1185, -3488, -3224, -847, 1254, 1945, -143, -3052, -3506, -3009, -1818, 943, 1154, -1751, -2846, -1737,
            -689, -107, -639, -2397, -3588, -1871, 1908, 2537, -883, -2404, -1044, -1188, -2207, -1484, -902, -1966,
            -2159, -883, -695, -1067, -1094, -2167, -2410, -395, 153, -2045, -3212, -1691, -357, -1434, -2560, -1502,
            -686, -1455, -1456, -705, -1306, -1609, -407, -1629, -4306, -3074, -163, -238, -3266, -5080, -2559, -94,
            -513, -1331, -2111, -1935, -552, -395, -1166, -1049, -782, -1114, -904, -583, -1248, -1887, -1835, -815,
            906, 762, -1570, -1644, 1398, 1780, -1387, -1782, 1622, 3116, 1111, -655, -576,
        ];
        mdct.run(&samples_in, &mut output, &mut energy_bands);

        // second run is used to actually check the results
        let samples_in = [
            139, 465, 798, 1705, 1612, 872, 1625, 2342, 1596, -105, -1872, -1333, 879, 1315, 386, 810, 1507, -828,
            -3806, -2048, 2631, 3986, 1088, -1240, -1053, -1165, -1384, 350, 1172, -1645, -3340, -376, 1993, 476,
            -1907, -2667, -1497, -616, -1119, -465, -53, -3594, -6635, -3459, 1959, 3349, 334, -2800, -2731, -1968,
            -2979, -2347, 1211, 1740, -2037, -3389, -1402, -939, -2572, -2614, -584, -741, -2460, -1977, -1184, -1966,
            -2317, -1503, -789, -1222, -2904, -2974, -786, -1189, -3338, -2976, -1537, -607, -1181, -3322, -3470,
            -1642, -1492, -2384, -2537, -2509, -2171, -2320, -2859, -1891, -1284, -2416, -2644, -1796, -2467, -3570,
            -2654, -1729, -2448, -3859, -4513, -3166, -1988, -1938, -1755, -3085, -5161, -4943, -2759, -829, -1532,
            -3509, -2731, -1452, -2915, -3412, -1876, -2017, -3721, -4148, -2133, -158, -1157, -2651, -1618, -392,
            -1302, -2478, -1665, 107, 359, -705, -1205, -822, -39, 677, 1043, 613, -1015, -1280, 1344, 2370, -1, -836,
            1209, 2618, 2174, 570, -657, 166, 1665, 1904, 1515, 1102, 1295, 2819, 3236, 2006, 1754, 1212, 633, 2091,
            2402, 1046, 1366, 2279, 1854, 1562, 2500, 3853, 4367, 2732, 185, -100, 2530, 5024, 3739, 1063, 1064, 1901,
            2322, 2882, 2594, 2020, 1936, 1542, 1527, 3135, 3718, 1347, 160, 1697, 1633, 19, 515, 1933, 1996, 914,
            -460, -92, 1284, 1255, 630, -686, -2215, -1331, 417, 480, -260, -1530, -3182, -3210, -1494, -114, -104,
            -1579, -3781, -4137, -2059, -720, -1685, -3663, -3548, -1295, -3020, -5865, -2819, -898, -3188, -2721,
            -1234, -2836, -4315, -3332, -2044, -1941, -2419, -2621, -2345, -2585, -2885, -2388, -2415, -2200, -212,
            782, -860, -1388, 412, 622, -895, -545, 923, 908, 689, 1032, 451, -139, 863, 2485, 2568, 1396, 1678, 2962,
            2508, 1338, 2137, 3531, 3470, 3345, 3946, 4491, 3984, 2654, 2895, 4016, 3182, 1771, 2799, 4764, 4283, 2461,
            1897, 3112, 4929, 5093, 4028, 3354, 2639, 1740, 1888, 3539, 4352, 2760, 1648, 2976, 3204, 730, 74, 2515,
            4258, 3119, 384, -288, 1331, 1348, 306, 1231, 1439, -422, -34, 1162, 172, 701, 2318, 1739, 864, 637, 350,
            350, 225, 404, 801, 504, 634, 246, -1354, -493, 2154, 1788, -678, -763, 967, 1303, 789, 153, -446, -157,
            158, 146, -326, -662, 201, 153, -456, 365, 570, 177, 600, 490, 426, 983, 14, -1363, -455, 894, 429, -118,
            477, 639, 360, 595, 381, -562, -531, 1196, 2170, 1394, 1495, 2089, 1061, 579, 2497, 3743, 2451, 875, 473,
            1638, 3153, 3096, 2712, 2652, 2058, 1388, 1734, 3515, 4421, 2741, 1090, 1879, 3096, 2952, 2921, 2504, 1907,
            2972, 3139, 1801, 1623, 2058, 2366, 2564, 2418, 2357, 1816, 1023, 1663, 2365, 1073, 80, 1301, 2449, 1417,
            -370, -345, 561, 374, 967, 2181, 503, -2115, -1959, -301, 1179, 1822, 506, -1521, -1348, -338, -992, -847,
            659, 877, -608, -2355, -1424, 1037, 844, -668, -957, -637, -493, -72, 633, 995, 1450, 1159, 133, 598, 1803,
            2202, 2224, 2496, 2819, 2035, 1321, 2803, 4132, 3172, 2739, 3798, 3929, 3572, 4074, 4214, 3704, 3917, 4895,
            6134, 6475, 4847, 3899, 4885, 5697, 6304, 6412, 5687, 5283, 4580, 3967,
        ];
        let near_nyquist_flag = mdct.run(&samples_in, &mut output, &mut energy_bands);

        #[rustfmt::skip]
        let output_expected = [
            -9036.828, -16840.13, 5379.57, 1920.0739, -5654.29, 12562.138, -5867.3047, -20966.326, 1639.6295,
            12796.866, 13211.093, -3257.8875, 2721.2131, -3110.4385, 3937.997, 5711.0015, 1387.1301, 1491.5234,
            884.8571, 1035.3586, -1491.5171, 62.12583, -202.91766, -762.2081, 250.92949, 293.18628, -141.08902,
            73.47959, -206.25943, -496.9612, 377.51, 122.24463, 197.2226, 967.97296, -389.54944, 941.24036, -435.57526,
            -71.13273, -568.3641, -945.20624, 630.44867, -612.0118, 1710.7167, -1828.8589, 523.9139, -673.6406,
            48.983547, 676.83844, 1511.4803, -451.05936, 355.58655, -21.287914, 29.319094, 550.1294, -642.84375,
            -533.08704, -1356.9635, -217.25662, -449.40097, 2009.4777, -182.86682, 767.709, -1597.3046, 1865.4133,
            -1763.1051, -599.8365, -404.27213, 803.07874, 1117.2128, -333.64108, 501.0466, -75.05941, -873.61505,
            601.84174, 103.2749, 631.1625, -946.9588, -530.2806, -224.38748, -328.97388, 418.7377, -78.47528, 252.136,
            275.23694, -1436.3699, 2557.9163, 144.60983, 58.88753, 1082.7534, 760.26587, -1961.6705, 363.1704,
            -163.97765, 897.3682, -1192.7642, 582.97473, 302.49252, -671.99335, 982.9511, -1406.4033, 877.5565,
            967.8994, 32.3264, 533.6997, -6.5154777, 454.25748, -434.66266, 282.58047, -523.25684, -405.60156,
            -167.06584, -659.69885, 1081.9199, -757.32587, 2154.9468, -3856.4097, 2343.5752, 667.9661, 0.36950618,
            1849.2738, -3452.5652, -28.410355, -1028.0013, 1034.7104, 3305.0989, -1907.3907, 238.27437, 69.48102,
            -705.0223, 1758.1127, -1765.4241, 2633.9985, 3148.7654, 818.0577, 2070.1223, -976.8674, -1388.8491,
            612.43634, 703.7555, -178.43008, 1874.0172, -1341.0807, 1639.8491, -1198.2227, -445.47723, -3745.6787,
            393.6288, 3394.011, -859.0792, -3536.2332, 3986.5964, -1511.4634, 3080.3652, -268.13052, -1928.6807,
            6245.83, -2605.0432, 1926.7202, -3270.2813, 3409.4211, 575.6335, -2053.6343, 793.16125, -392.98816,
            -2152.7288, 1724.8596, -3475.1003, 1630.527, -4028.4468, 1674.9692, -408.89557, -85.14543, -894.3574,
            1672.0718, 3014.6682, 1238.2407, -1808.129, 321.9944, 551.8095, -1447.0416, 1524.1786, -947.0644,
            1458.2646, 3171.868, 697.1315, -2283.8015, -598.722, 3639.323, -1187.5538, 2475.0256, -1402.365, 1982.1636,
            -3030.782, 196.38588, 2227.344, 831.2924, -1096.4403, 767.57367, 1691.5677, -2360.7947, -788.293, 106.137,
            -1301.2643, 597.9668, 440.08826, 1620.1721, 582.7262, -1832.8705, 1315.2827, -3037.2017, -119.15167,
            -1232.7704, 845.8204, -420.47015, 2043.178, -2177.3516, 3406.8293, 678.7968, 917.7717, -2952.3325,
            375.22083, 1106.5508, 224.31845, -233.61845, -3013.4277, -227.50772, -609.0729, -865.1191, -1749.7798,
            2483.3613, -2030.978, -315.18692, -2052.7507, -2431.4587, 973.3867, -4872.0566, 6687.9937, -565.9745,
            -1343.1342, 1381.7665, -3710.7888, 462.5483, 1823.9928, -205.10674, 406.0755, 21.6697, 1645.5869,
            -2226.623, -1007.3383, 146.2967, 130.93256, -140.11192, 283.91772, -2354.3147, 143.10812, 1141.92,
            146.96822, 226.18481, -725.68134, 905.84546, 1073.7988, 2910.231, -1103.2373, 1034.4335, -3470.6013,
            884.9221, 538.9687, 1734.5514, -369.73453, -127.74741, 537.8403, 1839.0625, -395.24734, 598.5361,
            -1675.1272, 1144.0638, 802.8306, 2275.502, -619.7326, 694.37006, 335.22308, 669.91797, -1477.466,
            -732.8301, -417.8962, -671.12585, -327.846, 57.6806, 455.17822, 573.53827, 526.18274, 74.96032, 687.44965,
            -819.12616, -115.65765, 731.302, 723.4466, 277.97592, -804.3628, 114.53857, 414.6246, 141.04538,
            -127.99048, 69.38761, 632.4435, 1075.8844, -533.16016, 478.49844, -448.3171, -813.08484, 430.09323,
            594.9666, 254.83499, -187.66609, -322.8905, 580.7564, -95.19276, 464.34586, -739.16364, -237.72954,
            -132.46182, 1159.8064, 402.10947, -1061.9579, -365.74478, 138.89984, -721.0325, -318.23605, -391.4858,
            353.83157, 1240.813, 55.262062, 212.564, 319.69693, -541.3718, 193.90681, 515.3884, -449.35696, 1.1246402,
            238.76659, 500.29437, -47.26234, 195.62572, -94.43772, 106.07335, -54.96497, 66.99391, 71.99765,
            -140.57524, 56.28036, -198.55539, 111.30731, 71.48924, -303.92914, -171.88248, 407.52655, -88.83321,
            61.751926, -82.07953, 177.82426, -51.550957, 84.43923, -464.9969, -119.43584, -91.63968, 95.10355,
            -31.879934, 21.207262, 74.75225, 85.39591, 208.35684, -238.06073, 6.879123, -91.32068, 8.614929, 34.916065,
            59.987984, -58.65022, 110.9761, 126.30455, 229.54889, 20.61087, 37.55213, -1.3116863, 92.09584, 49.49659,
            -16.808111, 64.13964, -1.3163747, 22.983799, -108.65103, 116.81307, -89.27904, 8.369222, 41.453278,
            -33.404858, -49.672966, -116.35366, -96.29913, 106.49366, 53.62365, -13.980692, 42.05223, -3.446602,
            3.4768913, 8.7055025, -1.5556388, 3.7568064, 0.5999669, -0.64507174, 0.776669, 0.052840628, 0.53634894,
            0.4167444, -0.25501552, 0.4738164, -0.4083211, 0.0015128842, 0.1888111, -0.3604407, -0.28974882,
            -0.5007016, 0.24160838, 0.49307415, -0.09876297, 0.5021357, -0.098108955, 0.80545616, -0.2618836,
            -0.009408248, 0.20516127, 0.120652504, 0.41393927, -0.3130897, -0.29810908, -0.36902553, 0.16715793,
            -0.21003874, -0.16001901, -0.38915792, 0.27199608, 0.38845664, 0.31225494, -0.04743916, 0.6012375,
            0.70385355, -0.5778114, 0.3833467, -0.19554716, -0.201383, -0.6554689, -0.23560412, -0.06812805,
            -0.23855898, 0.095859334, -0.4247501, -0.37960783, 0.6076948, -0.19560331, 0.2736902, 0.16077545,
            0.15634318, 0.37252408, 0.26382494, 0.046489667, -0.30924296, -0.19435833, -0.12936735, 0.25538588,
            -0.67203575, -0.18595867, 0.2430661, 0.2590814, 0.10880158, -0.16356483, -0.029028464, -0.34360594,
            0.10336071, 0.036986865, 0.21354418,
        ];
        #[rustfmt::skip]
        let energy_bands_expected = [
            81664264.0, 283590000.0, 28939772.0, 3686683.5, 31970996.0, 157807300.0, 34425264.0, 439586850.0,
            2688385.0, 163759790.0, 174532980.0, 10613831.0, 7405001.0, 9674828.0, 15507821.0, 32615538.0, 1924130.0,
            2224642.3, 927469.8, 1114241.4, 311068.4, 74461.91, 12652.681, 144756.7, 78728.78, 487934.2, 518841.1,
            172607.8, 555146.25, 2181920.8, 304767.13, 871489.7, 179300.81, 593673.4, 1803050.0, 1777886.6, 563865.94,
            308016.47, 302368.66, 1750316.0, 959130.5, 522963.47, 705288.7, 196651.02, 3885472.3, 4010854.3, 3564607.8,
            1468960.0, 6688379.5, 8087990.5, 4930599.5, 2583068.0, 3641870.3, 2200358.0, 3107434.0, 6546483.5,
            2075304.6, 1966916.8, 888433.94, 321260.47, 280652.2, 133804.75, 31468.62, 7194.3525,
        ];
        assert_eq!(output, output_expected);
        assert_eq!(energy_bands, energy_bands_expected);
        assert_eq!(near_nyquist_flag, false);
    }
}
