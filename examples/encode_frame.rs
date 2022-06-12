use lc3_codec::{
    common::{
        complex::Complex,
        config::{FrameDuration, SamplingFrequency},
    },
    encoder::lc3_encoder::Lc3Encoder,
};
use simple_logger::SimpleLogger;

fn main() {
    SimpleLogger::new().init().unwrap();
    let num_channels = 1;
    let sampling_frequency = SamplingFrequency::Hz48000;
    let frame_duration = FrameDuration::TenMs;
    let (integer_length, scaler_length, complex_length) =
        Lc3Encoder::calc_working_buffer_lengths(num_channels, frame_duration, sampling_frequency);
    let mut integer_buf = vec![0; integer_length];
    let mut scaler_buf = vec![0.0; scaler_length];
    let mut complex_buf = vec![Complex::default(); complex_length];
    let mut encoder = Lc3Encoder::new(
        num_channels, frame_duration, sampling_frequency, &mut integer_buf, &mut scaler_buf, &mut complex_buf,
    );
    let samples_in = vec![
        836, 739, 638, 510, 352, 200, 72, -56, -177, -297, -416, -520, -623, -709, -791, -911, -1062, -1199, -1298,
        -1375, -1433, -1484, -1548, -1603, -1648, -1687, -1724, -1744, -1730, -1699, -1650, -1613, -1594, -1563, -1532,
        -1501, -1473, -1441, -1409, -1393, -1355, -1280, -1201, -1118, -1032, -953, -860, -741, -613, -477, -355, -261,
        -168, -80, -8, 57, 127, 217, 296, 347, 385, 413, 456, 517, 575, 640, 718, 806, 888, 963, 1041, 1080, 1081,
        1083, 1072, 1062, 1067, 1056, 1035, 1019, 999, 964, 934, 909, 876, 854, 835, 813, 795, 781, 783, 772, 750, 747,
        728, 713, 726, 716, 680, 638, 580, 516, 451, 393, 351, 307, 244, 161, 79, 18, -45, -123, -215, -301, -389,
        -512, -644, -764, -888, -1006, -1126, -1253, -1378, -1500, -1614, -1716, -1813, -1926, -2051, -2176, -2301,
        -2416, -2514, -2595, -2680, -2783, -2883, -2977, -3068, -3163, -3262, -3341, -3381, -3392, -3392, -3379, -3368,
        -3353, -3318, -3292, -3244, -3169, -3109, -3049, -2989, -2922, -2844, -2790, -2743, -2672, -2588, -2490, -2371,
        -2222, -2046, -1861, -1695, -1546, -1384, -1214, -1058, -913, -761, -602, -441, -280, -124, 24, 169, 302, 421,
        546, 661, 738, 796, 851, 924, 1055, 1227, 1412, 1588, 1707, 1787, 1853, 1905, 1963, 2015, 2048, 2072, 2082,
        2093, 2099, 2095, 2097, 2086, 2063, 2060, 2069, 2052, 2012, 1977, 1956, 1948, 1918, 1843, 1748, 1641, 1533,
        1435, 1342, 1252, 1163, 1081, 1024, 989, 962, 937, 911, 879, 841, 769, 657, 541, 445, 365, 289, 202, 104, -4,
        -119, -245, -381, -523, -655, -770, -874, -957, -1017, -1069, -1118, -1173, -1256, -1370, -1497, -1629, -1745,
        -1827, -1882, -1934, -2021, -2115, -2165, -2196, -2230, -2258, -2282, -2302, -2320, -2332, -2340, -2344, -2338,
        -2313, -2269, -2215, -2152, -2072, -1978, -1885, -1793, -1704, -1621, -1528, -1419, -1310, -1213, -1116, -1014,
        -914, -820, -736, -656, -578, -514, -445, -358, -276, -206, -136, -62, 0, 56, 124, 190, 253, 316, 379, 458,
        552, 630, 686, 725, 735, 709, 661, 612, 572, 538, 507, 476, 453, 448, 453, 444, 415, 370, 316, 257, 203, 159,
        125, 107, 114, 137, 162, 181, 189, 186, 166, 145, 145, 154, 154, 161, 184, 200, 217, 254, 294, 325, 332, 320,
        302, 286, 273, 260, 266, 294, 297, 274, 251, 221, 170, 100, 29, -31, -82, -134, -187, -232, -278, -347, -426,
        -490, -548, -613, -677, -727, -755, -769, -770, -757, -741, -729, -713, -684, -659, -647, -631, -606, -588,
        -585, -577, -555, -534, -527, -528, -513, -480, -456, -440, -415, -382, -333, -244, -132, -32, 47, 130, 225,
        308, 383, 460, 533, 607, 687, 757, 817, 889, 977, 1038, 1064, 1100, 1165, 1250, 1349, 1456, 1563, 1665, 1755,
        1829, 1890, 1935, 1973, 2008, 2033, 2044, 2054, 2076, 2106, 2125, 2115, 2097, 2092, 2093, 2082, 2067, 2068,
        2095, 2135, 2169, 2193, 2213, 2219, 2202, 2163, 2101, 2033, 1992, 1985, 1990, 1986, 1978, 1977, 1976, 1969,
        1959, 1956, 1960, 1955, 1930, 1907, 1884, 1844, 1790, 1733, 1687, 1649, 1611, 1586,
    ];
    let mut buf_out: Vec<u8> = vec![0; 150];

    encoder.encode_frame(0, &samples_in, &mut buf_out).unwrap();

    log::info!("done");
}
