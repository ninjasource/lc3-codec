#![no_main]
#![no_std]

use core::{
    sync::atomic::{AtomicU32, Ordering},
    time::Duration,
};
use cortex_m::peripheral::{syst::SystClkSource, SYST};
use cortex_m_rt::{entry, exception};
use cortex_m_semihosting::{debug, hprintln};
use lc3_codec::{
    common::{
        complex::Complex,
        config::{FrameDuration, SamplingFrequency},
    },
    decoder::lc3_decoder::Lc3Decoder,
    encoder::lc3_encoder::Lc3Encoder,
};
use panic_semihosting as _;

fn init() {
    let p = cortex_m::Peripherals::take().unwrap();
    let mut syst = p.SYST;
    syst.set_clock_source(SystClkSource::Core);
    syst.set_reload(SYSTICK_EVERY_NUM_TICKS);
    syst.enable_counter();
    syst.enable_interrupt();
}

#[entry]
fn main() -> ! {
    init();
    encode_frame();
    decode_frame();
    debug::exit(debug::EXIT_SUCCESS);
    loop {}
}

fn decode_frame() {
    let buf_in = [
        0xe0, 0x12, 0x4f, 0x41, 0x29, 0xfa, 0xf5, 0x3c, 0x1b, 0x8a, 0x89, 0x88, 0xf8, 0xcd, 0x92, 0x85, 0x4f, 0x2c,
        0x7c, 0x44, 0x4a, 0xe4, 0x36, 0x4f, 0xc9, 0x23, 0x4e, 0x2f, 0xd5, 0x90, 0x37, 0xf7, 0x57, 0xba, 0x89, 0xeb,
        0x15, 0x2e, 0x70, 0xb3, 0x3e, 0x75, 0x8f, 0x5e, 0x3, 0x79, 0x63, 0x2d, 0xe0, 0xa5, 0xe0, 0xb2, 0x11, 0x22,
        0x80, 0x77, 0x0, 0x79, 0x74, 0x44, 0x1e, 0x1e, 0xc, 0xbd, 0x8c, 0x15, 0x98, 0x45, 0x32, 0x54, 0xf2, 0x7b, 0xa1,
        0xbc, 0xd6, 0x8, 0x14, 0xa7, 0xab, 0x24, 0xc1, 0x31, 0xa4, 0x73, 0xf0, 0xe1, 0x45, 0xa7, 0x60, 0xcb, 0x10,
        0x5c, 0x47, 0x1f, 0x22, 0xd, 0x4e, 0x2, 0x7f, 0x72, 0xae, 0x45, 0x4a, 0xae, 0x7d, 0x6b, 0xaa, 0x70, 0xe, 0xbb,
        0x19, 0x62, 0x9b, 0x2e, 0x1, 0x17, 0xb4, 0x2c, 0x8, 0x49, 0xf8, 0x59, 0x43, 0x64, 0x21, 0xbe, 0x34, 0xea, 0x37,
        0x97, 0xb3, 0x49, 0x9c, 0x84, 0x8d, 0xe5, 0xa6, 0x19, 0xcc, 0x20, 0x5b, 0xe0, 0x1b, 0x57, 0x67, 0x15, 0x32,
        0x4d, 0xc5, 0x7c,
    ];

    const NUM_CH: usize = 1;
    const FREQ: SamplingFrequency = SamplingFrequency::Hz48000;
    const DURATION: FrameDuration = FrameDuration::TenMs;
    const SCALER_COMPLEX_LENS: (usize, usize) = Lc3Decoder::<NUM_CH>::calc_working_buffer_lengths(DURATION, FREQ);
    let mut scaler_buf = [0.0; SCALER_COMPLEX_LENS.0];
    let mut complex_buf = [Complex::default(); SCALER_COMPLEX_LENS.1];
    let mut decoder = Lc3Decoder::<NUM_CH>::new(DURATION, FREQ, &mut scaler_buf, &mut complex_buf);
    let mut samples_out = [0; 480];

    let from = uptime();
    decoder.decode_frame(16, 0, &buf_in, &mut samples_out).unwrap();
    let to = uptime();

    hprintln!("Decoded in {} microseconds", (to - from).as_micros()).unwrap();
}

fn encode_frame() {
    let samples_in = [
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
    let mut buf_out = [0u8; 70];
    const NUM_CH: usize = 1;
    const DURATION: FrameDuration = FrameDuration::TenMs;
    const FREQ: SamplingFrequency = SamplingFrequency::Hz48000;
    const BUF_LENGTHS: (usize, usize, usize) = Lc3Encoder::<NUM_CH>::calc_working_buffer_lengths(DURATION, FREQ);
    let mut integer_buf = [0; BUF_LENGTHS.0];
    let mut scaler_buf = [0.0; BUF_LENGTHS.1];
    let mut complex_buf = [Complex::default(); BUF_LENGTHS.2];
    let mut encoder = Lc3Encoder::<NUM_CH>::new(DURATION, FREQ, &mut integer_buf, &mut scaler_buf, &mut complex_buf);

    let from = uptime();
    encoder.encode_frame(0, &samples_in, &mut buf_out).unwrap();
    let to = uptime();

    hprintln!("Encoded in {} microseconds", (to - from).as_micros()).unwrap();
}

// called every time the SYST reloads (every SYSTICK_EVERY_NUM_TICKS)
#[exception]
fn SysTick() {
    let curr = OVERFLOWS.load(Ordering::Relaxed);

    // so that this works on debug builds too
    let (curr, _) = curr.overflowing_add(1);

    OVERFLOWS.store(curr, Ordering::Relaxed);
}

// Counter of OVERFLOW events -- an OVERFLOW occurs every SYSTICK_EVERY_NUM_TICKS
static OVERFLOWS: AtomicU32 = AtomicU32::new(0);

// dont change this without looking at the uptime function
const SYSTICK_EVERY_NUM_TICKS: u32 = 0x00ff_ffff;

/// Returns the time elapsed since the call to the `init` function
/// Code below is adapted from Ferrous Systems embedded-trainings-2020
///
/// Calling this function before calling `init` will return a value of `0` nanoseconds.
pub fn uptime() -> Duration {
    // here we are going to perform a 64-bit read of the number of ticks elapsed
    //
    // a 64-bit load operation cannot performed in a single instruction so the operation can be
    // preempted by the RTC0 interrupt handler (which increases the OVERFLOWS counter)
    //
    // the loop below will load both the lower and upper parts of the 64-bit value while preventing
    // the issue of mixing a low value with an "old" high value -- note that, due to interrupts, an
    // arbitrary amount of time may elapse between the `hi` load and the `low` load
    let overflows = &OVERFLOWS as *const AtomicU32 as *const u32;
    let (high, low) = loop {
        unsafe {
            let syst = core::mem::transmute::<_, SYST>(());

            // NOTE volatile is used to order these load operations among themselves
            let hi = overflows.read_volatile();

            // this counts down from SYSTICK_EVERY_NUM_TICKS to zero
            let low = syst.cvr.read();

            let hi_again = overflows.read_volatile();

            // if hi1 equals hi2 it means that the SYSTICK interrupt did not fire and we can trust the low value
            if hi == hi_again {
                // low value counts down
                let low = SYSTICK_EVERY_NUM_TICKS - low;
                break (hi, low);
            }
        }
    };

    if high > 0 {
        // slow path
        let ticks = u64::from(high) * SYSTICK_EVERY_NUM_TICKS as u64 | u64::from(low);
        let ticks_per_10ms = SYST::get_ticks_per_10ms() as u64;
        let ticks_per_second = SYST::get_ticks_per_10ms() as u64 * 100;
        let secs = ticks / ticks_per_second;
        let nanos = ticks % ticks_per_second * 10_000_000 / ticks_per_10ms;
        Duration::new(secs, nanos as u32)
    } else {
        // fast(ish) path (couldnt figure out how to do this with 32 bit math without overflowing)
        let nanos = low as u64 * 10_000_000 as u64 / SYST::get_ticks_per_10ms() as u64;
        Duration::new(0, nanos as u32)
    }
}
