use lc3_codec::{
    common::{
        complex::Complex,
        config::{FrameDuration, Lc3Config, SamplingFrequency},
    },
    decoder::lc3_decoder::Lc3Decoder,
};
use simple_logger::SimpleLogger;

fn main() {
    SimpleLogger::new().init().unwrap();
    let sampling_frequency = SamplingFrequency::Hz48000;
    let frame_duration = FrameDuration::TenMs;
    let config = Lc3Config::new(sampling_frequency, frame_duration, 1);
    let (scaler_length, complex_length) = Lc3Decoder::<1>::calc_working_buffer_lengths(&config);
    let mut scaler_buf = vec![0.0; scaler_length];
    let mut complex_buf = vec![Complex::new(0., 0.); complex_length];

    let mut decoder = Lc3Decoder::<2>::new(config.clone(), &mut scaler_buf, &mut complex_buf);
    let mut samples_out = [0; 480];

    // slow
    let buf_in: [u8; 150] = [
        187, 56, 111, 155, 76, 236, 70, 99, 10, 135, 219, 76, 176, 3, 108, 203, 131, 111, 206, 221, 195, 25, 96, 240,
        18, 202, 163, 241, 109, 142, 198, 122, 176, 70, 37, 6, 35, 190, 110, 184, 251, 162, 71, 7, 151, 58, 42, 79,
        200, 192, 99, 157, 234, 156, 245, 43, 84, 64, 167, 32, 52, 106, 43, 75, 4, 102, 213, 123, 168, 120, 213, 252,
        208, 118, 78, 115, 154, 158, 157, 26, 152, 231, 121, 146, 203, 11, 169, 227, 75, 154, 237, 154, 227, 145, 196,
        182, 207, 94, 95, 26, 184, 248, 1, 118, 72, 47, 18, 205, 56, 96, 195, 139, 216, 240, 113, 233, 44, 198, 245,
        157, 139, 70, 162, 182, 139, 136, 165, 68, 79, 247, 161, 126, 17, 135, 36, 30, 229, 24, 196, 2, 5, 65, 111, 80,
        124, 168, 70, 156, 198, 60,
    ];

    /*
        // fast
        let buf_in: [u8; 150] = [
            0xe0, 0x12, 0x4f, 0x41, 0x29, 0xfa, 0xf5, 0x3c, 0x1b, 0x8a, 0x89, 0x88, 0xf8, 0xcd, 0x92,
            0x85, 0x4f, 0x2c, 0x7c, 0x44, 0x4a, 0xe4, 0x36, 0x4f, 0xc9, 0x23, 0x4e, 0x2f, 0xd5, 0x90,
            0x37, 0xf7, 0x57, 0xba, 0x89, 0xeb, 0x15, 0x2e, 0x70, 0xb3, 0x3e, 0x75, 0x8f, 0x5e, 0x3,
            0x79, 0x63, 0x2d, 0xe0, 0xa5, 0xe0, 0xb2, 0x11, 0x22, 0x80, 0x77, 0x0, 0x79, 0x74, 0x44,
            0x1e, 0x1e, 0xc, 0xbd, 0x8c, 0x15, 0x98, 0x45, 0x32, 0x54, 0xf2, 0x7b, 0xa1, 0xbc, 0xd6,
            0x8, 0x14, 0xa7, 0xab, 0x24, 0xc1, 0x31, 0xa4, 0x73, 0xf0, 0xe1, 0x45, 0xa7, 0x60, 0xcb,
            0x10, 0x5c, 0x47, 0x1f, 0x22, 0xd, 0x4e, 0x2, 0x7f, 0x72, 0xae, 0x45, 0x4a, 0xae, 0x7d,
            0x6b, 0xaa, 0x70, 0xe, 0xbb, 0x19, 0x62, 0x9b, 0x2e, 0x1, 0x17, 0xb4, 0x2c, 0x8, 0x49,
            0xf8, 0x59, 0x43, 0x64, 0x21, 0xbe, 0x34, 0xea, 0x37, 0x97, 0xb3, 0x49, 0x9c, 0x84, 0x8d,
            0xe5, 0xa6, 0x19, 0xcc, 0x20, 0x5b, 0xe0, 0x1b, 0x57, 0x67, 0x15, 0x32, 0x4d, 0xc5, 0x7c,
        ];
    */
    decoder.decode_frame(16, 0, &buf_in, &mut samples_out).unwrap();
}
