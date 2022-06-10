// Copyright 2022 David Haig
// Licensed under the Apache License, Version 2.0 (the "License");
//

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(not(feature = "alloc"))]
use heapless::Vec;

use super::{
    arithmetic_codec::{self, ArithmeticData, ArithmeticDecodeError},
    buffer_reader::BufferReader,
    global_gain,
    long_term_post_filter::LongTermPostFilter,
    modified_dct::ModDiscreteCosTrans,
    noise_filling, output_scaling,
    packet_loss_concealment::PacketLossConcealment,
    residual_spectrum,
    side_info::SideInfo,
    side_info_reader::{self, SideInfoError},
    spectral_noise_shaping, temporal_noise_shaping,
};
use crate::common::{
    complex::{Complex, Scaler},
    config::{FrameDuration, Lc3Config, SamplingFrequency},
    constants::MAX_LEN_SPECTRAL,
};

/// Main entry point of library - Start here for the Decoder

#[derive(Debug)]
pub enum Lc3DecoderError {
    SideInfo(SideInfoError),
    ArithmeticDecode(ArithmeticDecodeError),
    InvalidSampleOutBuffer(OutputBufferErrorDetails),
    Only16BitsPerAudioSampleSupported,
}

#[derive(Debug)]
pub struct OutputBufferErrorDetails {
    pub required_length: usize,
    pub actual_length: usize,
}

#[cfg(not(feature = "alloc"))]
pub struct Lc3Decoder<'a, const NUM_CHANNELS: usize = 2> {
    config: Lc3Config,
    channels: heapless::Vec<DecoderChannel<'a>, NUM_CHANNELS>,
}

#[cfg(feature = "alloc")]
pub struct Lc3Decoder<'a> {
    config: Lc3Config,
    channels: alloc::vec::Vec<DecoderChannel<'a>>,
}

struct DecoderChannel<'a> {
    spec_lines: &'a mut [Scaler],   // stores spectral lines (length ne e.g. 400)
    freq_samples: &'a mut [Scaler], // stores frequency samples (length nf e.g. 480)
    packet_loss: PacketLossConcealment<'a>,
    modified_dct: ModDiscreteCosTrans<'a>,
    post_filter: LongTermPostFilter<'a>,
    frame_index: usize,
}

impl<'a> DecoderChannel<'a> {
    // 7.42 ms (used to be 31.60 ms)
    pub fn decode(
        &mut self,
        config: &Lc3Config,
        num_bits_per_audio_sample: usize,
        buf_in: &[u8],
        samples_out: &mut [i16],
    ) -> Result<(), Lc3DecoderError> {
        if num_bits_per_audio_sample != 16 {
            return Err(Lc3DecoderError::Only16BitsPerAudioSampleSupported);
        }

        self.frame_index += 1;
        let nbits = buf_in.len() * 8;

        // TODO: should we rather preallocate this for better performance?
        let mut spec_lines_int = [0; MAX_LEN_SPECTRAL];

        // read_frame = 1.983 ms (from 2.197 ms)
        let long_term_post_filter_info = match read_frame(buf_in, config, &mut spec_lines_int) {
            Ok((side_info, arithmetic_data)) => {
                // copy to float buffer
                for (to, from) in self.spec_lines.iter_mut().zip(&spec_lines_int[..config.ne]) {
                    *to = *from as Scaler;
                }

                // 0.091 (from 0.03 ms)
                residual_spectrum::decode(side_info.lsb_mode, &arithmetic_data.residual_bits, self.spec_lines);

                // 0.549 ms (from 0.427 ms)
                noise_filling::apply_noise_filling(
                    arithmetic_data.is_zero_frame,
                    arithmetic_data.noise_filling_seed,
                    side_info.bandwidth,
                    config.n_ms,
                    side_info.noise_factor,
                    &spec_lines_int,
                    self.spec_lines,
                );

                // 0.05 ms
                global_gain::apply_global_gain(
                    arithmetic_data.frame_num_bits,
                    config.fs_ind,
                    side_info.global_gain_index,
                    self.spec_lines,
                );

                // 0.03 ms or 100 ms (if activated) or 0.915 ms (with new cache code) (from 0.854 ms (with new cache code))
                temporal_noise_shaping::apply_temporal_noise_shaping(
                    config.n_ms,
                    side_info.bandwidth,
                    side_info.num_tns_filters,
                    &arithmetic_data.reflect_coef_order,
                    &arithmetic_data.reflect_coef_ints,
                    self.spec_lines,
                );

                // 0.335 ms (with fast exp2() function otherwise 2.43 ms) (from 0.183 ms)
                spectral_noise_shaping::decode(config, &side_info.sns_vq, self.spec_lines);

                // (0 ms) (from 0.091 ms)
                self.packet_loss.save(self.spec_lines);

                side_info.long_term_post_filter_info
            }
            Err(_e) => {
                // log::warn!("Corrupt input: {:?}", _e);
                self.packet_loss.load_into(self.spec_lines)
            }
        };

        // 2.288 ms
        self.modified_dct.run(self.spec_lines, self.freq_samples);

        // 0.152 ms (from 0.122 ms)
        self.post_filter
            .run(&long_term_post_filter_info, nbits, self.freq_samples);

        // 0.213 ms (from 0.274 ms)
        output_scaling::scale_and_round(self.freq_samples, 16, samples_out);
        Ok(())
    }

    pub const fn calc_working_buffer_lengths(config: &Lc3Config) -> (usize, usize) {
        let (dct_scaler_length, dct_complex_length) = ModDiscreteCosTrans::calc_working_buffer_length(config);
        let packet_loss_length = PacketLossConcealment::calc_working_buffer_length(config);
        let long_term_length = LongTermPostFilter::calc_working_buffer_length(config);
        let scaler_length = config.ne + packet_loss_length + dct_scaler_length + long_term_length;
        (scaler_length, dct_complex_length)
    }
}

fn read_frame(buf: &[u8], config: &Lc3Config, x: &mut [i32]) -> Result<(SideInfo, ArithmeticData), Lc3DecoderError> {
    let mut reader = BufferReader::new();

    // 0.274 ms
    let side_info =
        side_info_reader::read(buf, &mut reader, config.fs_ind, config.ne).map_err(Lc3DecoderError::SideInfo)?;

    let arithmetic_data =
        arithmetic_codec::decode(buf, &mut reader, config.fs_ind, config.ne, &side_info, &config.n_ms, x)
            .map_err(Lc3DecoderError::ArithmeticDecode)?;

    Ok((side_info, arithmetic_data))
}

#[cfg(not(feature = "alloc"))]
impl<'a, const NUM_CHANNELS: usize> Lc3Decoder<'a, NUM_CHANNELS> {
    pub fn new(
        frame_duration: FrameDuration,
        sampling_frequency: SamplingFrequency,
        scaler_buf: &'a mut [Scaler],
        complex_buf: &'a mut [Complex],
    ) -> Self {
        let config = Lc3Config::new(sampling_frequency, frame_duration);
        let mut channels: Vec<DecoderChannel<'a>, NUM_CHANNELS> = Vec::new();
        let mut scaler_buf_saved = scaler_buf;
        let mut complex_buf_saved = complex_buf;

        for _ in 0..NUM_CHANNELS {
            let (x_hat, scaler_buf) = scaler_buf_saved.split_at_mut(config.ne);
            let (packet_loss, scaler_buf) = PacketLossConcealment::new(config.ne, scaler_buf);
            let (mdct, scaler_buf, complex_buf) = ModDiscreteCosTrans::new(config, scaler_buf, complex_buf_saved);
            let (post_filter, scaler_buf) = LongTermPostFilter::new(config, scaler_buf);
            let (freq_buf, scaler_buf) = scaler_buf.split_at_mut(config.nf);

            let channel = DecoderChannel {
                spec_lines: x_hat,
                packet_loss,
                modified_dct: mdct,
                post_filter,
                frame_index: 0,
                freq_samples: freq_buf,
            };

            channels.push(channel).ok();
            scaler_buf_saved = scaler_buf;
            complex_buf_saved = complex_buf;
        }

        Self { config, channels }
    }

    pub fn decode_frame(
        &mut self,
        num_bits_per_audio_sample: usize, // should be 16
        channel_index: usize,
        buf_in: &[u8],
        samples_out: &mut [i16],
    ) -> Result<(), Lc3DecoderError> {
        if channel_index < NUM_CHANNELS {
            let channel = &mut self.channels[channel_index];
            channel.decode(&self.config, num_bits_per_audio_sample, buf_in, samples_out)
        } else {
            panic!(
                "Cannot decode channel index {} as config only specifies {} channels",
                channel_index, NUM_CHANNELS
            );
        }
    }

    pub const fn calc_working_buffer_lengths(
        frame_duration: FrameDuration,
        sampling_frequency: SamplingFrequency,
    ) -> (usize, usize) {
        let config = Lc3Config::new(sampling_frequency, frame_duration);
        let (scaler_length, complex_length) = DecoderChannel::calc_working_buffer_lengths(&config);
        (NUM_CHANNELS * scaler_length, NUM_CHANNELS * complex_length)
    }
}

#[cfg(feature = "alloc")]
impl<'a> Lc3Decoder<'a> {
    pub fn new(
        num_channels: usize,
        frame_duration: FrameDuration,
        sampling_frequency: SamplingFrequency,
        scaler_buf: &'a mut [Scaler],
        complex_buf: &'a mut [Complex],
    ) -> Self {
        let mut channels = alloc::vec::Vec::new();
        let config = Lc3Config::new(sampling_frequency, frame_duration);
        let mut scaler_buf_saved = scaler_buf;
        let mut complex_buf_saved = complex_buf;

        for _ in 0..num_channels {
            let (x_hat, scaler_buf) = scaler_buf_saved.split_at_mut(config.ne);
            let (packet_loss, scaler_buf) = PacketLossConcealment::new(config.ne, scaler_buf);
            let (mdct, scaler_buf, complex_buf) = ModDiscreteCosTrans::new(config, scaler_buf, complex_buf_saved);
            let (post_filter, scaler_buf) = LongTermPostFilter::new(config, scaler_buf);
            let (freq_buf, scaler_buf) = scaler_buf.split_at_mut(config.nf);

            let channel = DecoderChannel {
                spec_lines: x_hat,
                packet_loss,
                modified_dct: mdct,
                post_filter,
                frame_index: 0,
                freq_samples: freq_buf,
            };

            channels.push(channel);
            scaler_buf_saved = scaler_buf;
            complex_buf_saved = complex_buf;
        }

        Self { config, channels }
    }

    pub fn decode_frame(
        &mut self,
        num_bits_per_audio_sample: usize, // should be 16
        channel_index: usize,
        buf_in: &[u8],
        samples_out: &mut [i16],
    ) -> Result<(), Lc3DecoderError> {
        if channel_index < self.channels.len() {
            let channel = &mut self.channels[channel_index];
            channel.decode(&self.config, num_bits_per_audio_sample, buf_in, samples_out)
        } else {
            panic!(
                "Cannot decode channel index {} as config only specifies {} channels",
                channel_index,
                self.channels.len()
            );
        }
    }

    pub const fn calc_working_buffer_lengths(
        num_channels: usize,
        frame_duration: FrameDuration,
        sampling_frequency: SamplingFrequency,
    ) -> (usize, usize) {
        let config = Lc3Config::new(sampling_frequency, frame_duration);
        let (scaler_length, complex_length) = DecoderChannel::calc_working_buffer_lengths(&config);
        (num_channels * scaler_length, num_channels * complex_length)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::common::config::{FrameDuration, SamplingFrequency};

    #[cfg(not(feature = "alloc"))]
    #[test]
    fn lc3_decode_channel() {
        const num_channels: usize = 1;
        const sampling_frequency: SamplingFrequency = SamplingFrequency::Hz48000;
        const frame_duration: FrameDuration = FrameDuration::TenMs;
        let (scaler_len, complex_len) =
            Lc3Decoder::<num_channels>::calc_working_buffer_lengths(sampling_frequency, frame_duration);
        let mut scaler_buf = [0.0; scaler_len];
        let mut complex_buf = [Complex::default(); complex_len];
        let mut decoder = Lc3Decoder::<1>::new(sampling_frequency, frame_duration, &mut scaler_buf, &mut complex_buf);
        let buf_in = [
            187, 56, 111, 155, 76, 236, 70, 99, 10, 135, 219, 76, 176, 3, 108, 203, 131, 111, 206, 221, 195, 25, 96,
            240, 18, 202, 163, 241, 109, 142, 198, 122, 176, 70, 37, 6, 35, 190, 110, 184, 251, 162, 71, 7, 151, 58,
            42, 79, 200, 192, 99, 157, 234, 156, 245, 43, 84, 64, 167, 32, 52, 106, 43, 75, 4, 102, 213, 123, 168, 120,
            213, 252, 208, 118, 78, 115, 154, 158, 157, 26, 152, 231, 121, 146, 203, 11, 169, 227, 75, 154, 237, 154,
            227, 145, 196, 182, 207, 94, 95, 26, 184, 248, 1, 118, 72, 47, 18, 205, 56, 96, 195, 139, 216, 240, 113,
            233, 44, 198, 245, 157, 139, 70, 162, 182, 139, 136, 165, 68, 79, 247, 161, 126, 17, 135, 36, 30, 229, 24,
            196, 2, 5, 65, 111, 80, 124, 168, 70, 156, 198, 60,
        ];
        let mut samples_out = [0; 480];

        decoder.decode_frame(16, 0, &buf_in, &mut samples_out).unwrap();

        let samples_out_expected = [
            0, 1, 1, 1, 0, -1, -2, -3, -5, -7, -9, -10, -11, -11, -9, -6, -1, 4, 13, 24, 37, 53, 71, 91, 107, 122, 138,
            148, 153, 153, 148, 140, 128, 114, 102, 90, 82, 80, 80, 86, 97, 108, 121, 135, 152, 166, 185, 204, 221,
            245, 263, 276, 291, 287, 271, 251, 212, 161, 100, 33, -34, -108, -181, -250, -310, -351, -391, -416, -413,
            -414, -405, -383, -373, -355, -336, -323, -303, -285, -270, -255, -240, -237, -251, -271, -315, -379, -451,
            -541, -631, -712, -787, -841, -872, -875, -838, -793, -726, -615, -499, -381, -268, -164, -59, 22, 96, 163,
            217, 252, 254, 261, 238, 190, 149, 90, 34, -26, -62, -88, -126, -119, -119, -100, -61, -31, 36, 65, 107,
            153, 146, 185, 194, 180, 192, 185, 195, 188, 196, 213, 210, 226, 246, 291, 325, 371, 437, 466, 534, 586,
            609, 658, 663, 660, 629, 547, 471, 336, 143, -59, -288, -532, -760, -983, -1161, -1289, -1401, -1453,
            -1463, -1431, -1346, -1249, -1109, -952, -786, -607, -451, -292, -165, -51, 58, 117, 192, 236, 233, 241,
            225, 214, 199, 177, 186, 189, 209, 248, 263, 289, 312, 322, 327, 299, 257, 201, 125, 38, -72, -174, -269,
            -366, -454, -521, -555, -593, -591, -554, -534, -466, -399, -338, -250, -189, -108, -40, 19, 115, 197, 289,
            380, 484, 596, 678, 772, 850, 914, 969, 994, 1033, 1071, 1117, 1174, 1231, 1291, 1344, 1401, 1429, 1449,
            1456, 1408, 1357, 1255, 1102, 950, 752, 531, 301, 52, -183, -416, -636, -818, -971, -1092, -1171, -1205,
            -1190, -1128, -1036, -909, -754, -594, -420, -233, -67, 75, 203, 300, 347, 370, 363, 291, 206, 97, -45,
            -158, -296, -434, -536, -644, -705, -740, -771, -776, -796, -811, -839, -879, -876, -834, -780, -726, -671,
            -654, -665, -697, -750, -770, -807, -841, -821, -800, -727, -592, -408, -131, 163, 431, 649, 789, 928,
            1073, 1220, 1405, 1543, 1615, 1646, 1599, 1525, 1449, 1391, 1392, 1453, 1574, 1663, 1714, 1716, 1657, 1665,
            1680, 1691, 1745, 1772, 1795, 1788, 1766, 1721, 1622, 1541, 1433, 1335, 1264, 1200, 1204, 1200, 1216, 1266,
            1291, 1364, 1484, 1626, 1779, 1943, 2096, 2211, 2333, 2450, 2541, 2613, 2656, 2664, 2680, 2713, 2737, 2807,
            2839, 2815, 2778, 2662, 2564, 2475, 2405, 2441, 2508, 2646, 2776, 2832, 2869, 2843, 2814, 2789, 2786, 2815,
            2829, 2899, 2963, 2994, 3007, 2944, 2846, 2728, 2625, 2556, 2495, 2479, 2459, 2399, 2285, 2089, 1879, 1689,
            1533, 1498, 1556, 1667, 1817, 1929, 2020, 2052, 2031, 2026, 1927, 1769, 1546, 1234, 947, 633, 349, 107,
            -156, -372, -563, -698, -784, -830, -828, -870, -934, -1060, -1257, -1491, -1747, -1956, -2120, -2175,
            -2178, -2164, -2072, -1994, -1873, -1727, -1603, -1451, -1341, -1245, -1193, -1172, -1139, -1138, -1080,
            -980, -853, -684, -529, -397, -328, -312, -387, -564, -784, -1066, -1359, -1629, -1854, -2020, -2164,
            -2266, -2337, -2388, -2406, -2382, -2338, -2307, -2263, -2233,
        ];
        assert_eq!(samples_out, samples_out_expected);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn lc3_decode_channel() {
        const NUM_CH: usize = 1;
        const FREQ: SamplingFrequency = SamplingFrequency::Hz48000;
        const DURATION: FrameDuration = FrameDuration::TenMs;
        const SCALER_COMPLEX_LENS: (usize, usize) = Lc3Decoder::calc_working_buffer_lengths(1, DURATION, FREQ);

        let mut scaler_buf = [0.0; SCALER_COMPLEX_LENS.0];
        let mut complex_buf = [Complex::default(); SCALER_COMPLEX_LENS.1];
        let mut decoder = Lc3Decoder::new(NUM_CH, DURATION, FREQ, &mut scaler_buf, &mut complex_buf);
        let buf_in = [
            187, 56, 111, 155, 76, 236, 70, 99, 10, 135, 219, 76, 176, 3, 108, 203, 131, 111, 206, 221, 195, 25, 96,
            240, 18, 202, 163, 241, 109, 142, 198, 122, 176, 70, 37, 6, 35, 190, 110, 184, 251, 162, 71, 7, 151, 58,
            42, 79, 200, 192, 99, 157, 234, 156, 245, 43, 84, 64, 167, 32, 52, 106, 43, 75, 4, 102, 213, 123, 168, 120,
            213, 252, 208, 118, 78, 115, 154, 158, 157, 26, 152, 231, 121, 146, 203, 11, 169, 227, 75, 154, 237, 154,
            227, 145, 196, 182, 207, 94, 95, 26, 184, 248, 1, 118, 72, 47, 18, 205, 56, 96, 195, 139, 216, 240, 113,
            233, 44, 198, 245, 157, 139, 70, 162, 182, 139, 136, 165, 68, 79, 247, 161, 126, 17, 135, 36, 30, 229, 24,
            196, 2, 5, 65, 111, 80, 124, 168, 70, 156, 198, 60,
        ];
        let mut samples_out = [0; 480];

        decoder.decode_frame(16, 0, &buf_in, &mut samples_out).unwrap();

        let samples_out_expected = [
            0, 1, 1, 1, 0, -1, -2, -3, -5, -7, -9, -10, -11, -11, -9, -6, -1, 4, 13, 24, 37, 53, 71, 91, 107, 122, 138,
            148, 153, 153, 148, 140, 128, 114, 102, 90, 82, 80, 80, 86, 97, 108, 121, 135, 152, 166, 185, 204, 221,
            245, 263, 276, 291, 287, 271, 251, 212, 161, 100, 33, -34, -108, -181, -250, -310, -351, -391, -416, -413,
            -414, -405, -383, -373, -355, -336, -323, -303, -285, -270, -255, -240, -237, -251, -271, -315, -379, -451,
            -541, -631, -712, -787, -841, -872, -875, -838, -793, -726, -615, -499, -381, -268, -164, -59, 22, 96, 163,
            217, 252, 254, 261, 238, 190, 149, 90, 34, -26, -62, -88, -126, -119, -119, -100, -61, -31, 36, 65, 107,
            153, 146, 185, 194, 180, 192, 185, 195, 188, 196, 213, 210, 226, 246, 291, 325, 371, 437, 466, 534, 586,
            609, 658, 663, 660, 629, 547, 471, 336, 143, -59, -288, -532, -760, -983, -1161, -1289, -1401, -1453,
            -1463, -1431, -1346, -1249, -1109, -952, -786, -607, -451, -292, -165, -51, 58, 117, 192, 236, 233, 241,
            225, 214, 199, 177, 186, 189, 209, 248, 263, 289, 312, 322, 327, 299, 257, 201, 125, 38, -72, -174, -269,
            -366, -454, -521, -555, -593, -591, -554, -534, -466, -399, -338, -250, -189, -108, -40, 19, 115, 197, 289,
            380, 484, 596, 678, 772, 850, 914, 969, 994, 1033, 1071, 1117, 1174, 1231, 1291, 1344, 1401, 1429, 1449,
            1456, 1408, 1357, 1255, 1102, 950, 752, 531, 301, 52, -183, -416, -636, -818, -971, -1092, -1171, -1205,
            -1190, -1128, -1036, -909, -754, -594, -420, -233, -67, 75, 203, 300, 347, 370, 363, 291, 206, 97, -45,
            -158, -296, -434, -536, -644, -705, -740, -771, -776, -796, -811, -839, -879, -876, -834, -780, -726, -671,
            -654, -665, -697, -750, -770, -807, -841, -821, -800, -727, -592, -408, -131, 163, 431, 649, 789, 928,
            1073, 1220, 1405, 1543, 1615, 1646, 1599, 1525, 1449, 1391, 1392, 1453, 1574, 1663, 1714, 1716, 1657, 1665,
            1680, 1691, 1745, 1772, 1795, 1788, 1766, 1721, 1622, 1541, 1433, 1335, 1264, 1200, 1204, 1200, 1216, 1266,
            1291, 1364, 1484, 1626, 1779, 1943, 2096, 2211, 2333, 2450, 2541, 2613, 2656, 2664, 2680, 2713, 2737, 2807,
            2839, 2815, 2778, 2662, 2564, 2475, 2405, 2441, 2508, 2646, 2776, 2832, 2869, 2843, 2814, 2789, 2786, 2815,
            2829, 2899, 2963, 2994, 3007, 2944, 2846, 2728, 2625, 2556, 2495, 2479, 2459, 2399, 2285, 2089, 1879, 1689,
            1533, 1498, 1556, 1667, 1817, 1929, 2020, 2052, 2031, 2026, 1927, 1769, 1546, 1234, 947, 633, 349, 107,
            -156, -372, -563, -698, -784, -830, -828, -870, -934, -1060, -1257, -1491, -1747, -1956, -2120, -2175,
            -2178, -2164, -2072, -1994, -1873, -1727, -1603, -1451, -1341, -1245, -1193, -1172, -1139, -1138, -1080,
            -980, -853, -684, -529, -397, -328, -312, -387, -564, -784, -1066, -1359, -1629, -1854, -2020, -2164,
            -2266, -2337, -2388, -2406, -2382, -2338, -2307, -2263, -2233,
        ];
        assert_eq!(samples_out, samples_out_expected);
    }
}
