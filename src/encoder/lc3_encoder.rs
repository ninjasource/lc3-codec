// Copyright 2022 David Haig
// Licensed under the Apache License, Version 2.0 (the "License");

use super::{
    attack_detector::AttackDetector, bandwidth_detector::BandwidthDetector, bitstream_encoding::BitstreamEncoding,
    long_term_post_filter::LongTermPostFilter, modified_dct::ModDiscreteCosTrans,
    noise_level_estimation::NoiseLevelEstimation, residual_spectrum::ResidualBitsEncoder,
    spectral_noise_shaping::SpectralNoiseShaping, spectral_quantization::SpectralQuantization,
    temporal_noise_shaping::TemporalNoiseShaping,
};
use crate::common::{
    complex::{Complex, Scaler},
    config::Lc3Config,
};
use heapless::Vec;

#[allow(unused_imports)]
use num_traits::real::Real;

/// Main entry point of library - Start here for the Encoder

#[derive(Debug)]
pub enum Lc3EncoderError {}

pub struct Lc3Encoder<'a, const NUM_CHANNELS: usize> {
    config: Lc3Config,
    channels: Vec<EncoderChannel<'a>, NUM_CHANNELS>,
}

struct EncoderChannel<'a> {
    config: Lc3Config,
    mdct: ModDiscreteCosTrans<'a>,
    bandwidth_detector: BandwidthDetector,
    attack_detector: AttackDetector,
    spectral_noise_shaping: SpectralNoiseShaping,
    temporal_noise_shaping: TemporalNoiseShaping,
    long_term_post_filter: LongTermPostFilter<'a>,
    spectral_quantization: SpectralQuantization,
    noise_level_estimation: NoiseLevelEstimation,
    bitstream_encoding: BitstreamEncoding,
    frame_index: usize,
    energy_bands: &'a mut [Scaler],
    output: &'a mut [Scaler],
    x_q: &'a mut [i16],
    residual: ResidualBitsEncoder,
}

impl<'a> EncoderChannel<'a> {
    pub fn encode(&mut self, x_s: &[i16], buf_out: &mut [u8]) -> Result<(), Lc3EncoderError> {
        self.frame_index += 1;
        let nbits = buf_out.len() * 8;

        // modified discrete cosine transform (mutates output and energy_bands)
        let near_nyquist_flag = self.mdct.run(x_s, self.output, self.energy_bands);

        // bandwidth detector
        let (bandwidth_ind, nbits_bandwidth) = self.bandwidth_detector.run(self.energy_bands);

        // attack detector
        let attack_detected = self.attack_detector.run(x_s, buf_out.len());

        // spectral noise shaping
        let sns = self
            .spectral_noise_shaping
            .run(self.output, self.energy_bands, attack_detected);

        // temporal noise shaping
        let tns = self
            .temporal_noise_shaping
            .run(sns.output, bandwidth_ind, nbits, near_nyquist_flag);

        // long term post filter
        let post_filter = self.long_term_post_filter.run(x_s, near_nyquist_flag, nbits);

        // spectral quantization
        let spec = self.spectral_quantization.run(
            tns.output,
            self.x_q,
            nbits,
            nbits_bandwidth,
            tns.nbits_tns,
            post_filter.nbits_ltpf,
        );

        // residual bits
        let residual_bits = self.residual.encode(
            spec.nbits_spec,
            spec.nbits_trunc,
            self.config.ne,
            spec.gg,
            tns.output,
            spec.output,
        );

        // noise level estimation
        let noise_factor =
            self.noise_level_estimation
                .calc_noise_factor(tns.output, spec.output, bandwidth_ind, spec.gg as Scaler);

        // bitstream encoding
        self.bitstream_encoding.init(buf_out);

        // side information
        self.bitstream_encoding
            .bandwidth(bandwidth_ind, self.bandwidth_detector.get_num_bits_bandwidth(), buf_out);
        self.bitstream_encoding.last_non_zero_tuple(spec.lastnz_trunc, buf_out);
        self.bitstream_encoding.lsb_mode_bit(spec.lsb_mode, buf_out);
        self.bitstream_encoding.global_gain(spec.gg_ind as usize, buf_out);
        self.bitstream_encoding
            .tns_activation_flag(tns.num_tns_filters, &tns.rc_order, buf_out);
        self.bitstream_encoding
            .pitch_present_flag(post_filter.pitch_present, buf_out);

        // encode SCF VQ parameters - 1st stage (10 bits)
        self.bitstream_encoding
            .encode_scf_vq_1st_stage(sns.ind_lf, sns.ind_hf, buf_out);

        // encode SCF VQ parameters - 2nd stage side-info (3-4 bits) and MPVQ data
        self.bitstream_encoding.encode_scf_vq_2nd_stage(
            sns.shape_j,
            sns.gind,
            sns.ls_inda as usize,
            sns.index_joint_j,
            buf_out,
        );

        if post_filter.pitch_present {
            self.bitstream_encoding
                .ltpf_data(post_filter.ltpf_active, post_filter.pitch_index, buf_out);
        }

        self.bitstream_encoding.noise_factor(noise_factor, buf_out);

        // arithmetic encoding
        self.bitstream_encoding.ac_enc_init();

        // tns data
        self.bitstream_encoding.tns_data(
            tns.lpc_weighting,
            tns.num_tns_filters,
            &tns.rc_order,
            &tns.rc_i,
            buf_out,
        );

        // spectral data
        // TODO (theirs) check whether the degenerated case with nlsbs==0 works
        self.bitstream_encoding.spectral_data(
            spec.lastnz_trunc,
            spec.rate_flag,
            spec.lsb_mode,
            spec.output,
            spec.nbits_lsb,
            buf_out,
        );

        // residual data and finalization
        self.bitstream_encoding
            .residual_data_and_finalization(spec.lsb_mode, residual_bits, buf_out);

        Ok(())
    }
}

impl<'a, const NUM_CHANNELS: usize> Lc3Encoder<'a, NUM_CHANNELS> {
    pub fn new(
        config: Lc3Config,
        integer_buf: &'a mut [i16],
        scaler_buf: &'a mut [Scaler],
        complex_buf: &'a mut [Complex],
    ) -> Self {
        let mut channels: Vec<EncoderChannel<'a>, NUM_CHANNELS> = Vec::new();

        let mut integer_buf_save = integer_buf;
        let mut scaler_buf_save = scaler_buf;
        let mut complex_buf_save = complex_buf;

        for _ in 0..config.nc {
            let (mdct, integer_buf, complex_buf) = ModDiscreteCosTrans::new(config, integer_buf_save, complex_buf_save);
            let bandwidth_detector = BandwidthDetector::new(config.n_ms, config.fs_ind);
            let attack_detector = AttackDetector::new(config);
            let spectral_noise_shaping = SpectralNoiseShaping::new(config, mdct.get_i_fs());
            let temporal_noise_shaping = TemporalNoiseShaping::new(config);
            let (long_term_post_filter, scaler_buf, integer_buf) =
                LongTermPostFilter::new(config, scaler_buf_save, integer_buf);
            let spectral_quantization = SpectralQuantization::new(config.ne, config.fs_ind);
            let noise_level_estimation = NoiseLevelEstimation::new(config.n_ms, config.ne);
            let bitstream_encoding = BitstreamEncoding::new(config.ne);
            let (output, scaler_buf) = scaler_buf.split_at_mut(config.nf);
            let (energy_bands, scaler_buf) = scaler_buf.split_at_mut(config.nb);
            let residual = ResidualBitsEncoder::default();
            let (x_q, integer_buf) = integer_buf.split_at_mut(config.ne);

            let channel = EncoderChannel {
                mdct,
                bandwidth_detector,
                attack_detector,
                spectral_noise_shaping,
                temporal_noise_shaping,
                long_term_post_filter,
                spectral_quantization,
                noise_level_estimation,
                bitstream_encoding,
                frame_index: 0,
                output,
                energy_bands,
                residual,
                config,
                x_q,
            };

            channels.push(channel).ok();
            integer_buf_save = integer_buf;
            scaler_buf_save = scaler_buf;
            complex_buf_save = complex_buf;
        }

        Self { config, channels }
    }

    pub fn encode_frame(
        &mut self,
        channel_index: usize,
        samples_in: &[i16],
        buf_out: &mut [u8],
    ) -> Result<(), Lc3EncoderError> {
        if channel_index < self.config.nc {
            let channel = &mut self.channels[channel_index];
            channel.encode(samples_in, buf_out)
        } else {
            panic!(
                "Cannot decode channel index {} as config only specifies {} channels",
                channel_index, self.config.nc
            );
        }
    }

    pub fn calc_working_buffer_lengths(config: &Lc3Config) -> (usize, usize, usize) {
        let (mdct_integer_len, mdct_complex_len) = ModDiscreteCosTrans::calc_working_buffer_lengths(config);
        let (ltpf_integer_len, ltpf_scaler_len) = LongTermPostFilter::calc_working_buffer_length(config);
        let scaler_len = ltpf_scaler_len + config.nf + config.nb;
        let integer_len = mdct_integer_len + ltpf_integer_len + config.ne;
        (
            integer_len * config.nc,
            scaler_len * config.nc,
            mdct_complex_len * config.nc,
        )
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::common::config::{FrameDuration, SamplingFrequency};

    #[test]
    fn lc3_encode_channel() {
        let config = Lc3Config::new(SamplingFrequency::Hz48000, FrameDuration::TenMs, 1);
        let mut integer_buf = [0; 1900];
        let mut scaler_buf = [0.0; 2034];
        let mut complex_buf = [Complex::new(0.0, 0.0); 960];

        let mut encoder = Lc3Encoder::<1>::new(config, &mut integer_buf, &mut scaler_buf, &mut complex_buf);
        let samples_in = [
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
        let mut buf_out = [0; 150];

        encoder.encode_frame(0, &samples_in, &mut buf_out).unwrap();

        let buf_out_expected = [
            230, 243, 160, 169, 152, 75, 36, 156, 223, 96, 241, 214, 150, 248, 180, 106, 115, 92, 147, 213, 56, 100,
            96, 52, 194, 178, 44, 31, 222, 246, 83, 116, 240, 220, 40, 241, 82, 228, 209, 57, 128, 152, 9, 144, 112,
            249, 48, 46, 135, 182, 250, 59, 135, 221, 129, 46, 204, 178, 232, 100, 172, 27, 177, 120, 86, 253, 35, 137,
            19, 253, 191, 202, 97, 240, 10, 45, 124, 110, 234, 149, 49, 115, 209, 177, 153, 231, 93, 211, 214, 19, 127,
            143, 103, 47, 239, 86, 73, 91, 231, 94, 248, 143, 54, 54, 190, 51, 47, 136, 92, 157, 13, 226, 13, 96, 104,
            159, 17, 206, 66, 25, 157, 51, 5, 252, 166, 135, 213, 118, 107, 152, 226, 253, 51, 136, 74, 186, 52, 64,
            236, 152, 115, 0, 29, 23, 247, 3, 20, 124, 21, 116,
        ];
        assert_eq!(buf_out, buf_out_expected);
    }
}
