#![allow(unused_assignments)]
use byteorder::{ByteOrder, LittleEndian};
use lc3_codec::common::{
    complex::Complex,
    config::{FrameDuration, Lc3Config, SamplingFrequency},
    wav::{self, WavError, FULL_WAV_HEADER_LEN, RIFF_HEADER_ONLY_LEN},
};
use lc3_codec::decoder::lc3_decoder::{Lc3Decoder, Lc3DecoderError};
use simple_logger::SimpleLogger;
use std::fs::File;
use std::io;
use std::io::prelude::*;

#[macro_use]
extern crate log;

#[derive(Debug)]
pub enum MainError {
    Io(io::Error),
    Wav(WavError),
    Lc3Decoder(usize, Lc3DecoderError),
}

impl From<io::Error> for MainError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<WavError> for MainError {
    fn from(err: WavError) -> Self {
        Self::Wav(err)
    }
}

fn decode_lc3_to_wav(
    lc3_file_name: &str,
    wav_file_name: &str,
    sampling_frequency: SamplingFrequency,
    num_bits_per_audio_sample: usize,
    num_channels: usize,
    frame_duration: FrameDuration,
    num_bytes_per_channel: usize,
) -> Result<(), MainError> {
    let mut file = File::open(lc3_file_name)?;
    let mut buf_in_full = Vec::new();
    file.read_to_end(&mut buf_in_full)?;
    info!("Read {} bytes from file", buf_in_full.len());

    let (scaler_length, complex_length) =
        Lc3Decoder::calc_working_buffer_lengths(num_channels, frame_duration, sampling_frequency);
    let mut scaler_buf = vec![0.0; scaler_length]; // 9942
    let mut complex_buf = vec![Complex::default(); complex_length]; // 1920
    let num_bytes_per_channel = num_bytes_per_channel; // 150 = 240 kbps for 2 channels, 120 kbps for 1 channel
    let mut decoder = Lc3Decoder::new(
        num_channels, frame_duration, sampling_frequency, &mut scaler_buf, &mut complex_buf,
    );

    let mut in_cursor = 0;
    let mut out_cursor = FULL_WAV_HEADER_LEN;

    let config = Lc3Config::new(sampling_frequency, frame_duration);
    let mut samples_out_by_channel = vec![vec![0_i16; config.nf]; num_channels];
    let mut samples_out_interleved = vec![0_i16; num_channels * config.nf];

    let mut new_wav_file = File::create(wav_file_name)?;
    let bits_per_sample = 16;
    let wav_header = wav::WavHeader {
        num_channels,
        sample_rate: config.fs,
        byte_rate: config.fs as usize * num_channels * bits_per_sample as usize / 8,
        block_align: 4,
        bits_per_sample,
        data_size: out_cursor - FULL_WAV_HEADER_LEN,
        data_start_position: FULL_WAV_HEADER_LEN,
        data_with_header_size: out_cursor - RIFF_HEADER_ONLY_LEN,
    };

    let mut wav_header_buffer = vec![0; 256];
    let len = wav::write_header(&wav_header, &mut wav_header_buffer)?;
    new_wav_file.write_all(&wav_header_buffer[..len])?;
    let mut buf_out = vec![0; num_channels * config.nf * num_bits_per_audio_sample / 8];

    let mut frame_index = 0;
    loop {
        frame_index += 1;

        // log 10 seconds of audio
        if frame_index % 1000 == 0 {
            info!("Decoding frame: {}", frame_index);
        }

        for channel_index in 0..num_channels {
            if in_cursor >= buf_in_full.len() {
                return Ok(());
            }

            let to_index = in_cursor + num_bytes_per_channel;
            if to_index >= buf_in_full.len() {
                info!("Decoding frame: {} Complete", frame_index);
                return Ok(());
            }

            let buf_in = &buf_in_full[in_cursor..to_index];

            let samples_out = &mut samples_out_by_channel[channel_index];
            decoder
                .decode_frame(num_bits_per_audio_sample, channel_index, buf_in, samples_out)
                .map_err(|e| MainError::Lc3Decoder(in_cursor, e))?;

            in_cursor += num_bytes_per_channel;
        }

        for i in 0..config.nf {
            for ch in 0..num_channels {
                samples_out_interleved[i * num_channels + ch] = samples_out_by_channel[ch][i];
            }
        }

        LittleEndian::write_i16_into(&samples_out_interleved, &mut buf_out);
        new_wav_file.write_all(&buf_out)?;
        out_cursor += buf_out.len();
    }
}

fn main() -> Result<(), MainError> {
    SimpleLogger::new().init().unwrap();
    info!("LC3 codec started");
    /*
        decode_lc3_to_wav(
            "./audio_samples/48khz_16bit_mono_10ms_150byte_tones_new.lc3",
            "./audio_samples/48khz_16bit_mono_10ms_150byte_tones_new2.wav",
            SamplingFrequency::Hz48000,
            16,
            1,
            FrameDuration::TenMs,
            150,
        )?;
    */
    /*
                    decode_lc3_to_wav(
                        "./audio_samples/48khz_16bit_mono_10ms_150byte_tones.lc3",
                        "./audio_samples/48khz_16bit_mono_10ms_150byte_tones_new.wav",
                        SamplingFrequency::Hz48000,
                        16,
                        1,
                        FrameDuration::TenMs,
                        150,
                    )?;


        decode_lc3_to_wav(
            "./audio_samples/48khz_16bit_stereo_10ms_150byte_electronic.lc3",
            "./audio_samples/48khz_16bit_stereo_10ms_150byte_electronic_new.wav",
            SamplingFrequency::Hz48000,
            16,
            2,
            FrameDuration::TenMs,
            150,
        )?;


    decode_lc3_to_wav(
        "./audio_samples/48khz_16bit_stereo_10ms_150byte_piano_new.lc3",
        "./audio_samples/48khz_16bit_stereo_10ms_150byte_piano_new2.wav",
        SamplingFrequency::Hz48000,
        16,
        2,
        FrameDuration::TenMs,
        150,
    )?;

     */
    decode_lc3_to_wav(
        "/home/david/Music/out_short_new_small.lc3",
        "/home/david/Music/out_short_new_small.wav",
        SamplingFrequency::Hz48000,
        16,
        2,
        FrameDuration::TenMs,
        70,
    )?;

    Ok(())
}
