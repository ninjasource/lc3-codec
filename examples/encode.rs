#![allow(unused_assignments)]
use byteorder::{ByteOrder, LittleEndian};
use lc3_codec::common::{
    complex::Complex,
    config::{FrameDuration, Lc3Config, SamplingFrequency},
    wav::{self, WavError},
};
use lc3_codec::encoder::lc3_encoder::{Lc3Encoder, Lc3EncoderError};
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
    Lc3Encoder(usize, Lc3EncoderError),
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

fn encode_wav_to_lc3(
    wav_file_name: &str,
    lc3_file_name: &str,
    sampling_frequency: SamplingFrequency,
    num_bits_per_audio_sample: usize,
    num_channels: usize,
    frame_duration: FrameDuration,
    num_bytes_per_channel: usize,
) -> Result<(), MainError> {
    let mut file_in = File::open(wav_file_name)?;
    let mut buf_in_full = Vec::new();
    file_in.read_to_end(&mut buf_in_full)?;
    info!("Read {} bytes from file", buf_in_full.len());

    let mut file_out = File::create(lc3_file_name)?;

    let wav_header = wav::read_header(&buf_in_full)?;

    let (integer_length, scaler_length, complex_length) =
        Lc3Encoder::calc_working_buffer_lengths(num_channels, frame_duration, sampling_frequency);
    let mut integer_buf = vec![0; integer_length];
    let mut scaler_buf = vec![0.0; scaler_length];
    let mut complex_buf = vec![Complex::default(); complex_length];

    let config = Lc3Config::new(sampling_frequency, frame_duration);
    let bytes_per_frame = config.nf * num_channels * num_bits_per_audio_sample / 8;

    let mut encoder = Lc3Encoder::new(
        num_channels, frame_duration, sampling_frequency, &mut integer_buf, &mut scaler_buf, &mut complex_buf,
    );

    let mut samples_in_temp = vec![0; config.nf * num_channels];
    let mut samples_in = vec![0; config.nf * num_channels];

    let mut buf_out = vec![0; num_bytes_per_channel];

    let mut frame_index = 0;
    for cursor_in in (wav_header.data_start_position..buf_in_full.len()).step_by(bytes_per_frame) {
        frame_index += 1;

        // One log message for every second of audio
        if frame_index % 1000 == 0 {
            info!("Encoding frame {}", frame_index);
        }

        let num_bytes_to_read = (buf_in_full.len() - cursor_in).min(bytes_per_frame);
        let num_samples_to_read = num_bytes_to_read / 2;

        LittleEndian::read_i16_into(
            &buf_in_full[cursor_in..cursor_in + num_bytes_to_read],
            &mut samples_in_temp[..(num_bytes_to_read / 2)],
        );

        // zero the rest of the bytes
        if num_bytes_to_read < bytes_per_frame {
            for x in samples_in_temp.iter_mut().skip(num_samples_to_read) {
                *x = 0;
            }
        }

        // reshuffle samples so that all samples for a channel are contiguous
        for ch in 0..num_channels {
            for i in 0..config.nf {
                let in_index = i * num_channels + ch;
                let out_index = config.nf * ch + i;
                samples_in[out_index] = samples_in_temp[in_index];
            }
        }

        for ch in 0..num_channels {
            encoder
                .encode_frame(
                    ch,
                    &samples_in[ch * config.nf..ch * config.nf + config.nf],
                    &mut buf_out,
                )
                .map_err(|e| MainError::Lc3Encoder(cursor_in, e))?;

            file_out.write_all(&buf_out)?;
        }
    }

    file_out.flush()?;
    Ok(())
}

fn main() -> Result<(), MainError> {
    SimpleLogger::new().init().unwrap();
    info!("LC3 codec started");

    /*
        encode_wav_to_lc3(
            "./audio_samples/48khz_16bit_mono_10ms_150byte_tones.wav",
            "./audio_samples/48khz_16bit_mono_10ms_150byte_tones_new.lc3",
            SamplingFrequency::Hz48000,
            16,
            1,
            FrameDuration::TenMs,
            150,
        )?;

        encode_wav_to_lc3(
            "./audio_samples/48khz_16bit_stereo_10ms_150byte_electronic.wav",
            "./audio_samples/48khz_16bit_stereo_10ms_150byte_electronic_new.lc3",
            SamplingFrequency::Hz48000,
            16,
            2,
            FrameDuration::TenMs,
            150,
        )?;

        encode_wav_to_lc3(
            "./audio_samples/48khz_16bit_stereo_10ms_150byte_piano.wav",
            "./audio_samples/48khz_16bit_stereo_10ms_150byte_piano_new.lc3",
            SamplingFrequency::Hz48000,
            16,
            2,
            FrameDuration::TenMs,
            150,
        )?;
    */

    encode_wav_to_lc3(
        "/home/david/Music/out_short_new.wav",
        "/home/david/Music/out_short_new_small.lc3",
        SamplingFrequency::Hz48000,
        16,
        2,
        FrameDuration::TenMs,
        70,
    )?;
    Ok(())
}
