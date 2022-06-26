#![allow(unused_assignments)]
use byteorder::{ByteOrder, LittleEndian};
use lc3_codec::common::wav::{self, WavError, FULL_WAV_HEADER_LEN};
use lc3_codec::decoder::lc3_decoder::Lc3DecoderError;
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

fn main() -> Result<(), MainError> {
    SimpleLogger::new().init().unwrap();
    info!("Wav creation started");

    let wav_file_name = "demo.wav";
    let mut new_wav_file = File::create(wav_file_name)?;
    let bits_per_sample = 16;
    let num_channels = 1;
    let fs = 48000;

    const SIGNAL_LEN: usize = 32;
    const NUM_SIGNALS: usize = 3750;

    let data_size = SIGNAL_LEN * NUM_SIGNALS * 2; // 2 bytes per sample

    let wav_header = wav::WavHeader {
        num_channels,
        sample_rate: fs,
        byte_rate: fs as usize * num_channels * bits_per_sample as usize / 8,
        block_align: 4,
        bits_per_sample,
        data_size,
        data_start_position: FULL_WAV_HEADER_LEN,
        data_with_header_size: data_size + FULL_WAV_HEADER_LEN,
    };

    let mut signal = [0i16; SIGNAL_LEN];
    for (x, sample) in signal.iter_mut().enumerate() {
        *sample = triangle_wave(x as i32, SIGNAL_LEN, u16::MAX as i32 - 1, 0, 1) as i16;
    }

    let mut wav_header_buffer = vec![0; 256];
    let len = wav::write_header(&wav_header, &mut wav_header_buffer)?;
    new_wav_file.write_all(&wav_header_buffer[..len])?;

    let mut buf_out = vec![0; SIGNAL_LEN * 2];
    LittleEndian::write_i16_into(&signal, &mut buf_out);
    let quiet_out = vec![0; SIGNAL_LEN * 2];

    /*
    // 100ms
    for _ in 0..25 {
        new_wav_file.write_all(&buf_out)?;
        for _ in 0..149 {
            new_wav_file.write_all(&quiet_out)?;
        }
    }
    // 10ms
    for _ in 0..250 {
        new_wav_file.write_all(&buf_out)?;
        for _ in 0..14 {
            new_wav_file.write_all(&quiet_out)?;
        }
    }


    // 24ms
    for _ in 0..100 {
        new_wav_file.write_all(&buf_out)?;
        for _ in 0..35 {
            new_wav_file.write_all(&quiet_out)?;
        }
    }*/

    // 7.5ms
    for _ in 0..375 {
        new_wav_file.write_all(&buf_out)?;
        for _ in 0..9 {
            new_wav_file.write_all(&quiet_out)?;
        }
    }

    Ok(())
}

fn triangle_wave(x: i32, length: usize, amplitude: i32, phase: i32, periods: i32) -> i32 {
    let length = length as i32;
    amplitude
        - ((2 * periods * (x + phase + length / (4 * periods)) * amplitude / length) % (2 * amplitude) - amplitude)
            .abs()
        - amplitude / 2
}
