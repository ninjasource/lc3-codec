use lc3_codec::common::config::{FrameDuration, Lc3Config, SamplingFrequency};
use lc3_codec::decoder::side_info_reader::SideInfoError;
use lc3_codec::decoder::{buffer_reader::BufferReader, *};
use simple_logger::SimpleLogger;
use std::fs::File;
use std::io;
use std::io::prelude::*;

#[macro_use]
extern crate log;

#[derive(Debug)]
pub enum MainError {
    Io(io::Error),
    SideInfo(usize, SideInfoError),
}

impl From<io::Error> for MainError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

fn main() -> Result<(), MainError> {
    SimpleLogger::new().init().unwrap();
    info!("LC3 codec started");
    let mut file = File::open("./audio_samples/48khz_16bit_mono_10ms_150byte_tones.lc3")?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    info!("Read {} bytes from file", buf.len());

    let config = Lc3Config::new(SamplingFrequency::Hz48000, FrameDuration::TenMs);
    let num_bytes_per_channel = 150;

    let mut cursor = 0;
    loop {
        if cursor >= buf.len() {
            return Ok(());
        }

        let slice = &buf[cursor..cursor + num_bytes_per_channel];
        let mut reader = BufferReader::new();
        let side_info = side_info_reader::read(slice, &mut reader, config.fs_ind, config.ne)
            .map_err(|e| MainError::SideInfo(cursor, e))?;
        println!("{:#?}", &side_info);
        println!();

        cursor += num_bytes_per_channel;
    }
}
