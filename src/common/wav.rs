// TODO: move this to its own crate as it is incomplete and does not belong here

use byteorder::{ByteOrder, LittleEndian};
use core::str;

// see http://tiny.systems/software/soundProgrammer/WavFormatDocs.pdf

#[derive(Debug)]
pub enum WavError {
    WriteHeaderBufferTooSmall,
    ReadHeaderInvalidHeaderLength,
    ReadHeaderChunkIdNotRIFF,
    ReadHeaderFormatNotWAVE,
    ReadHeaderSubChunk1IdNotFmt,
    ReadHeaderInvalidPcmHeaderLength,
    ReadHeaderAudioFormatNotPcm,
    ReadHeaderMissingDataSection,
}

pub const RIFF_HEADER_ONLY_LEN: usize = 8;
pub const FULL_WAV_HEADER_LEN: usize = 44;

const PCM_HEADER_LENGTH: u32 = 16;
const AUDIO_FORMAT_PCM: u16 = 1;

#[derive(Debug)]
pub struct WavHeader {
    // chunk_id is always "RIFF"
    // chunk_size: u32, //  This is the size of the entire file in bytes minus 8 bytes for the two fields not included in this count
    // format is always "WAVE"
    // subchunk1_id is always "fmt "
    // subchunk1_size: u32, // 16 for PCM
    // audio_format: u16, // 1 for PCM
    pub num_channels: usize,    // Mono = 1, Stereo = 2, etc.
    pub sample_rate: usize,     // e.g. 44100
    pub byte_rate: usize,       // SampleRate * NumChannels * BitsPerSample/8
    pub block_align: usize,     // NumChannels * BitsPerSample/8
    pub bits_per_sample: usize, // 8 bits = 8, 16 bits = 16, etc.
    // subchunk2_id is always "data"
    pub data_size: usize,             // NumSamples * NumChannels * BitsPerSample/8
    pub data_start_position: usize,   // the position of the first byte of data
    pub data_with_header_size: usize, // num bytes of the entire file excluding the first 8 bytes
}

pub fn write_header(header: &WavHeader, buf: &mut [u8]) -> Result<usize, WavError> {
    if buf.len() < FULL_WAV_HEADER_LEN {
        return Err(WavError::WriteHeaderBufferTooSmall);
    }

    buf[..4].copy_from_slice(b"RIFF"); // chunk ID
    LittleEndian::write_u32(&mut buf[4..8], header.data_with_header_size as u32); // length of entire file below this line
    buf[8..12].copy_from_slice(b"WAVE"); // format
    buf[12..16].copy_from_slice(b"fmt "); // subchunk1 ID
    LittleEndian::write_u32(&mut buf[16..20], PCM_HEADER_LENGTH); // pcm header length - subchunk1 size
    LittleEndian::write_u16(&mut buf[20..22], AUDIO_FORMAT_PCM); // pcm audio format - 1
    LittleEndian::write_u16(&mut buf[22..24], header.num_channels as u16);
    LittleEndian::write_u32(&mut buf[24..28], header.sample_rate as u32);
    LittleEndian::write_u32(&mut buf[28..32], header.byte_rate as u32);
    LittleEndian::write_u16(&mut buf[32..34], header.block_align as u16);
    LittleEndian::write_u16(&mut buf[34..36], header.bits_per_sample as u16);
    buf[36..40].copy_from_slice(b"data"); // subchunk2 ID
    LittleEndian::write_u32(&mut buf[40..44], header.data_size as u32);

    Ok(FULL_WAV_HEADER_LEN)
}

pub fn read_header(buf: &[u8]) -> Result<WavHeader, WavError> {
    if buf.len() < FULL_WAV_HEADER_LEN {
        return Err(WavError::ReadHeaderInvalidHeaderLength);
    }

    if str::from_utf8(&buf[..4]).map_err(|_| WavError::ReadHeaderChunkIdNotRIFF)? != "RIFF" {
        return Err(WavError::ReadHeaderChunkIdNotRIFF);
    }

    if str::from_utf8(&buf[8..12]).map_err(|_| WavError::ReadHeaderFormatNotWAVE)? != "WAVE" {
        return Err(WavError::ReadHeaderFormatNotWAVE);
    }

    if str::from_utf8(&buf[12..16]).map_err(|_| WavError::ReadHeaderSubChunk1IdNotFmt)? != "fmt " {
        return Err(WavError::ReadHeaderSubChunk1IdNotFmt);
    }

    if LittleEndian::read_u32(&buf[16..20]) != 16 {
        return Err(WavError::ReadHeaderInvalidPcmHeaderLength);
    }

    if LittleEndian::read_u16(&buf[20..22]) != 1 {
        return Err(WavError::ReadHeaderAudioFormatNotPcm);
    }

    let data_with_header_size = LittleEndian::read_u32(&buf[4..8]) as usize;
    let num_channels = LittleEndian::read_u16(&buf[22..24]) as usize;
    let sample_rate = LittleEndian::read_u32(&buf[24..28]) as usize;
    let byte_rate = LittleEndian::read_u32(&buf[28..32]) as usize;
    let block_align = LittleEndian::read_u16(&buf[32..34]) as usize;
    let bits_per_sample = LittleEndian::read_u16(&buf[34..36]) as usize;

    let section = str::from_utf8(&buf[36..40]).map_err(|_| WavError::ReadHeaderMissingDataSection)?;
    let (data_size, data_start_position) = match section {
        "data" => (LittleEndian::read_u32(&buf[40..44]) as usize, FULL_WAV_HEADER_LEN),
        "LIST" => {
            let size = LittleEndian::read_u32(&buf[40..44]) as usize;
            LittleEndian::read_u32(&buf[40..44]); // list type id (ignore)
            (size, FULL_WAV_HEADER_LEN + 4)
        }
        _section => {
            //  warn!("Unknown section: {}", section);
            return Err(WavError::ReadHeaderMissingDataSection);
        }
    };

    Ok(WavHeader {
        data_with_header_size,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        data_size,
        data_start_position,
    })
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn can_read_pcm_wav_header() {
        let buffer = [
            0x52, 0x49, 0x46, 0x46, 0x16, 0x29, 0x0B, 0x00, 0x57, 0x41, 0x56, 0x45, 0x66, 0x6D, 0x74, 0x20, 0x10, 0x00,
            0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x44, 0xAC, 0x00, 0x00, 0x10, 0xB1, 0x02, 0x00, 0x04, 0x00, 0x10, 0x00,
            0x64, 0x61, 0x74, 0x61, 0x70, 0x28, 0x0B, 0x00, 0x00,
        ];

        let header = read_header(&buffer).unwrap();

        assert_eq!(header.num_channels, 2);
        assert_eq!(header.sample_rate, 44100);
        assert_eq!(header.byte_rate, 176400);
        assert_eq!(header.block_align, 4);
        assert_eq!(header.bits_per_sample, 16);
        assert_eq!(header.data_size, 731248);
        assert_eq!(header.data_start_position, 44);
    }
}
