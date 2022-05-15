use std::{fs::File, io::Read};

use log::info;
use simple_logger::SimpleLogger;

fn main() -> Result<(), std::io::Error> {
    SimpleLogger::new().init().unwrap();

    let mut file_left = File::open("./audio_samples/48khz_16bit_mono_10ms_150byte_tones.lc3")?;
    let mut file_right = File::open("./audio_samples/48khz_16bit_mono_10ms_150byte_tones_new.lc3")?;

    let mut buf_left = vec![0; 150];
    let mut buf_right = vec![0; 150];

    let mut frame_index = 0;
    loop {
        frame_index += 1;

        let len_left = file_left.read(&mut buf_left)?;
        let len_right = file_right.read(&mut buf_right)?;

        if len_left != len_right || len_left == 0 {
            info!("Completed comparing. len_left: {len_left} len_right: {len_right}");
            break;
        }

        for (i, (left, right)) in buf_left.iter().zip(&buf_right).enumerate() {
            if left != right {
                info!("Diff at frame {frame_index} byte index {i}: left: {left} right: {right}");
                return Ok(());
            }
        }
    }

    Ok(())
}
