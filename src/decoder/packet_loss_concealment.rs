use crate::common::{complex::Scaler, config::Lc3Config};

use super::side_info::LongTermPostFilterInfo;

// checked against spec

pub struct PacketLossConcealment<'a> {
    // A copy of the last saved spectral lines
    spec_lines_last_good: &'a mut [Scaler],

    /// Number of consecutive erased frames
    num_lost_frames: usize,

    /// Attenuation factor (fade out speed)
    alpha: Scaler,

    /// Packet loss concealment seed
    plc_seed: usize,

    /// Number of spectral lines (e.g. 400)
    ne: usize,
}

impl<'a> PacketLossConcealment<'a> {
    pub fn new(ne: usize, scaler_buf: &'a mut [Scaler]) -> (Self, &'a mut [Scaler]) {
        let (spec_lines_last_good, scaler_buf) = scaler_buf.split_at_mut(ne);

        (
            Self {
                spec_lines_last_good,
                plc_seed: 24607,
                num_lost_frames: 0,
                alpha: 1.0,
                ne,
            },
            scaler_buf,
        )
    }

    pub fn calc_working_buffer_length(config: &Lc3Config) -> usize {
        config.ne
    }

    /// Saves the last good decoded frame so that we have something to work with
    /// if subsquent frames are corrupted or missing
    ///
    /// # Arguments
    ///
    /// * `spec_lines` - Decoded spectral lines to save
    pub fn save(&mut self, spec_lines: &[Scaler]) {
        self.num_lost_frames = 0;
        self.alpha = 1.0;
        self.spec_lines_last_good[..self.ne].copy_from_slice(&spec_lines[..self.ne]);
    }

    /// Loads the last good frame from cache
    /// Every time we load spectral lines for a frame we add some randomness and lower the volume a little
    ///
    /// # Arguments
    ///
    /// * `spec_lines` - Empty spectral lines to load data into (will ovewrite if data exists)
    //
    pub fn load_into(&mut self, spec_lines: &mut [Scaler]) -> LongTermPostFilterInfo {
        if self.num_lost_frames >= 4 {
            self.alpha *= if self.num_lost_frames < 8 { 0.9 } else { 0.85 };
        }
        self.num_lost_frames += 1;

        for (current, last_good) in spec_lines.iter_mut().zip(self.spec_lines_last_good.iter()) {
            self.plc_seed = (16831 + self.plc_seed * 12821) & 0xFFFF;

            *current = if self.plc_seed < 0x8000 {
                last_good * self.alpha
            } else {
                last_good * -self.alpha
            };
        }

        // since we cannot read the side info we have to use defaults for this
        LongTermPostFilterInfo {
            is_active: false,
            pitch_index: 0,
            pitch_present: false,
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn save_and_load() {
        let mut spec_lines = [-2268.137, 7869.9785, 15884.984, 9776.979];
        let mut scaler_buf = [0.; 4];
        let (mut packet_loss, _) = PacketLossConcealment::new(spec_lines.len(), &mut scaler_buf);

        packet_loss.save(&spec_lines);
        packet_loss.load_into(&mut spec_lines);
        packet_loss.load_into(&mut spec_lines);
        packet_loss.load_into(&mut spec_lines);

        let spec_lines_expected = [2268.137, 7869.9785, -15884.984, -9776.979];
        assert_eq!(spec_lines, spec_lines_expected);
    }
}
