#[derive(Clone, Copy)]
pub enum SamplingFrequency {
    Hz8000,
    Hz16000,
    Hz24000,
    Hz32000,
    Hz44100,
    Hz48000,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDuration {
    SevenPointFiveMs,
    TenMs,
}

#[derive(Clone, Copy)]
pub struct Lc3Config {
    /// Sampling frequency index (e.g. 4)
    pub fs_ind: usize,

    /// Sampling frequency in hz (e.g. 48000)
    pub fs: usize,

    /// Number of encoded spectral lines (per frame and channel) (e.g. 400)
    pub ne: usize,

    /// Frame duration in milliseconds (e.g. TenMs)
    pub n_ms: FrameDuration,

    /// Number of bands (e.g. 64)
    pub nb: usize,

    /// Number of samples processed in one frame of one channel (also known as frame size) (e.g. 480)
    pub nf: usize,

    /// Number of leading zeros in MDCT window (e.g. 180)
    pub z: usize,
}

impl Lc3Config {
    pub const fn new(sampling_frequency: SamplingFrequency, frame_duration: FrameDuration) -> Self {
        let (fs_ind, fs) = match sampling_frequency {
            SamplingFrequency::Hz8000 => (0, 8000),
            SamplingFrequency::Hz16000 => (1, 16000),
            SamplingFrequency::Hz24000 => (2, 24000),
            SamplingFrequency::Hz32000 => (3, 32000),
            SamplingFrequency::Hz44100 => (4, 44100),
            SamplingFrequency::Hz48000 => (4, 48000),
        };

        let nf;
        let ne;
        let nb;
        let z;

        match frame_duration {
            FrameDuration::SevenPointFiveMs => {
                nf = match sampling_frequency {
                    SamplingFrequency::Hz8000 => 60,
                    SamplingFrequency::Hz16000 => 120,
                    SamplingFrequency::Hz24000 => 180,
                    SamplingFrequency::Hz32000 => 240,
                    SamplingFrequency::Hz44100 => 360,
                    SamplingFrequency::Hz48000 => 360,
                };

                ne = if nf == 360 { 300 } else { nf };
                nb = match sampling_frequency {
                    SamplingFrequency::Hz8000 => 60,
                    _ => 64,
                };
                z = 7 * nf / 30;
            }
            FrameDuration::TenMs => {
                nf = match sampling_frequency {
                    SamplingFrequency::Hz8000 => 80,
                    SamplingFrequency::Hz16000 => 160,
                    SamplingFrequency::Hz24000 => 240,
                    SamplingFrequency::Hz32000 => 320,
                    SamplingFrequency::Hz44100 => 480,
                    SamplingFrequency::Hz48000 => 480,
                };

                ne = if nf == 480 { 400 } else { nf };
                nb = 64;
                z = 3 * nf / 8;
            }
        }

        Self {
            fs_ind,
            fs,
            n_ms: frame_duration,
            nb,
            ne,
            nf,
            z,
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn simple_config() {
        let config = Lc3Config::new(SamplingFrequency::Hz48000, FrameDuration::TenMs);

        assert_eq!(config.fs, 48000);
        assert_eq!(config.fs_ind, 4);
        assert_eq!(config.n_ms, FrameDuration::TenMs);
        assert_eq!(config.z, 180);
        assert_eq!(config.nf, 480);
        assert_eq!(config.nb, 64);
        assert_eq!(config.ne, 400);
    }
}
