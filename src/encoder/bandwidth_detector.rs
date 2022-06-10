use crate::common::{complex::Scaler, config::FrameDuration};

// checked against spec

const I_BW_START_TABLE: [[usize; 4]; 4] = [[53, 0, 0, 0], [47, 59, 0, 0], [44, 54, 60, 0], [41, 51, 57, 61]];
const I_BW_STOP_TABLE: [[usize; 4]; 4] = [[63, 0, 0, 0], [56, 63, 0, 0], [52, 59, 63, 0], [49, 55, 60, 63]];
const I_BW_START_TABLE_7P5MS: [[usize; 4]; 4] = [[51, 0, 0, 0], [45, 58, 0, 0], [42, 53, 60, 0], [40, 51, 57, 61]];
const I_BW_STOP_TABLE_7P5MS: [[usize; 4]; 4] = [[63, 0, 0, 0], [55, 63, 0, 0], [51, 58, 63, 0], [48, 55, 60, 63]];
const NBITS_BW_TABLE: [usize; 5] = [0, 1, 2, 2, 3];

// quietness thresholds (TQ) for each bandwidth index
const QUIETNESS_THRESH: [usize; 4] = [20, 10, 10, 10];

// cutoff throsholds (TC) for each bandwidth index
const CUTOFF_THRESH: [usize; 4] = [15, 23, 20, 20];

const L_10MS: [usize; 4] = [4, 4, 3, 1];
const L_7P5MS: [usize; 4] = [4, 4, 3, 2];

pub struct BandwidthDetector {
    sample_freq_ind: usize, // n_bw
    i_bw_start: &'static [usize],
    i_bw_stop: &'static [usize],
    l: &'static [usize],
}

pub struct BandwidthDetectorResult {
    pub bandwidth_ind: usize,
    pub nbits_bandwidth: usize,
}

impl BandwidthDetector {
    pub fn new(frame_duration: FrameDuration, sample_freq_ind: usize) -> Self {
        let (i_bw_start, i_bw_stop, l) = match frame_duration {
            FrameDuration::TenMs => (
                I_BW_START_TABLE[sample_freq_ind - 1].as_slice(),
                I_BW_STOP_TABLE[sample_freq_ind - 1].as_slice(),
                L_10MS.as_slice(),
            ),
            FrameDuration::SevenPointFiveMs => (
                I_BW_START_TABLE_7P5MS[sample_freq_ind - 1].as_slice(),
                I_BW_STOP_TABLE_7P5MS[sample_freq_ind - 1].as_slice(),
                L_7P5MS.as_slice(),
            ),
        };

        Self {
            sample_freq_ind,
            i_bw_start,
            i_bw_stop,
            l,
        }
    }

    pub fn get_num_bits_bandwidth(&self) -> usize {
        NBITS_BW_TABLE[self.sample_freq_ind]
    }

    /// Detect bandlimited signals coded at higher sampling rates (upsampled signals)
    ///
    /// # Arguments
    ///
    /// * `e_b` - Input
    pub fn run(&self, e_b: &[Scaler]) -> BandwidthDetectorResult {
        let nbits_bandwidth = NBITS_BW_TABLE[self.sample_freq_ind];
        if self.sample_freq_ind == 0 {
            return BandwidthDetectorResult {
                bandwidth_ind: 0,
                nbits_bandwidth,
            };
        }

        // bandwidth index candidate
        let mut bandwidth_ind = 0;

        // first stage - find the highest non-quiet band
        // start with the highest bandwidth and work down until a non-quiet band is detected
        for k in (0..self.sample_freq_ind).rev() {
            let start = self.i_bw_start[k];
            let stop = self.i_bw_stop[k];
            let width = (stop + 1 - start) as Scaler;
            let mut quietness = 0.0;

            // calculate the quietness of the energy band
            for energy in e_b[start..=stop].iter() {
                quietness += *energy / width;
            }

            // if this quiteness is over the threshold then then this is the candidate bandwidth for this band
            if quietness >= QUIETNESS_THRESH[k] as Scaler {
                bandwidth_ind = k + 1;
                break;
            }
        }

        // second stage - determine the final bandwidth
        if self.sample_freq_ind == bandwidth_ind {
            BandwidthDetectorResult {
                bandwidth_ind,
                nbits_bandwidth,
            }
        } else {
            // detect an energy drop above the cut-off frequency of the candidate bandwidth bw_0
            let mut cutoff_max = 0.0;
            let l_bw = self.l[bandwidth_ind];
            let from = self.i_bw_start[bandwidth_ind] + 1 - l_bw;
            let to = self.i_bw_start[bandwidth_ind];
            for n in from..to {
                // NOTE: the spec calls for adding EPSILON and multiplying this by 10log10
                // which does not seem to have any effect
                let cutoff = e_b[n - l_bw] / e_b[n];
                cutoff_max = cutoff.max(cutoff_max);
            }

            if cutoff_max > CUTOFF_THRESH[bandwidth_ind] as Scaler {
                BandwidthDetectorResult {
                    bandwidth_ind,
                    nbits_bandwidth,
                }
            } else {
                BandwidthDetectorResult {
                    bandwidth_ind: self.sample_freq_ind,
                    nbits_bandwidth,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::common::config::{FrameDuration, Lc3Config, SamplingFrequency};

    #[test]
    fn bandwidth_detector_run() {
        let config = Lc3Config::new(SamplingFrequency::Hz48000, FrameDuration::TenMs);
        let detector = BandwidthDetector::new(config.n_ms, config.fs_ind);
        #[rustfmt::skip]
        let e_b = [
            18782358.0, 38743940.0, 616163.4, 395644.22, 20441758.0, 31336932.0, 130985740.0, 116001040.0, 297851520.0,
            52884070.0, 13922108.0, 1211718.9, 643103.44, 2697760.3, 1674064.1, 8157596.5, 11880675.0, 615179.8,
            1548138.0, 15001.449, 72356.61, 424075.13, 339332.63, 9093.422, 118530.22, 106300.99, 93872.414, 103234.38,
            129645.81, 12188.947, 48738.766, 124244.47, 12835.649, 47138.63, 10050.675, 24161.475, 15844.599, 16136.73,
            37085.188, 5236.445, 14986.677, 10497.837, 8121.843, 2109.306, 3711.1233, 3116.423, 3749.5027, 4903.189,
            3149.5522, 1745.0712, 1382.3269, 1555.3384, 994.6934, 1484.393, 888.5528, 926.9374, 639.82434, 801.4557,
            743.6313, 487.39868, 681.486, 519.567, 481.0444, 454.6319,
        ];

        let result = detector.run(&e_b);

        assert_eq!(result.bandwidth_ind, 4);
        assert_eq!(result.nbits_bandwidth, 3);
    }
}
