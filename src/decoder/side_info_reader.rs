use super::{
    buffer_reader::{BufferReader, BufferReaderError},
    side_info::{Bandwidth, LongTermPostFilterInfo, SideInfo, SnsVq},
};

#[allow(unused_imports)]
use num_traits::real::Real;

// checked against spec. The only function to double check is read_sns_vq

const NBITS_BW_TABLE: [usize; 5] = [0, 1, 2, 2, 3];

#[derive(Debug)]
pub enum SideInfoError {
    BandwidthIdxOutOfRange(usize),
    LastNonZeroTupleGreaterThanYLen(usize, usize),
    NotImplemented,
    PlcTriggerSns1OutOfRange(usize),
    PlcTriggerSns2OutOfRange(usize),
    BufferReaderError(BufferReaderError),
}

impl From<BufferReaderError> for SideInfoError {
    fn from(err: BufferReaderError) -> Self {
        Self::BufferReaderError(err)
    }
}

pub fn read(
    buf: &[u8],
    reader: &mut BufferReader,
    fs_ind: usize, // sampling rate index (fs_ind)
    ne: usize,     // number of encoded spectral lines (y_len)
) -> Result<SideInfo, SideInfoError> {
    let nbits_bw = NBITS_BW_TABLE[fs_ind];

    // bandwidth
    let p_bw = if nbits_bw > 0 {
        let idx = reader.read_tail_usize(buf, nbits_bw)?;
        if fs_ind < idx {
            return Err(SideInfoError::BandwidthIdxOutOfRange(idx));
        } else {
            idx
        }
    } else {
        0
    };

    // last non-zero tuple
    let lastnz_num_bits = ((ne / 2) as f32).log2().ceil() as usize;
    let lastnz = reader.read_tail_usize(buf, lastnz_num_bits)?;
    let lastnz = (lastnz + 1) << 1;
    if lastnz > ne {
        return Err(SideInfoError::LastNonZeroTupleGreaterThanYLen(lastnz, ne));
    }

    // lsb mode bit
    let lsb_mode = reader.read_tail_bool(buf)?;

    // global gain
    let gg_ind = reader.read_tail_usize(buf, 8)?;

    // number of TNS filters
    let num_tns_filters = if p_bw < 3 { 1 } else { 2 };

    // TNS order of quantized reflection coefficients
    let mut rc_order: [usize; 2] = [0; 2];
    for item in rc_order.iter_mut().take(num_tns_filters) {
        *item = reader.read_tail_bool(buf)? as usize;
    }

    // pitch present flag
    let pitch_present = reader.read_tail_bool(buf)?;

    // read sns-vq
    let sns_vq = read_sns_vq(buf, reader)?;

    // long term post filter
    let long_term_post_filter_info = read_long_term_post_filter_info(buf, reader, pitch_present)?;

    let f_nf = reader.read_tail_usize(buf, 3)?;

    let bandwidth = match p_bw {
        0 => Bandwidth::NarrowBand,
        1 => Bandwidth::WideBand,
        2 => Bandwidth::SemiSuperWideBand,
        3 => Bandwidth::SuperWideBand,
        4 => Bandwidth::FullBand,
        _ => return Err(SideInfoError::BandwidthIdxOutOfRange(p_bw)),
    };

    Ok(SideInfo {
        bandwidth,
        lastnz,
        lsb_mode,
        global_gain_index: gg_ind,
        num_tns_filters,
        reflect_coef_order_ari_input: rc_order,
        sns_vq,
        long_term_post_filter_info,
        noise_factor: f_nf,
    })
}

fn read_long_term_post_filter_info(
    buf: &[u8],
    reader: &mut BufferReader,
    pitch_present: bool,
) -> Result<LongTermPostFilterInfo, SideInfoError> {
    let ltpf_active;
    let pitch_index;
    if pitch_present {
        ltpf_active = reader.read_tail_bool(buf)?;
        pitch_index = reader.read_tail_usize(buf, 9)?;
    } else {
        ltpf_active = false;
        pitch_index = 0;
    }

    Ok(LongTermPostFilterInfo {
        pitch_present,
        is_active: ltpf_active,
        pitch_index,
    })
}

fn read_sns_vq(buf: &[u8], reader: &mut BufferReader) -> Result<SnsVq, SideInfoError> {
    // stage 1 decoding
    let ind_lf = reader.read_tail_usize(buf, 5)?;
    let ind_hf = reader.read_tail_usize(buf, 5)?;

    // stage 2 decoding
    let submode_msb = reader.read_tail_bool(buf)? as u8;
    let mut g_ind = if submode_msb == 0 {
        reader.read_tail_usize(buf, 1)?
    } else {
        reader.read_tail_usize(buf, 2)?
    };

    let ls_inda = reader.read_tail_bool(buf)? as usize;
    let ls_indb;
    let idx_a;
    let idx_b;
    let mut submode_lsb;

    if submode_msb == 0 {
        let tmp = reader.read_tail_usize(buf, 25)?;
        if tmp >= 33460056 {
            return Err(SideInfoError::PlcTriggerSns1OutOfRange(tmp));
        }

        let idx_bor_gain_lsb = tmp / 2390004;
        idx_a = tmp - idx_bor_gain_lsb * 2390004;
        submode_lsb = 0;
        let idx_bor_gain_lsb: i32 = idx_bor_gain_lsb as i32 - 2_i32;
        if idx_bor_gain_lsb < 0 {
            submode_lsb = 1;
        }

        let idx_bor_gain_lsb = (idx_bor_gain_lsb + submode_lsb as i32 * 2) as usize;
        if submode_lsb != 0 {
            g_ind = (g_ind << 1) + idx_bor_gain_lsb;
            idx_b = 0;
            ls_indb = 0;
        } else {
            idx_b = idx_bor_gain_lsb >> 1;
            ls_indb = idx_bor_gain_lsb & 1;
        }
    } else {
        ls_indb = 0;
        idx_b = 0;
        submode_lsb = 0;
        let tmp = reader.read_tail_usize(buf, 24)?;

        if tmp >= 16708096 {
            return Err(SideInfoError::PlcTriggerSns2OutOfRange(tmp));
        }

        if tmp >= 15158272 {
            let tmp = tmp - 15158272;
            submode_lsb = 1;
            g_ind = (g_ind << 1) + (tmp & 1);
            idx_a = tmp >> 1;
        } else {
            idx_a = tmp;
        }
    }

    Ok(SnsVq {
        ind_lf,
        ind_hf,
        ls_inda,
        ls_indb,
        idx_a,
        idx_b,
        submode_lsb,
        submode_msb,
        g_ind,
    })
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn read_side_info_test() {
        // the last 8 bytes from a typical lc3 encoded frame
        // side info is read from the end of the frame towards the front
        let buf = [192, 74, 255, 80, 28, 187, 134, 52];
        let mut reader = BufferReader::new();

        let side_info = read(&buf, &mut reader, 4, 400).unwrap();

        assert_eq!(side_info.bandwidth, Bandwidth::FullBand);
        assert_eq!(side_info.lastnz, 398);
        assert_eq!(side_info.lsb_mode, false);
        assert_eq!(side_info.global_gain_index, 184);
        assert_eq!(side_info.num_tns_filters, 2);
        assert_eq!(side_info.reflect_coef_order_ari_input, [1, 1]);
        let sns_vq = side_info.sns_vq;
        assert_eq!(sns_vq.ind_lf, 25);
        assert_eq!(sns_vq.ind_hf, 1);
        assert_eq!(sns_vq.ls_inda, 0);
        assert_eq!(sns_vq.ls_indb, 0);
        assert_eq!(sns_vq.idx_a, 307189);
        assert_eq!(sns_vq.idx_b, 0);
        assert_eq!(sns_vq.submode_lsb, 1);
        assert_eq!(sns_vq.submode_msb, 0);
        assert_eq!(sns_vq.g_ind, 0);
        let post_filter = side_info.long_term_post_filter_info;
        assert_eq!(post_filter.pitch_present, false);
        assert_eq!(post_filter.is_active, false);
        assert_eq!(post_filter.pitch_index, 0);
        assert_eq!(side_info.noise_factor, 6);
    }
}
