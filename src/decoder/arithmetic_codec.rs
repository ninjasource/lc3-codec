use super::{
    buffer_reader::{BufferReader, BufferReaderError},
    side_info::SideInfo,
};
use crate::{
    common::{
        config::FrameDuration,
        constants::{MAX_LEN_FREQUENCY, MAX_LEN_SPECTRAL},
    },
    tables::{
        spectral_data_tables::{AC_SPEC_CUMFREQ, AC_SPEC_FREQ, AC_SPEC_LOOKUP},
        temporal_noise_shaping_tables::{
            AC_TNS_COEF_CUMFREQ, AC_TNS_COEF_FREQ, AC_TNS_ORDER_CUMFREQ, AC_TNS_ORDER_FREQ, MAXLAG, TNS_NUMFILTERS_MAX,
        },
    },
};
use heapless::Vec;
#[allow(unused_imports)]
use num_traits::real::Real;

#[derive(Debug)]
struct ArithmeticDecoderState {
    ac_low: u32,   // should this be i32?
    ac_range: u32, // should this be i32?
}

#[derive(Debug)]
pub enum ArithmeticCodecError {
    AcRangeFlOutOfRange(u32, u32),
    BufferReader(BufferReaderError),
}

impl From<BufferReaderError> for ArithmeticCodecError {
    fn from(err: BufferReaderError) -> Self {
        Self::BufferReader(err)
    }
}

#[derive(Debug)]
pub enum ArithmeticDecodeError {
    ArithmeticCodec(ArithmeticCodecError),
    TnsOrder(usize, ArithmeticCodecError),
    TnsCoef(usize, usize, ArithmeticCodecError),
    SpectralData(usize, usize, ArithmeticCodecError),
    SpectralBoolData(usize, usize, BufferReaderError),
    NegativeResidualNumBits,
    ResidualBoolData(bool, usize),
    ResidualBoolDataOverflow(bool, usize, usize),
}

impl From<ArithmeticCodecError> for ArithmeticDecodeError {
    fn from(err: ArithmeticCodecError) -> Self {
        Self::ArithmeticCodec(err)
    }
}

fn ac_dec_init(buf: &[u8], reader: &mut BufferReader) -> Result<ArithmeticDecoderState, ArithmeticCodecError> {
    let ac_low_fl = reader.read_head_u24(buf)?;
    let ac_range_fl = 0x00ffffff;

    Ok(ArithmeticDecoderState {
        ac_low: ac_low_fl,
        ac_range: ac_range_fl,
    })
}

fn ac_decode(
    buf: &[u8],
    reader: &mut BufferReader,
    st: &mut ArithmeticDecoderState,
    cum_freq: &[i16],
    sym_freq: &[i16],
) -> Result<usize, ArithmeticCodecError> {
    let tmp = st.ac_range >> 10;

    let limit = tmp << 10;
    if st.ac_low >= limit {
        return Err(ArithmeticCodecError::AcRangeFlOutOfRange(st.ac_low, limit));
    }

    let mut val = cum_freq.len() - 1;
    while st.ac_low < (tmp * cum_freq[val] as u32) {
        val -= 1;
    }

    st.ac_low -= tmp * cum_freq[val] as u32;
    st.ac_range = tmp * sym_freq[val] as u32;

    while st.ac_range < 0x10000 {
        st.ac_low <<= 8;
        st.ac_low &= 0x00ffffff;
        st.ac_low += reader.read_head_byte(buf)? as u32;
        st.ac_range <<= 8;
    }

    Ok(val)
}

#[derive(Debug, PartialEq)]
pub struct ArithmeticData {
    pub reflect_coef_order: [usize; 2], // also called rc_order
    pub reflect_coef_ints: [usize; 16], // also called rc_i or tns_idx
    pub residual_bits: Vec<bool, 480>,
    pub noise_filling_seed: i32,
    pub is_zero_frame: bool,
    pub frame_num_bits: usize, // number of bits in the frame (nbits) (frame length * 8) (e.g. 1200)
}

pub fn decode(
    buf: &[u8],                // the entire frame
    reader: &mut BufferReader, // a cursor for reading parts of the frame
    fs_ind: usize,             // sampling rate index
    ne: usize,                 // number of encoded spectral lines (NE) (also known as L_spec or ylen)
    side_info: &SideInfo,      // the side info already read from the frame
    n_ms: &FrameDuration,
    x: &mut [i32],
) -> Result<ArithmeticData, ArithmeticDecodeError> {
    let num_bytes = buf.len();
    let nbits = num_bytes * 8;

    // start decoding
    let mut st = ac_dec_init(buf, reader)?;

    // decode TNS data
    let (tns_idx, tns_order) = decode_tns_data(buf, reader, side_info, &mut st, nbits, n_ms)?;

    // spectral data (mutates st, x and save_lev)
    let mut save_lev: [i32; MAX_LEN_SPECTRAL] = [0; MAX_LEN_SPECTRAL];
    decode_spectral_data(buf, reader, side_info, nbits, fs_ind, ne, &mut st, x, &mut save_lev)?;

    // residual data and finalization
    for item in &mut x[side_info.lastnz..] {
        *item = 0;
    }

    // mutates x and save_lev
    let residual_bits = decode_residual_bits(buf, reader, side_info, &st, nbits, ne, x, &mut save_lev)?;

    // noise filling seed
    let mut tmp = 0;
    for (k, item) in x[..ne].iter().enumerate() {
        tmp += item.abs() * k as i32;
    }
    let noise_filling_seed = (tmp as i32) & 0xFFFF;

    // zero frame flag
    let is_zero_frame = side_info.lastnz == 2 && x[0] == 0 && x[1] == 0 && side_info.global_gain_index == 0;

    Ok(ArithmeticData {
        is_zero_frame,
        noise_filling_seed,
        reflect_coef_ints: tns_idx,
        reflect_coef_order: tns_order,
        residual_bits,
        frame_num_bits: nbits,
    })
}

fn decode_residual_bits(
    buf: &[u8],
    reader: &mut BufferReader,
    side_info: &SideInfo,
    st: &ArithmeticDecoderState,
    nbits: usize,
    ne: usize,
    x: &mut [i32],
    save_lev: &mut [i32],
) -> Result<Vec<bool, MAX_LEN_FREQUENCY>, ArithmeticDecodeError> {
    // number of residual bits
    let mut nbits_residual = calc_num_residual_bits(reader, st, nbits)?;
    let lsb_mode = side_info.lsb_mode;
    let mut residual_bits = Vec::new();

    // decode residual bits
    if !lsb_mode {
        // Ne (from the spec - also called ylen) - number of encoded spectral lines
        for (k, x_k) in x[..ne].iter().enumerate() {
            if *x_k != 0 {
                if residual_bits.len() == nbits_residual {
                    break;
                }

                let bit = reader
                    .read_tail_bool(buf)
                    .map_err(|_| ArithmeticDecodeError::ResidualBoolData(lsb_mode, k))?;

                residual_bits
                    .push(bit)
                    .map_err(|_| ArithmeticDecodeError::ResidualBoolDataOverflow(lsb_mode, k, residual_bits.len()))?;
            }
        }
    } else {
        for k in (0..side_info.lastnz).step_by(2) {
            if save_lev[k] > 0 {
                if !read_res_bit(x, reader, buf, k, &mut nbits_residual, lsb_mode)? {
                    break;
                }

                if !read_res_bit(x, reader, buf, k + 1, &mut nbits_residual, lsb_mode)? {
                    break;
                }
            }
        }
    }

    Ok(residual_bits)
}

// 1.3 ms
fn decode_spectral_data(
    buf: &[u8],
    reader: &mut BufferReader,
    side_info: &SideInfo,
    nbits: usize,
    fs_ind: usize,
    ne: usize,
    st: &mut ArithmeticDecoderState,
    x: &mut [i32],
    save_lev: &mut [i32],
) -> Result<(), ArithmeticDecodeError> {
    // rate flag
    let rate_flag = if nbits > (160 + fs_ind * 160) { 512 } else { 0 };
    let mut c = 0;

    for (k, chunk) in x[..side_info.lastnz].chunks_exact_mut(2).enumerate() {
        let mut t = c + rate_flag + if (k * 2) > (ne / 2) { 256 } else { 0 };

        // seems horrible but the only way to get a reference to this data
        let (x_k, x_kplus1) = chunk.split_at_mut(1);
        let x_k = &mut x_k[0];
        let x_kplus1 = &mut x_kplus1[0];

        *x_k = 0;
        *x_kplus1 = 0;
        let mut sym = 0;
        let mut lev: usize = 0;

        // 1.0 ms
        while lev < 14 {
            let pki_index = t + lev.min(3) * 1024;
            let pki = AC_SPEC_LOOKUP[pki_index] as usize;

            let cum_freq = &AC_SPEC_CUMFREQ[pki];
            let spec_freq = &AC_SPEC_FREQ[pki];
            sym = ac_decode(buf, reader, st, cum_freq, spec_freq)
                .map_err(|err| ArithmeticDecodeError::SpectralData(k, lev, err))?;

            if sym < 16 {
                break;
            }

            if !side_info.lsb_mode || lev > 0 {
                let bit = reader
                    .read_tail_bool(buf)
                    .map_err(|err| ArithmeticDecodeError::SpectralBoolData(k, lev, err))?
                    as i32;
                *x_k += bit << lev;
                let bit = reader
                    .read_tail_bool(buf)
                    .map_err(|err| ArithmeticDecodeError::SpectralBoolData(k, lev, err))?
                    as i32;
                *x_kplus1 += bit << lev;
            }

            lev += 1;
        }

        if side_info.lsb_mode {
            // used later for residual info
            save_lev[k] = lev as i32;
        }

        let a = sym & 0x3;
        let b = sym >> 2;

        *x_k += (a as i32) << lev;
        *x_kplus1 += (b as i32) << lev;

        if *x_k > 0 {
            let bit = reader
                .read_tail_bool(buf)
                .map_err(|err| ArithmeticDecodeError::SpectralBoolData(k, lev, err))?;
            if bit {
                *x_k = -*x_k;
            }
        }

        if *x_kplus1 > 0 {
            let bit = reader
                .read_tail_bool(buf)
                .map_err(|err| ArithmeticDecodeError::SpectralBoolData(k, lev, err))?;
            if bit {
                *x_kplus1 = -*x_kplus1;
            }
        }

        lev = lev.min(3);
        t = if lev <= 1 { 1 + (a + b) * (lev + 1) } else { 12 + lev };

        c = (c & 15) * 16 + t;
    }

    Ok(())
}

fn decode_tns_data(
    buf: &[u8],
    reader: &mut BufferReader,
    side_info: &SideInfo,
    st: &mut ArithmeticDecoderState,
    nbits: usize,
    n_ms: &FrameDuration,
) -> Result<([usize; 16], [usize; 2]), ArithmeticDecodeError> {
    let max_bits = match n_ms {
        FrameDuration::SevenPointFiveMs => 360,
        FrameDuration::TenMs => 480,
    };

    let tns_lpc_weighting = nbits < max_bits; // enable linear predictive coding weighting
    let tns_lpc_weighting_idx = tns_lpc_weighting as usize;

    let mut tns_idx: [usize; TNS_NUMFILTERS_MAX * MAXLAG] = [0; TNS_NUMFILTERS_MAX * MAXLAG];
    let mut tns_order = side_info.reflect_coef_order_ari_input; // a copy of tns_order is taken
    for (f, tns_order_f) in tns_order[..side_info.num_tns_filters].iter_mut().enumerate() {
        if *tns_order_f > 0 {
            let cum_freq = &AC_TNS_ORDER_CUMFREQ[tns_lpc_weighting_idx];
            let sym_freq = &AC_TNS_ORDER_FREQ[tns_lpc_weighting_idx];
            let order = ac_decode(buf, reader, st, cum_freq, sym_freq)
                .map_err(|err| ArithmeticDecodeError::TnsOrder(f, err))?;

            *tns_order_f = order + 1;
            for k in 0..*tns_order_f {
                let idx = f * 8 + k;
                let cum_freq = &AC_TNS_COEF_CUMFREQ[k];
                let sym_freq = &AC_TNS_COEF_FREQ[k];
                tns_idx[idx] = ac_decode(buf, reader, st, cum_freq, sym_freq)
                    .map_err(|err| ArithmeticDecodeError::TnsCoef(f, k, err))?;
            }
        }
    }

    Ok((tns_idx, tns_order))
}

fn read_res_bit(
    x: &mut [i32],
    reader: &mut BufferReader,
    buf: &[u8],
    x_index: usize,
    nbits_res: &mut usize,
    lsb_mode: bool,
) -> Result<bool, ArithmeticDecodeError> {
    // check and read bit
    if *nbits_res == 0 {
        return Ok(false);
    }
    let bit = reader
        .read_tail_bool(buf)
        .map_err(|_| ArithmeticDecodeError::ResidualBoolData(lsb_mode, x_index))?;
    *nbits_res -= 1;

    if bit {
        let val = &mut x[x_index];
        match val {
            v if *v > 0 => {
                *v += 1;
            }
            v if *v < 0 => {
                *v -= 1;
            }
            v => {
                // check and read bit
                if *nbits_res == 0 {
                    return Ok(false);
                }
                let bit = reader
                    .read_tail_bool(buf)
                    .map_err(|_| ArithmeticDecodeError::ResidualBoolData(lsb_mode, x_index))?;
                *nbits_res -= 1;

                *v = if bit { -1 } else { 1 };
            }
        };
    }

    Ok(true)
}

fn calc_num_residual_bits(
    reader: &BufferReader,
    st: &ArithmeticDecoderState,
    total_bits: usize,
) -> Result<usize, ArithmeticDecodeError> {
    let nbits_side = reader.get_tail_bit_cursor() - 8;

    // TODO: surely there is a better way to do this
    let nbits_ari = (reader.get_head_byte_cursor() + 1 - 3) * 8 + 25 - (st.ac_range as f64).log2().floor() as usize;

    if total_bits >= (nbits_side + nbits_ari) {
        Ok(total_bits - nbits_side - nbits_ari)
    } else {
        Err(ArithmeticDecodeError::NegativeResidualNumBits)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use crate::decoder::side_info::{Bandwidth, LongTermPostFilterInfo, SnsVq};

    use super::*;

    #[test]
    fn arithmetic_decode() {
        let buf = [
            187, 56, 111, 155, 76, 236, 70, 99, 10, 135, 219, 76, 176, 3, 108, 203, 131, 111, 206, 221, 195, 25, 96,
            240, 18, 202, 163, 241, 109, 142, 198, 122, 176, 70, 37, 6, 35, 190, 110, 184, 251, 162, 71, 7, 151, 58,
            42, 79, 200, 192, 99, 157, 234, 156, 245, 43, 84, 64, 167, 32, 52, 106, 43, 75, 4, 102, 213, 123, 168, 120,
            213, 252, 208, 118, 78, 115, 154, 158, 157, 26, 152, 231, 121, 146, 203, 11, 169, 227, 75, 154, 237, 154,
            227, 145, 196, 182, 207, 94, 95, 26, 184, 248, 1, 118, 72, 47, 18, 205, 56, 96, 195, 139, 216, 240, 113,
            233, 44, 198, 245, 157, 139, 70, 162, 182, 139, 136, 165, 68, 79, 247, 161, 126, 17, 135, 36, 30, 229, 24,
            196, 2, 5, 65, 111, 80, 124, 168, 70, 156, 198, 60,
        ];
        let mut reader = BufferReader::new_at(0, 64);
        let fs_ind = 4;
        let ne = 400;
        let side_info = SideInfo {
            bandwidth: Bandwidth::FullBand,
            lastnz: 400,
            lsb_mode: false,
            global_gain_index: 204,
            num_tns_filters: 2,
            reflect_coef_order_ari_input: [1, 0],
            sns_vq: SnsVq {
                ind_lf: 13,
                ind_hf: 4,
                ls_inda: 1,
                ls_indb: 0,
                idx_a: 1718290,
                idx_b: 2,
                submode_lsb: 0,
                submode_msb: 0,
                g_ind: 0,
            },
            long_term_post_filter_info: LongTermPostFilterInfo {
                pitch_present: false,
                is_active: false,
                pitch_index: 0,
            },
            noise_factor: 3,
        };
        let n_ms = &FrameDuration::TenMs;
        let mut x = [0; MAX_LEN_SPECTRAL];

        let arithmetic_data = decode(&buf, &mut reader, fs_ind, ne, &side_info, &n_ms, &mut x).unwrap();

        assert_eq!(arithmetic_data.is_zero_frame, false);
        assert_eq!(arithmetic_data.frame_num_bits, 1200);
        assert_eq!(arithmetic_data.noise_filling_seed, 56909);
        assert_eq!(
            arithmetic_data.reflect_coef_ints,
            [6, 10, 7, 8, 7, 9, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            arithmetic_data.residual_bits,
            [
                false, true, true, true, false, false, false, true, false, false, true, true, true, false, false,
                false, true, true, true, false, true, false, true, true, false, false, true, true, false, true, true,
                false, true, true, true, false, true, false, true, true, false, false, true, true, true
            ]
        );
        assert_eq!(arithmetic_data.reflect_coef_order, [8, 0]);
    }
}
