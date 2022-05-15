use super::buffer_writer::BufferWriter;
use crate::tables::{spec_noise_shape_quant_tables::*, spectral_data_tables::*, temporal_noise_shaping_tables::*};
use heapless::Vec;

#[allow(unused_imports)]
use num_traits::real::Real;

const MAX_NBITS_LSB: usize = 480 * 8; // NOTE: this is a guess, find out what the max can be!!

#[derive(Default)]
pub struct BitstreamEncoding {
    ne: usize,
    nbytes: usize,
    nbits: usize,
    nbits_side_initial: usize,
    nlsbs: usize,
    lsbs: Vec<u8, MAX_NBITS_LSB>,
    st: ArithmeticEncoderState,
    writer: BufferWriter,
}

#[derive(Default)]
struct ArithmeticEncoderState {
    pub low: u32,
    pub range: u32,
    pub cache: i32,
    pub carry: i32,
    pub carry_count: i32,
}

impl BitstreamEncoding {
    pub fn new(ne: usize) -> Self {
        Self {
            ne,
            ..Default::default()
        }
    }

    fn write_uint_backward(&mut self, value: usize, num_bits: usize, bytes: &mut [u8]) {
        self.writer.write_uint_backward(bytes, value, num_bits)
    }

    fn write_bool_backward(&mut self, value: bool, bytes: &mut [u8]) {
        self.writer.write_bool_backward(bytes, value)
    }

    fn write_byte_forward(&mut self, value: u8, bytes: &mut [u8]) {
        self.writer.write_byte_forward(bytes, value)
    }

    fn write_uint_forward(&mut self, value: usize, num_bits: usize, bytes: &mut [u8]) {
        self.writer.write_uint_forward(bytes, value as u16, num_bits)
    }

    fn nbits_side_written(&self) -> usize {
        self.writer.nbits_side_written(self.nbits)
    }

    fn nbits_side_forcast(&self) -> usize {
        let mut nbits_ari = (self.writer.bp * 8) as i32;
        nbits_ari += 25 - (self.st.range as f64).log2().floor() as i32;
        if self.st.carry >= 0 {
            nbits_ari += 8;
        }
        if self.st.carry_count > 0 {
            nbits_ari += self.st.carry_count * 8;
        }

        nbits_ari as usize
    }

    pub fn init(&mut self, bytes: &mut [u8]) {
        self.nbytes = bytes.len();
        self.nbits = self.nbytes * 8;
        self.writer = BufferWriter::new(bytes.len());
        bytes.fill(0);
        self.nlsbs = 0;
    }

    pub fn bandwidth(&mut self, p_bw: usize, nbits_bw: usize, bytes: &mut [u8]) {
        if nbits_bw > 0 {
            self.write_uint_backward(p_bw, nbits_bw, bytes);
        }
    }

    pub fn last_non_zero_tuple(&mut self, lastnz_trunc: usize, bytes: &mut [u8]) {
        let value = (lastnz_trunc >> 1) - 1;
        let num_bits = (self.ne as f64 / 2.0).log2().ceil() as usize;
        self.write_uint_backward(value, num_bits, bytes)
    }

    pub fn lsb_mode_bit(&mut self, lsb_mode: bool, bytes: &mut [u8]) {
        self.write_bool_backward(lsb_mode, bytes)
    }

    pub fn global_gain(&mut self, gg_ind: usize, bytes: &mut [u8]) {
        self.write_uint_backward(gg_ind, 8, bytes)
    }

    pub fn tns_activation_flag(&mut self, num_tns_filters: usize, rc_order: &[usize], bytes: &mut [u8]) {
        for rc_order_f in rc_order[..num_tns_filters].iter() {
            let value = *rc_order_f != 0;
            self.write_bool_backward(value, bytes);
        }
    }

    pub fn pitch_present_flag(&mut self, pitch_present: bool, bytes: &mut [u8]) {
        self.write_bool_backward(pitch_present, bytes)
    }

    pub fn encode_scf_vq_1st_stage(&mut self, ind_lf: usize, ind_hf: usize, bytes: &mut [u8]) {
        self.write_uint_backward(ind_lf, 5, bytes);
        self.write_uint_backward(ind_hf, 5, bytes);
    }

    pub fn encode_scf_vq_2nd_stage(
        &mut self,
        shape_j: usize,
        gain_i: usize,
        ls_inda: usize, // the type of this is strange over all code places -> finally only one bit is stored!
        index_joint_j: usize,
        bytes: &mut [u8],
    ) {
        /* Encode SCF VQ parameters - 2nd stage side-info (3-4 bits) */
        let submode_msb = (shape_j >> 1) != 0;
        self.write_bool_backward(submode_msb, bytes);
        //uint8_t submode_LSB = (shape_j & 0x1); /* shape_j is the stage2 shape_index [0…3] */
        let gain_msbs_num_bits = SNS_GAIN_MSB_BITS[shape_j];
        let gain_msbs = gain_i >> SNS_GAIN_LSB_BITS[shape_j]; /* where gain_i is the SNS-VQ stage 2 gain_index */
        self.write_uint_backward(gain_msbs, gain_msbs_num_bits, bytes);
        let ls_inda_flag = ls_inda != 0;
        self.write_bool_backward(ls_inda_flag, bytes);

        /* Encode SCF VQ parameters - 2nd stage MPVQ data */
        if !submode_msb {
            self.write_uint_backward(index_joint_j, 13, bytes);
            self.write_uint_backward(index_joint_j >> 13, 12, bytes);
        } else {
            self.write_uint_backward(index_joint_j, 12, bytes);
            self.write_uint_backward(index_joint_j >> 12, 12, bytes);
        }
    }

    pub fn ltpf_data(&mut self, ltpf_active: bool, pitch_index: usize, bytes: &mut [u8]) {
        self.write_bool_backward(ltpf_active, bytes);
        self.write_uint_backward(pitch_index, 9, bytes);
    }

    pub fn noise_factor(&mut self, f_nf: usize, bytes: &mut [u8]) {
        self.write_uint_backward(f_nf, 3, bytes);
    }

    pub fn ac_enc_init(&mut self) {
        self.st.low = 0;
        self.st.range = 0x00ff_ffff;
        self.st.cache = -1;
        self.st.carry = 0;
        self.st.carry_count = 0;
    }

    pub fn tns_data(
        &mut self,
        tns_lpc_weighting: u8, // TODO: this is actually used as a usize
        num_tns_filters: usize,
        rc_order: &[usize],
        rc_i: &[usize],
        bytes: &mut [u8],
    ) {
        for f in 0..num_tns_filters {
            if rc_order[f] > 0 {
                let cum_freq = AC_TNS_ORDER_CUMFREQ[tns_lpc_weighting as usize][rc_order[f] - 1];
                let sym_freq = AC_TNS_ORDER_FREQ[tns_lpc_weighting as usize][rc_order[f] - 1];
                self.ac_encode(cum_freq, sym_freq, bytes);
                for k in 0..rc_order[f] {
                    let cum_freq = AC_TNS_COEF_CUMFREQ[k][rc_i[k + 8 * f]];
                    let sym_freq = AC_TNS_COEF_FREQ[k][rc_i[k + 8 * f]];
                    self.ac_encode(cum_freq, sym_freq, bytes);
                }
            }
        }
    }

    pub fn spectral_data(
        &mut self,
        lastnz_trunc: usize,
        rate_flag: usize,
        lsb_mode: bool,
        x_q: &[i16],
        nbits_lsb: usize,
        bytes: &mut [u8],
    ) {
        self.nbits_side_initial = self.nbits_side_written();
        // self.lsbs = [0; nbits_lsb]; // store this pointer for subsequent use in residualDataAndFinalization
        self.lsbs.clear();
        for _ in 0..nbits_lsb {
            self.lsbs.push(0).unwrap();
        }

        let mut c = 0;
        for k in (0..lastnz_trunc).step_by(2) {
            let mut t = c + rate_flag + if k > (self.ne / 2) { 256 } else { 0 };
            let mut a = x_q[k].unsigned_abs();
            let mut a_lsb = a;
            let mut b = x_q[k + 1].unsigned_abs();
            let mut b_lsb = b;
            let mut lev = 0;
            let mut lsb0: u8 = 0;
            let mut lsb1: u8 = 0;
            while a.max(b) >= 4 {
                let pki_index = t + lev.min(3) * 1024;
                let pki = AC_SPEC_LOOKUP[pki_index] as usize;
                let cum_freq = AC_SPEC_CUMFREQ[pki][16];
                let sym_freq = AC_SPEC_FREQ[pki][16];
                self.ac_encode(cum_freq, sym_freq, bytes);
                if lsb_mode && lev == 0 {
                    lsb0 = a as u8 & 1;
                    lsb1 = b as u8 & 1;
                } else {
                    self.write_bool_backward((a & 1) == 1, bytes);
                    self.write_bool_backward((b & 1) == 1, bytes);
                }
                a >>= 1;
                b >>= 1;
                lev += 1;
            }
            let pki_index = t + lev.min(3) * 1024;
            let pki = AC_SPEC_LOOKUP[pki_index] as usize;
            let sym = (a + 4 * b) as usize;
            let cum_freq = AC_SPEC_CUMFREQ[pki][sym];
            let sym_freq = AC_SPEC_FREQ[pki][sym];
            self.ac_encode(cum_freq, sym_freq, bytes);
            //a_lsb = abs(𝑋𝑞[k]); -> implemented earlier
            //b_lsb = abs(𝑋𝑞[k+1]); -> implemented earlier
            if lsb_mode && lev > 0 {
                a_lsb >>= 1;
                b_lsb >>= 1;

                self.lsbs[self.nlsbs] = lsb0;
                self.nlsbs += 1;
                //if (a_lsb == 0 && 𝑋𝑞[k] != 0)
                if a_lsb == 0 && x_q[k] != 0 {
                    //lsbs[nlsbs++] = 𝑋𝑞[k]>0?0:1;
                    self.lsbs[self.nlsbs] = if x_q[k] > 0 { 0 } else { 1 };
                    self.nlsbs += 1;
                }
                self.lsbs[self.nlsbs] = lsb1;
                self.nlsbs += 1;
                //if (b_lsb == 0 && 𝑋𝑞[k+1] != 0)
                if b_lsb == 0 && x_q[k + 1] != 0 {
                    //lsbs[nlsbs++] = 𝑋𝑞[k+1]>0?0:1;
                    self.lsbs[self.nlsbs] = if x_q[k + 1] > 0 { 0 } else { 1 };
                    self.nlsbs += 1;
                }
            }
            if a_lsb > 0 {
                // 𝑋_q[k] > 0 ? 0 : 1
                self.write_bool_backward(x_q[k] <= 0, bytes);
            }
            if b_lsb > 0 {
                // 𝑋_q[k + 1] > 0 ? 0 : 1
                self.write_bool_backward(x_q[k + 1] <= 0, bytes);
            }
            lev = lev.min(3);
            t = if lev <= 1 {
                1 + ((a + b) as usize) * (lev + 1)
            } else {
                12 + lev
            };
            c = (c & 15) * 16 + t;
        }
    }

    pub fn residual_data_and_finalization(
        &mut self,
        lsb_mode: bool,
        residual_bits: impl Iterator<Item = bool>,
        bytes: &mut [u8],
    ) {
        let nbits_side = self.nbits_side_written();
        let nbits_ari = self.nbits_side_forcast();
        let nbits_residual_enc = self.nbits as i32 - (nbits_side + nbits_ari) as i32;
        let nbits_residual_enc = nbits_residual_enc.max(0) as usize;

        if !lsb_mode {
            for res_bit in residual_bits.take(nbits_residual_enc) {
                self.write_bool_backward(res_bit, bytes);
            }
        } else {
            let nbits_residual_enc = nbits_residual_enc.min(self.nlsbs);
            for k in 0..nbits_residual_enc {
                let value = self.lsbs[k] == 1;
                self.write_bool_backward(value, bytes);
            }
        }

        self.ac_enc_finish(bytes);
    }

    fn ac_enc_finish(&mut self, bytes: &mut [u8]) {
        let mut bits: i8 = 1;
        while (self.st.range >> (24 - bits)) == 0 {
            bits += 1;
        }
        let mut mask = 0x00ff_ffff >> bits;
        let mut val = self.st.low + mask;
        let over1 = val >> 24;
        let high = self.st.low + self.st.range;
        let over2 = high >> 24;
        val &= 0x00ff_ffff & !mask;
        if over1 == over2 {
            if (val + mask) >= high {
                bits += 1;
                mask >>= 1;
                val = ((self.st.low + mask) & 0x00ff_ffff) & !mask;
            }
            if val < self.st.low {
                self.st.carry = 1;
            }
        }
        self.st.low = val;
        while bits > 0 {
            self.ac_shift(bytes);
            bits -= 8;
        }
        bits += 8;
        if bits < 0 {
            panic!("bits is negative: {}", bits);
        }
        if self.st.carry_count > 0 {
            self.write_byte_forward(self.st.cache as u8, bytes);
            while self.st.carry_count > 1 {
                self.write_byte_forward(0xff, bytes);
                self.st.carry_count -= 1;
            }
            let value = 0xff >> (8 - bits);
            self.write_uint_forward(value, bits as usize, bytes);
        } else {
            self.write_uint_forward(self.st.cache as usize, bits as usize, bytes);
        }
    }

    fn ac_shift(&mut self, bytes: &mut [u8]) {
        if self.st.low < 0x00ff_0000 || self.st.carry == 1 {
            if self.st.cache >= 0 {
                let byte = ((self.st.cache + self.st.carry) & 0xff) as u8;
                self.write_byte_forward(byte, bytes);
            }
            while self.st.carry_count > 0 {
                let byte = ((self.st.carry + 0xff) & 0xff) as u8;
                self.write_byte_forward(byte, bytes);
                self.st.carry_count -= 1;
            }
            self.st.cache = (self.st.low >> 16) as i32;
            self.st.carry = 0;
        } else {
            self.st.carry_count += 1;
        }
        self.st.low <<= 8;
        self.st.low &= 0x00ff_ffff;
    }

    fn ac_encode(&mut self, cum_freq: i16, sym_freq: i16, bytes: &mut [u8]) {
        let r = self.st.range >> 10;
        self.st.low += r * cum_freq as u32;
        if self.st.low >> 24 != 0 {
            self.st.carry = 1;
        }
        self.st.low &= 0x00ff_ffff;
        self.st.range = r * sym_freq as u32;
        while self.st.range < 0x10000 {
            self.st.range <<= 8;
            self.ac_shift(bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use core::slice::Iter;

    use super::*;

    pub struct ResidualBitsTest<'a> {
        inner: Iter<'a, bool>,
    }

    impl<'a> Iterator for ResidualBitsTest<'a> {
        type Item = bool;

        fn next(&mut self) -> Option<Self::Item> {
            match self.inner.next() {
                Some(x) => Some(*x),
                None => None,
            }
        }
    }

    // TODO: this test sucks, fix it
    #[test]
    fn bitstream_encoding_run() {
        let mut bitstream_encoding = BitstreamEncoding::new(400);
        let mut buf_out = [0; 150];

        bitstream_encoding.init(&mut buf_out);

        // 3.3.13.3 Side information  (d09r02_F2F)
        bitstream_encoding.bandwidth(4, 3, &mut buf_out);

        bitstream_encoding.last_non_zero_tuple(350, &mut buf_out);
        bitstream_encoding.lsb_mode_bit(false, &mut buf_out);
        bitstream_encoding.global_gain(193, &mut buf_out);
        let rc_order = [8, 6];
        bitstream_encoding.tns_activation_flag(2, &rc_order, &mut buf_out);
        let pitch_present = true;
        bitstream_encoding.pitch_present_flag(pitch_present, &mut buf_out);

        // Encode SCF VQ parameters - 1st stage (10 bits)
        bitstream_encoding.encode_scf_vq_1st_stage(8, 17, &mut buf_out);

        // Encode SCF VQ parameters - 2nd stage side-info (3-4 bits)
        // Encode SCF VQ parameters - 2nd stage MPVQ data
        bitstream_encoding.encode_scf_vq_2nd_stage(3, 0, 0, 15253432, &mut buf_out);

        if pitch_present {
            bitstream_encoding.ltpf_data(false, 0, &mut buf_out);
        }

        bitstream_encoding.noise_factor(6, &mut buf_out);

        // 3.3.13.4 Arithmetic encoding  (d09r02_F2F)
        // Arithmetic Encoder Initialization
        bitstream_encoding.ac_enc_init();
        // TNS data
        let rc_i = [10, 7, 8, 9, 7, 9, 8, 9, 14, 11, 6, 9, 7, 9, 8, 8];
        bitstream_encoding.tns_data(0, 2, &rc_order, &rc_i, &mut buf_out);

        // Spectral data
        // TODO (theirs) check whether the degenerated case with nlsbs==0 works
        let x_q = [
            102, -146, -18, -14, -104, -128, 264, 254, -417, -180, 94, -28, 20, -38, 21, -62, -125, 10, -15, -4, 27,
            -9, -4, 3, 3, -1, 0, -13, -2, 0, -11, 3, 5, 4, -10, -18, -22, 4, 10, -5, 17, 4, -6, 2, 6, 11, -3, -3, 29,
            16, -15, 3, 4, 7, 4, -3, 5, 0, 6, 0, -6, 1, 0, -1, -2, 7, 6, 2, -9, -4, 3, -5, 3, 6, 4, -1, 3, 5, -1, -10,
            -16, 1, 1, 0, -4, -1, 7, -5, -4, -2, 0, -4, 1, 4, -1, -2, -7, 1, -2, 1, 1, -7, 1, 4, -1, -1, 2, 0, -1, -2,
            1, 3, -5, -1, 0, 2, 0, 0, 2, 0, 1, -3, 1, 2, 0, -5, -1, 5, -1, 0, -3, 0, 0, -1, 0, -2, 2, -3, 0, 1, -2, -1,
            -2, 0, 1, 2, -2, 0, -1, -3, -2, -1, 3, -2, -2, 0, 1, 0, -3, 1, 0, 0, -1, 0, 1, 0, 1, -2, 1, 1, 0, -1, 0, 0,
            1, 2, -1, 0, -1, 1, 0, -1, 1, -1, 1, -1, 0, 0, 0, -1, -1, 0, -2, 1, -1, -1, -1, -1, 0, -2, 0, -1, -1, 0, 0,
            0, 1, 0, -1, 0, 1, 1, 0, 0, 0, -1, 0, 0, -2, -1, 0, 1, 0, 0, 0, 1, -1, -1, 1, 0, 0, -1, -2, -1, -1, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, -1, -1, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        bitstream_encoding.spectral_data(350, 512, false, &x_q, 107, &mut buf_out);

        // 3.3.13.5 Residual data and finalization  (d09r02_F2F)

        /*
        let res_bits = [
            0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1,
            1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1,
            1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1,
            1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        ];*/
        //let res_bits: [bool; 140] = res_bits.map(|x| x == 1).iter().collect::<Vec<bool, 140>>();
        let res_bits = [
            false, true, false, false, false, false, true, true, false, true, false, true, true, true, true, false,
            false, true, false, true, true, true, false, true, true, true, false, true, false, true, true, true, false,
            false, true, true, true, true, false, false, true, true, true, false, true, false, false, true, false,
            true, true, true, true, false, true, false, true, false, false, true, false, true, true, false, false,
            true, false, false, false, true, false, true, true, true, false, false, true, false, false, true, true,
            false, true, true, false, false, true, false, false, true, false, true, false, false, false, true, false,
            true, false, true, true, true, true, true, true, false, false, true, true, false, false, true, false,
            false, false, true, true, true, false, true, false, true, true, true, true, false, false, false, true,
            true, true, false, true, true, true, true, true, true, true, true,
        ];

        let res_bits = ResidualBitsTest { inner: res_bits.iter() };
        bitstream_encoding.residual_data_and_finalization(false, res_bits, &mut buf_out);

        let buf_out_expected = [
            230, 243, 160, 169, 152, 75, 36, 156, 223, 96, 241, 214, 150, 248, 180, 106, 115, 92, 147, 213, 56, 100,
            96, 52, 194, 178, 44, 31, 222, 246, 83, 116, 240, 220, 40, 241, 82, 228, 209, 57, 128, 152, 9, 144, 112,
            249, 48, 46, 135, 182, 250, 59, 135, 221, 129, 46, 204, 178, 232, 100, 172, 27, 177, 120, 86, 253, 35, 137,
            19, 253, 191, 202, 97, 240, 10, 45, 124, 110, 234, 149, 49, 115, 209, 177, 153, 231, 93, 211, 214, 19, 127,
            143, 103, 47, 239, 86, 73, 91, 231, 94, 248, 143, 54, 54, 190, 51, 47, 136, 92, 157, 13, 226, 13, 96, 104,
            159, 17, 206, 66, 25, 157, 51, 5, 252, 166, 135, 213, 118, 107, 152, 226, 253, 51, 136, 74, 186, 52, 64,
            236, 152, 115, 0, 29, 23, 247, 3, 20, 124, 21, 116,
        ];

        assert_eq!(buf_out, buf_out_expected);
    }
}
