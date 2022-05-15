#[allow(unused_imports)]
use num_traits::real::Real;

#[derive(Default)]
pub struct BufferWriter {
    pub bp: usize,
    bp_side: u16,
    mask_side: u8,
}

impl BufferWriter {
    pub fn new(buf_len: usize) -> Self {
        Self {
            bp: 0,
            bp_side: buf_len as u16 - 1,
            mask_side: 1,
        }
    }
    pub fn write_uint_backward(&mut self, buf: &mut [u8], mut val: usize, num_bits: usize) {
        for _ in 0..num_bits {
            let bit = (val as u16 & 1) as u8;
            self.write_bool_backward(buf, bit != 0);
            val >>= 1;
        }
    }

    pub fn write_bool_backward(&mut self, buf: &mut [u8], bit: bool) {
        if !bit {
            buf[self.bp_side as usize] &= !self.mask_side;
        } else {
            buf[self.bp_side as usize] |= self.mask_side;
        }

        if self.mask_side == 0x80 {
            self.mask_side = 1;
            self.bp_side -= 1
        } else {
            self.mask_side <<= 1;
        }
    }

    pub fn write_uint_forward(&mut self, buf: &mut [u8], val: u16, num_bits: usize) {
        let mut mask: u8 = 0x80;
        for _ in 0..num_bits {
            let bit = val as u8 & mask; // TODO: check this!!!
            if bit == 0 {
                buf[self.bp] &= !mask;
            } else {
                buf[self.bp] |= mask;
            }
            mask >>= 1;
        }
    }

    pub fn write_byte_forward(&mut self, buf: &mut [u8], val: u8) {
        buf[self.bp] = val;
        self.bp += 1;
    }

    pub fn nbits_side_written(&self, nbits: usize) -> usize {
        let value = nbits as i32 - (8 * self.bp_side as i32 + 8 - (self.mask_side as f64).log2() as i32);
        if value < 0 {
            panic!("nbits_side_written is negative: {}", value);
        }
        value as usize
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn buffer_writer_forward_and_backwards() {
        let mut writer = BufferWriter::new(10);
        let mut buf = [0; 10];

        writer.write_bool_backward(&mut buf, true);
        writer.write_byte_forward(&mut buf, 123);
        writer.write_uint_backward(&mut buf, 22, 6);
        writer.write_bool_backward(&mut buf, false);
        // TODO: this implementation looks wrong, fix it!!
        // writer.write_uint_forward(&mut buf, 2, 4);

        assert_eq!(buf, [123, 0, 0, 0, 0, 0, 0, 0, 0, 45]);
    }

    #[test]
    fn nbits_written_calc() {
        let mut writer = BufferWriter::new(150);
        writer.bp_side = 140;
        writer.mask_side = 4;

        let nbits_side_written = writer.nbits_side_written(1200);

        assert_eq!(nbits_side_written, 74);
    }
}
