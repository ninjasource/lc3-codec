use byteorder::{BigEndian, ByteOrder, LittleEndian};

#[derive(Default)]
pub struct BufferWriter {
    head_byte_cursor: usize,
    tail_bit_cursor: usize,
}

#[derive(Debug)]
pub enum BufferWriteError {
    WriteByteOutOfBounds(usize),
}

impl BufferWriter {
    pub fn new() -> Self {
        BufferWriter::default()
    }

    pub fn write_uint_backward(&mut self, buf: &mut [u8], value: u32, num_bits: usize) {
        let byte_index = self.tail_bit_cursor / 8;
        let bit_index = self.tail_bit_cursor % 8;
        let existing = buf[buf.len() - byte_index - 1];
        let new_value = (value << bit_index) | existing as u32;
        let byte_start = (num_bits / 8) + if num_bits % 8 == 0 { 0 } else { 1 };
        LittleEndian::write_u32(buf, new_value);
        self.tail_bit_cursor += num_bits;
        //let mut tmp_buffer = [0; 4];
        //BigEndian::write_u32(&mut tmp_buffer, value);
    }

    pub fn write_bool_backward(&mut self, _value: bool) {
        // TODO: implement this!!!
    }

    pub fn write_byte_forward(&mut self, buf: &mut [u8], value: u8) {
        buf[self.head_byte_cursor] = value;
        self.head_byte_cursor += 1;
    }

    pub fn write_uint_forward(&mut self, buf: &mut [u8], value: usize, num_bits: usize) {
        if num_bits % 8 != 0 {
            panic!("num_bits dows not fall on a byte boundary: {}", num_bits);
        }

        let num_bytes = num_bits / 8;
        match num_bytes {
            1 => self.write_byte_forward(buf, value as u8),
            2 => {
                BigEndian::write_u16(&mut buf[self.head_byte_cursor..], value as u16);
                self.head_byte_cursor += 2;
            }
            3 => {
                BigEndian::write_u24(&mut buf[self.head_byte_cursor..], value as u32);
                self.head_byte_cursor += 3;
            }
            4 => {
                BigEndian::write_u32(&mut buf[self.head_byte_cursor..], value as u32);
                self.head_byte_cursor += 4;
            }
            _ => panic!("num_bits not supported: {}", num_bits),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn write_tail_usize_to_3_bits() {
        let mut writer = BufferWriter::new();
        let mut buf = [0; 8];
        writer.write_uint_backward(&mut buf, 5, 10);

        println!("{:?}", buf);
    }
}
