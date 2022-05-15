use byteorder::{BigEndian, ByteOrder};

#[derive(Debug)]
pub enum BufferReaderError {
    ReadByteOutOfBounds(usize),
    ReadU24OutOfBounds(usize),
    BigEndianBitReaderReadUsizeNumBitsOutOfRange(usize, usize),
    BigEndianBitReaderReadBoolOutOfRange(usize),
}

#[derive(Default)]
pub struct BufferReader {
    head_byte_cursor: usize,
    tail_bit_cursor: usize,
}

// Big Endian buffer reader
// This reader can read bits from the tail end of the buffer
// (working its way towards the head) as well as reading bytes from the head of
// the buffer working its way to the tail.
// FIXME: return error if head cursor meets tail cursor
impl BufferReader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_at(head_byte_cursor: usize, tail_bit_cursor: usize) -> Self {
        Self {
            head_byte_cursor,
            tail_bit_cursor,
        }
    }

    pub fn get_tail_bit_cursor(&self) -> usize {
        self.tail_bit_cursor
    }

    pub fn get_head_byte_cursor(&self) -> usize {
        self.head_byte_cursor
    }

    pub fn read_head_byte(&mut self, buf: &[u8]) -> Result<u8, BufferReaderError> {
        match buf.get(self.head_byte_cursor) {
            Some(byte) => {
                self.head_byte_cursor += 1;
                Ok(*byte)
            }
            None => Err(BufferReaderError::ReadByteOutOfBounds(self.head_byte_cursor)),
        }
    }

    pub fn read_head_u24(&mut self, buf: &[u8]) -> Result<u32, BufferReaderError> {
        if self.head_byte_cursor + 2 < buf.len() {
            let value = BigEndian::read_u24(&buf[self.head_byte_cursor..]);
            self.head_byte_cursor += 3;
            Ok(value)
        } else {
            Err(BufferReaderError::ReadU24OutOfBounds(self.head_byte_cursor))
        }
    }

    // This function will read bits from the end of a buffer towards the front and interpret multi-byte payloads in BigEndian format
    pub fn read_tail_usize(&mut self, buf: &[u8], num_bits: usize) -> Result<usize, BufferReaderError> {
        let byte_index = self.tail_bit_cursor / 8;
        let bit_index = self.tail_bit_cursor % 8;

        let bits_left = 8 - bit_index;
        let add_bytes = if num_bits > bits_left && num_bits < 8 { 2 } else { 1 };
        let num_bytes = num_bits / 8 + add_bytes;

        if (buf.len() as i32 - self.head_byte_cursor as i32 - byte_index as i32 - num_bytes as i32) < 0 {
            return Err(BufferReaderError::BigEndianBitReaderReadUsizeNumBitsOutOfRange(
                num_bits, bit_index,
            ));
        }

        let from_index = buf.len() - byte_index - num_bytes;
        let to_index = from_index + num_bytes;
        let slice = &buf[from_index..to_index];

        let mut value = match num_bytes {
            1 => slice[0] as u32,
            2 => BigEndian::read_u16(slice) as u32,
            3 => BigEndian::read_u24(slice) as u32,
            4 => BigEndian::read_u32(slice),
            _ => 0_u32,
        };

        // shift the bits we want to ignore out of the way
        let shift_by = 32 - num_bits - bit_index;
        value <<= shift_by;
        value >>= shift_by + bit_index;

        self.tail_bit_cursor += num_bits;
        Ok(value as usize)
    }

    pub fn read_tail_bool(&mut self, buf: &[u8]) -> Result<bool, BufferReaderError> {
        let byte_index = self.tail_bit_cursor / 8;
        let bit_index = self.tail_bit_cursor % 8;

        // FIXME: test this range check properly
        if (buf.len() as i32 - self.head_byte_cursor as i32 - byte_index as i32 + 2) < 0 {
            return Err(BufferReaderError::BigEndianBitReaderReadBoolOutOfRange(bit_index));
        }

        let from_index = buf.len() - byte_index - 1;
        let mut byte = buf[from_index];
        byte <<= 7 - bit_index;
        byte >>= 7;

        self.tail_bit_cursor += 1;
        Ok(byte == 1)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn read_5_bits_over_byte_boundary_unto_usize() {
        let buf = [248, 52, 26, 166, 60];
        let mut reader = BufferReader::new();
        reader.tail_bit_cursor = 23;
        let value1 = reader.read_tail_usize(&buf, 5).unwrap();
        assert_eq!(8, value1);
    }

    #[test]
    fn read_multiple_values_from_bigendian_bitstream() {
        // value positions:         222    2222 2111
        let buf: [u8; 2] = [0b0001_1011, 0b0000_1100];
        let mut reader = BufferReader::new();

        // 0b0000_0100
        let value1 = reader.read_tail_usize(&buf, 3).unwrap();
        // 0b0110_0001
        let value2 = reader.read_tail_usize(&buf, 8).unwrap();

        assert_eq!(4, value1);
        assert_eq!(97, value2);
    }

    #[test]
    fn read_bool_from_bigendian_bitstream() {
        let byte = 0b0100_1000;
        let buf: [u8; 1] = [byte];
        let mut reader = BufferReader::new();

        let value0 = reader.read_tail_bool(&buf).unwrap();
        let value1 = reader.read_tail_bool(&buf).unwrap();
        let value2 = reader.read_tail_bool(&buf).unwrap();
        let value3 = reader.read_tail_bool(&buf).unwrap();
        let value4 = reader.read_tail_bool(&buf).unwrap();
        let value5 = reader.read_tail_bool(&buf).unwrap();
        let value6 = reader.read_tail_bool(&buf).unwrap();
        let value7 = reader.read_tail_bool(&buf).unwrap();

        assert_eq!(false, value0);
        assert_eq!(false, value1);
        assert_eq!(false, value2);
        assert_eq!(true, value3);
        assert_eq!(false, value4);
        assert_eq!(false, value5);
        assert_eq!(true, value6);
        assert_eq!(false, value7);
    }
}
