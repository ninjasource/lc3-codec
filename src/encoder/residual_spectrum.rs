use crate::common::complex::Scaler;
use bitvec::prelude::*;
use core::iter::Take;

pub struct ResidualBitsEncoder {
    // NOTE: there can be no more than `ne` residual bits (400 lines)
    // This can be confirmed by looking at the algorithm in encode
    // 400 bits / 8 bits per byte = 50 bytes
    res_bits: BitArray<[u8; 50], Lsb0>,
}

pub struct ResidualBits<'a> {
    inner: Take<bitvec::slice::Iter<'a, u8, LocalBits>>,
}

impl<'a> Iterator for ResidualBits<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|x| *x)
    }
}

impl Default for ResidualBitsEncoder {
    fn default() -> Self {
        Self {
            res_bits: BitArray::new([0; 50]),
        }
    }
}

impl ResidualBitsEncoder {
    pub fn encode(
        &mut self,
        nbits_spec: usize,
        nbits_spec_trunc: usize,
        ne: usize,
        gg: Scaler,
        tns_xf: &[Scaler],
        spec_quant_xq: &[i16],
    ) -> ResidualBits {
        let nbits_residual_max = nbits_spec as i32 - nbits_spec_trunc as i32 + 4;
        let nbits_residual_max = 0.max(nbits_residual_max) as usize;
        let mut nbits_residual = 0;

        if nbits_residual_max > 0 {
            for (tns, spec_quant) in tns_xf[..ne].iter().zip(spec_quant_xq[..ne].iter()) {
                if nbits_residual >= nbits_residual_max {
                    break;
                }

                if *spec_quant != 0 {
                    let res_bit = *tns >= *spec_quant as Scaler * gg as Scaler;
                    self.res_bits.set(nbits_residual, res_bit);
                    nbits_residual += 1;
                }
            }
        }

        let inner = self.res_bits.as_bitslice().iter().take(nbits_residual);
        ResidualBits { inner }
    }
}
