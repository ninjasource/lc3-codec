use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
pub type Scaler = f32;

#[derive(Debug, Clone, Copy, Default)]
pub struct Complex {
    pub r: Scaler,
    pub i: Scaler,
}

impl Complex {
    pub fn new(r: Scaler, i: Scaler) -> Self {
        Self { r, i }
    }
}

impl Mul for Complex {
    type Output = Complex;
    fn mul(self, rhs: Complex) -> Self::Output {
        Complex {
            r: self.r * rhs.r - self.i * rhs.i,
            i: self.r * rhs.i + self.i * rhs.r,
        }
    }
}

impl Add for Complex {
    type Output = Complex;
    fn add(self, rhs: Self) -> Self::Output {
        Complex {
            r: self.r + rhs.r,
            i: self.i + rhs.i,
        }
    }
}

impl Sub for Complex {
    type Output = Complex;
    fn sub(self, rhs: Self) -> Self::Output {
        Complex {
            r: self.r - rhs.r,
            i: self.i - rhs.i,
        }
    }
}

impl MulAssign<Scaler> for Complex {
    fn mul_assign(&mut self, rhs: Scaler) {
        self.r *= rhs;
        self.i *= rhs;
    }
}

impl AddAssign<Complex> for Complex {
    fn add_assign(&mut self, rhs: Complex) {
        self.r += rhs.r;
        self.i += rhs.i;
    }
}
impl SubAssign<Complex> for Complex {
    fn sub_assign(&mut self, rhs: Complex) {
        self.r -= rhs.r;
        self.i -= rhs.i;
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn complex_arithmatic() {
        let mut complex = Complex::new(2.0, 3.0);

        // complex addition
        complex += Complex::new(1., 2.);
        assert_eq!(complex.r, 3.0);
        assert_eq!(complex.i, 5.0);
        complex = complex + Complex::new(1., 2.);
        assert_eq!(complex.r, 4.0);
        assert_eq!(complex.i, 7.0);

        // complex subtraction
        complex -= Complex::new(9.0, 15.0);
        assert_eq!(complex.r, -5.0);
        assert_eq!(complex.i, -8.0);
        complex = complex - Complex::new(9.0, 15.0);
        assert_eq!(complex.r, -14.0);
        assert_eq!(complex.i, -23.0);

        // complex multiplication
        complex = complex * Complex::new(2.0, 4.0);
        assert_eq!(complex.r, 64.0);
        assert_eq!(complex.i, -102.0);

        // scalar multiplication
        complex *= 2.0;
        assert_eq!(complex.r, 128.0);
        assert_eq!(complex.i, -204.0);
    }
}
