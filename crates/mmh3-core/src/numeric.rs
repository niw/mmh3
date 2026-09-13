//! Conversions between f32 and the 16-bit float formats stored in checkpoints.

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Rounds to nearest, ties to even. NaN stays NaN.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

/// Rounds to nearest, ties to even, with subnormals and overflow to infinity. NaN stays NaN.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;
    if exponent == 0xFF {
        return sign | 0x7C00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1F {
        return sign | 0x7C00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let full = mantissa | 0x80_0000;
        let shift = (14 - half_exponent) as u32;
        let rounded = (full >> shift) + round_bit(full, shift);
        return sign | rounded as u16;
    }
    let rounded = ((half_exponent as u32) << 10 | (mantissa >> 13)) + round_bit(mantissa, 13);
    sign | rounded as u16
}

/// 1 when dropping the low `shift` bits of `value` rounds up under ties-to-even.
fn round_bit(value: u32, shift: u32) -> u32 {
    let half = 1 << (shift - 1);
    let remainder = value & ((1 << shift) - 1);
    let kept = value >> shift;
    u32::from(remainder > half || (remainder == half && kept & 1 == 1))
}

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;
    let magnitude = match exponent {
        0 if mantissa == 0 => 0,
        0 => {
            // Subnormal: normalize the mantissa into an f32 exponent.
            let shift = mantissa.leading_zeros() - 21;
            let normalized = (mantissa << shift) & 0x3FF;
            ((127 - 15 + 1 - shift) << 23) | (normalized << 13)
        }
        0x1F => 0x7F80_0000 | (mantissa << 13),
        _ => ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(sign | magnitude)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_half_precision_values() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x7BFF), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x03FF), 1023.0 * 2f32.powi(-24));
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7E00).is_nan());
        assert_eq!(f16_to_f32(0x8000).to_bits(), 0x8000_0000);
    }

    #[test]
    fn rounds_to_half_precision() {
        for bits in [0x0000u16, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x3C01, 0x7BFF, 0x8001, 0xC000] {
            assert_eq!(f32_to_f16(f16_to_f32(bits)), bits, "{bits:#06x}");
        }
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3C00);
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3C02);
        assert_eq!(f32_to_f16(70000.0), 0x7C00);
        assert_eq!(f32_to_f16(1e-9), 0x0000);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn round_trips_bfloat16() {
        for value in [1.0f32, -3.5, 0.1, 1e-20, 65504.0] {
            let rounded = bf16_to_f32(f32_to_bf16(value));
            assert!((rounded - value).abs() <= value.abs() / 128.0);
        }
        assert_eq!(f32_to_bf16(1.0 + 2f32.powi(-8)), 0x3F80);
        assert_eq!(f32_to_bf16(1.0 + 3.0 * 2f32.powi(-8)), 0x3F82);
    }
}
