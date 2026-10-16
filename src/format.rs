use half::{bf16, f16};

/// Hardware float formats, ordered by precision (unit roundoff).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Format {
    Bf16,
    F16,
    F32,
    F64,
}

impl Format {
    pub const LOWEST: Format = Format::Bf16;
    pub const HIGHEST: Format = Format::F64;

    /// Unit roundoff `u = 2^-p` for a `p`-bit significand (round-to-nearest).
    pub fn unit_roundoff(self) -> f64 {
        match self {
            Format::Bf16 => 2f64.powi(-8),
            Format::F16 => 2f64.powi(-11),
            Format::F32 => 2f64.powi(-24),
            Format::F64 => 2f64.powi(-53),
        }
    }

    /// Next format by precision. Note `Bf16 -> F16` gains precision but loses
    /// range (f16 overflows above 65504).
    pub fn promote(self) -> Option<Format> {
        match self {
            Format::Bf16 => Some(Format::F16),
            Format::F16 => Some(Format::F32),
            Format::F32 => Some(Format::F64),
            Format::F64 => None,
        }
    }
}

/// A node's computed value: flat row-major storage tagged by its format.
/// The variant *is* the format, so a value/format mismatch is unrepresentable.
#[derive(Clone, Debug)]
pub enum Buffer {
    Bf16(Box<[bf16]>),
    F16(Box<[f16]>),
    F32(Box<[f32]>),
    F64(Box<[f64]>),
}

impl Buffer {
    pub fn format(&self) -> Format {
        match self {
            Buffer::Bf16(_) => Format::Bf16,
            Buffer::F16(_) => Format::F16,
            Buffer::F32(_) => Format::F32,
            Buffer::F64(_) => Format::F64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Buffer::Bf16(v) => v.len(),
            Buffer::F16(v) => v.len(),
            Buffer::F32(v) => v.len(),
            Buffer::F64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Round f64 data into `fmt`.
    pub fn from_f64(data: &[f64], fmt: Format) -> Buffer {
        match fmt {
            Format::Bf16 => Buffer::Bf16(data.iter().map(|&x| bf16::from_f64(x)).collect()),
            Format::F16 => Buffer::F16(data.iter().map(|&x| f16::from_f64(x)).collect()),
            Format::F32 => Buffer::F32(data.iter().map(|&x| x as f32).collect()),
            Format::F64 => Buffer::F64(data.into()),
        }
    }

    /// Widen to f64 (exact for every format).
    pub fn to_f64(&self) -> Box<[f64]> {
        match self {
            Buffer::Bf16(v) => v.iter().map(|x| x.to_f64()).collect(),
            Buffer::F16(v) => v.iter().map(|x| x.to_f64()).collect(),
            Buffer::F32(v) => v.iter().map(|&x| x as f64).collect(),
            Buffer::F64(v) => v.clone(),
        }
    }
}
