use smallvec::SmallVec;

/// Axis indices into a shape.
pub type Axes = SmallVec<[usize; 4]>;

/// Tensor shape of any rank. Rank 0 is a scalar. Storage is contiguous
/// row-major; there are no strided views.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Shape(SmallVec<[usize; 4]>);

impl Shape {
    pub fn scalar() -> Self {
        Self::default()
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Number of elements (1 for a scalar).
    pub fn len(&self) -> usize {
        self.0.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_scalar(&self) -> bool {
        self.rank() == 0
    }

    /// NumPy broadcasting: align trailing axes; each pair must be equal or contain a 1.
    pub fn broadcast(&self, other: &Shape) -> Option<Shape> {
        let rank = self.rank().max(other.rank());
        let dim = |s: &Shape, i: usize| {
            let pad = rank - s.rank();
            if i < pad { 1 } else { s.0[i - pad] }
        };
        (0..rank)
            .map(|i| match (dim(self, i), dim(other, i)) {
                (a, b) if a == b => Some(a),
                (1, b) => Some(b),
                (a, 1) => Some(a),
                _ => None,
            })
            .collect::<Option<_>>()
            .map(Shape)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape(SmallVec::from_slice(dims))
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Shape(SmallVec::from_slice(&dims))
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape(SmallVec::from_vec(dims))
    }
}
