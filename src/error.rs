use std::fmt;

use crate::float_node::NodeId;
use crate::shape::{Axes, Shape};

/// Engine errors. A plain enum (not `anyhow`) so a future C ABI can map each
/// variant to an error code.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineError {
    /// Input data length does not match the declared shape.
    DataLength { expected: usize, got: usize },
    /// Operand shapes are incompatible for the op.
    ShapeMismatch { op: &'static str, lhs: Shape, rhs: Shape },
    /// Axis out of range, repeated, or mismatched in size.
    InvalidAxes { op: &'static str, shape: Shape, axes: Axes },
    /// `NodeId` does not belong to this engine.
    UnknownNode(NodeId),
    /// Demanded output is not a scalar (milestone 1 supports scalar outputs only).
    NotScalar(Shape),
    /// Every node is already in the highest format and the estimate still exceeds the tolerance.
    ToleranceUnreachable { estimate: f64, tol: f64 },
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DataLength { expected, got } => {
                write!(f, "data length {got} does not match shape length {expected}")
            }
            Self::ShapeMismatch { op, lhs, rhs } => {
                write!(f, "shape mismatch in {op}: {lhs:?} vs {rhs:?}")
            }
            Self::InvalidAxes { op, shape, axes } => {
                write!(f, "invalid axes {axes:?} for {op} on shape {shape:?}")
            }
            Self::UnknownNode(id) => write!(f, "unknown node {id:?}"),
            Self::NotScalar(shape) => write!(f, "demanded output is not a scalar: {shape:?}"),
            Self::ToleranceUnreachable { estimate, tol } => {
                write!(f, "error estimate {estimate:e} exceeds tolerance {tol:e} at highest precision")
            }
        }
    }
}

impl std::error::Error for EngineError {}

pub type Result<T> = std::result::Result<T, EngineError>;
