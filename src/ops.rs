//! Per-op math: shape rules, evaluation, VJPs and local rounding-error bounds.

use crate::error::Result;
use crate::float_node::{NodeId, Op};
use crate::format::{Buffer, Format};
use crate::shape::Shape;

/// An evaluated operand: value plus its shape.
#[derive(Copy, Clone, Debug)]
pub struct TensorRef<'a> {
    pub buf: &'a Buffer,
    pub shape: &'a Shape,
}

impl Op {
    /// Operand node IDs, in order.
    pub fn inputs(&self) -> impl Iterator<Item = NodeId> {
        let ins = match *self {
            Op::Input { .. } => [None, None],
            Op::Unary(_, x) | Op::Sum { x, .. } => [Some(x), None],
            Op::Binary(_, a, b) | Op::Dot { lhs: a, rhs: b, .. } => [Some(a), Some(b)],
        };
        ins.into_iter().flatten()
    }

    /// Output shape from operand shapes. Errors on incompatible shapes or bad axes.
    pub fn out_shape(&self, ins: &[&Shape]) -> Result<Shape> {
        todo!()
    }

    /// Evaluate in `fmt`. Operands may be in other formats; they are converted to `fmt` first.
    pub fn eval(&self, ins: &[TensorRef<'_>], fmt: Format) -> Buffer {
        todo!()
    }

    /// Vector-Jacobian product: given the output's adjoint `cot`, return one
    /// adjoint per operand (same order as `inputs()`). Broadcast operands get
    /// their adjoint summed back over the broadcast axes.
    pub fn vjp(&self, ins: &[TensorRef<'_>], out: TensorRef<'_>, cot: &[f64]) -> Vec<Box<[f64]>> {
        todo!()
    }

    /// Elementwise bound on `|δ|`, this op's own rounding error in `fmt`
    /// assuming exact operands. Same length as the output.
    ///
    /// Input: `u·|x|`. Elementwise: `u·|y|` (a few ulps for Exp/Log).
    /// Dot: `γₖ·(|A| ⋅ |B|)`, `k` = product of contracted dims.
    /// Sum: `γₙ·Σ|xᵢ|`, `n` = product of reduced dims.
    pub fn local_error(&self, ins: &[TensorRef<'_>], out: TensorRef<'_>, fmt: Format) -> Box<[f64]> {
        todo!()
    }
}

/// Probabilistic accumulation constant `γₙ ≈ √n · u` (Higham–Mary), used
/// instead of the worst-case `n · u`.
pub fn gamma(n: usize, u: f64) -> f64 {
    (n as f64).sqrt() * u
}
