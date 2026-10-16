use crate::format::{Buffer, Format};
use crate::shape::{Axes, Shape};

/// Index into the engine's arena. Append-only arena, so IDs are in
/// topological order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub(crate) u32);

impl NodeId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Elementwise single-input functions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnaryFn {
    Neg,
    Sqrt,
    Exp,
    Log,
}

/// Elementwise two-input functions. Operands broadcast NumPy-style.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryFn {
    Add,
    Sub,
    Mul,
    Div,
}

/// Closed op set. Every variant has a hand-written `eval`, `vjp` and
/// `local_error` in `ops.rs`.
#[derive(Clone, Debug)]
pub enum Op {
    /// Input data
    Input { data: Box<[f64]> },

    /// Function on a single tensor
    Unary(UnaryFn, NodeId),

    /// Function on two tensors
    Binary(BinaryFn, NodeId, NodeId),

    /// General contraction (XLA `dot_general`). Covers matmul, batched
    /// matmul, tensordot, inner and outer products.
    Dot { lhs: NodeId, rhs: NodeId, dims: Box<DotDims> },

    /// Sum over `axes` (removed from the output). All axes = full reduction.
    Sum { x: NodeId, axes: Axes },
}

/// Axis roles for `Op::Dot`. Paired lists must have equal lengths and equal
/// dimension sizes. Output axes: batch, then lhs free, then rhs free.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DotDims {
    pub lhs_contract: Axes,
    pub rhs_contract: Axes,
    pub lhs_batch: Axes,
    pub rhs_batch: Axes,
}

/// Recorded once, never mutated.
#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub shape: Shape,
}

/// Rewritten on every evaluation. Parallel to `Node`, same index.
#[derive(Clone, Debug)]
pub struct NodeState {
    /// Format this node is assigned to run in.
    pub format: Format,
    /// Cached value. `None` when stale (e.g. after this node or an input was promoted).
    pub value: Option<Buffer>,
    /// Adjoint `ȳ` w.r.t. the demanded output, same length as the value.
    /// Kept in f64; it only needs a couple of correct digits.
    pub adjoint: Option<Box<[f64]>>,
    /// Filled by the error pass.
    pub err: Option<ErrInfo>,
}

impl NodeState {
    pub fn new(format: Format) -> Self {
        Self { format, value: None, adjoint: None, err: None }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ErrInfo {
    /// This node's first-order contribution to the output error:
    /// `Σₖ |ȳₖ| · |δₖ|`, with `|δ|` from `Op::local_error` in the current format.
    pub contribution: f64,
}
