use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::error::{EngineError, Result};
use crate::float_node::{BinaryFn, DotDims, Node, NodeId, NodeState, Op, UnaryFn};
use crate::format::Format;
use crate::shape::{Axes, Shape};

/// Owns every node. The core API works on `NodeId` so a C ABI can wrap it.
///
/// Interior mutability lets `Copy` handles record ops through `&self`.
/// This makes the engine `!Sync`: one engine per thread.
#[derive(Default)]
pub struct FloatEngine {
    nodes: RefCell<Vec<Node>>,       // arena, append-only
    state: RefCell<Vec<NodeState>>,  // parallel to `nodes`, same indices
}

impl FloatEngine {
    pub fn new() -> Self {
        Self::default()
    }

    // ---- recording -------------------------------------------------------

    /// Record an input. `data` is row-major and kept in f64.
    pub fn input(&self, data: impl Into<Box<[f64]>>, shape: impl Into<Shape>) -> Result<NodeId> {
        todo!()
    }

    pub fn unary(&self, f: UnaryFn, x: NodeId) -> Result<NodeId> {
        self.push(Op::Unary(f, x))
    }

    pub fn binary(&self, f: BinaryFn, a: NodeId, b: NodeId) -> Result<NodeId> {
        self.push(Op::Binary(f, a, b))
    }

    pub fn dot(&self, lhs: NodeId, rhs: NodeId, dims: DotDims) -> Result<NodeId> {
        self.push(Op::Dot { lhs, rhs, dims: Box::new(dims) })
    }

    /// Contract the last axis of `a` with the first axis of `b` (`np.tensordot(a, b, 1)`).
    /// For rank 2 this is ordinary matmul.
    pub fn matmul(&self, a: NodeId, b: NodeId) -> Result<NodeId> {
        todo!()
    }

    /// Sum over all axes, giving a scalar.
    pub fn sum(&self, x: NodeId) -> Result<NodeId> {
        todo!()
    }

    pub fn sum_axes(&self, x: NodeId, axes: &[usize]) -> Result<NodeId> {
        self.push(Op::Sum { x, axes: Axes::from_slice(axes) })
    }

    pub fn shape(&self, x: NodeId) -> Result<Shape> {
        todo!()
    }

    pub fn handle(&self, id: NodeId) -> Handle<'_> {
        Handle { id, engine: self }
    }

    // ---- demand ----------------------------------------------------------

    /// Evaluate scalar output `out` with estimated absolute error ≤ `tol`.
    ///
    /// Loop: forward pass at the current formats, adjoint pass, per-node error
    /// contributions; if the total exceeds `tol`, promote the worst nodes and repeat.
    pub fn value(&self, out: NodeId, tol: f64) -> Result<f64> {
        todo!()
    }

    // ---- internals -------------------------------------------------------

    /// Validate operands, compute the output shape, append the node at `Format::LOWEST`.
    fn push(&self, op: Op) -> Result<NodeId> {
        todo!()
    }

    /// Evaluate stale nodes in `0..=out` in their assigned formats.
    fn forward(&self, out: NodeId) {
        todo!()
    }

    /// Reverse scan from `out` (seed adjoint 1), filling `NodeState::adjoint`.
    fn adjoint(&self, out: NodeId) {
        todo!()
    }

    /// Fill `NodeState::err` for `0..=out` and return the total estimate `Σᵢ contributionᵢ`.
    fn estimate_error(&self, out: NodeId) -> f64 {
        todo!()
    }

    /// Promote nodes until the predicted total is within `tol`, invalidating
    /// their values and their dependents'. Returns `false` if nothing could be promoted.
    fn promote(&self, out: NodeId, tol: f64) -> bool {
        todo!()
    }
}

/// Lightweight user-facing handle with overloaded operators.
/// Operators panic on shape mismatch; use the `FloatEngine` methods for `Result`s.
#[derive(Copy, Clone)]
pub struct Handle<'e> {
    id: NodeId,
    engine: &'e FloatEngine,
}

impl<'e> Handle<'e> {
    pub fn id(self) -> NodeId {
        self.id
    }

    pub fn shape(self) -> Shape {
        self.engine.shape(self.id).expect("handle refers to its own engine")
    }

    pub fn matmul(self, rhs: Handle<'e>) -> Handle<'e> {
        self.lift(self.engine.matmul(self.id, rhs.id))
    }

    pub fn dot(self, rhs: Handle<'e>, dims: DotDims) -> Handle<'e> {
        self.lift(self.engine.dot(self.id, rhs.id, dims))
    }

    pub fn sum(self) -> Handle<'e> {
        self.lift(self.engine.sum(self.id))
    }

    pub fn sum_axes(self, axes: &[usize]) -> Handle<'e> {
        self.lift(self.engine.sum_axes(self.id, axes))
    }

    pub fn sqrt(self) -> Handle<'e> {
        self.lift(self.engine.unary(UnaryFn::Sqrt, self.id))
    }

    pub fn exp(self) -> Handle<'e> {
        self.lift(self.engine.unary(UnaryFn::Exp, self.id))
    }

    pub fn ln(self) -> Handle<'e> {
        self.lift(self.engine.unary(UnaryFn::Log, self.id))
    }

    pub fn value(self, tol: f64) -> Result<f64> {
        self.engine.value(self.id, tol)
    }

    fn lift(self, r: Result<NodeId>) -> Handle<'e> {
        match r {
            Ok(id) => Handle { id, engine: self.engine },
            Err(e) => panic!("{e}"),
        }
    }
}

impl<'e> Neg for Handle<'e> {
    type Output = Handle<'e>;
    fn neg(self) -> Handle<'e> {
        self.lift(self.engine.unary(UnaryFn::Neg, self.id))
    }
}

macro_rules! binary_operator {
    ($trait:ident, $method:ident, $f:expr) => {
        impl<'e> $trait for Handle<'e> {
            type Output = Handle<'e>;
            fn $method(self, rhs: Handle<'e>) -> Handle<'e> {
                assert!(std::ptr::eq(self.engine, rhs.engine), "handles from different engines");
                self.lift(self.engine.binary($f, self.id, rhs.id))
            }
        }
    };
}

binary_operator!(Add, add, BinaryFn::Add);
binary_operator!(Sub, sub, BinaryFn::Sub);
binary_operator!(Mul, mul, BinaryFn::Mul);
binary_operator!(Div, div, BinaryFn::Div);
