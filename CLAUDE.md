# floating_point

Lazy floating-point runtime in Rust that picks precision per operation from an output error target.

## Status

Type skeleton only: types and signatures exist, function bodies are mostly `todo!()`. `src/main.rs` still returns "Not Implemented".

## Module layout

- `src/lib.rs`: module root.
- `src/error.rs`: `EngineError` (plain enum, maps to C error codes later) and `Result`.
- `src/format.rs`: `Format` (bf16/f16/f32/f64, unit roundoff) and `Buffer` (tensor storage tagged by format).
- `src/shape.rs`: `Shape` (any rank, contiguous row-major, NumPy broadcasting) and `Axes`.
- `src/float_node.rs`: `NodeId`, `Op`, `DotDims`, `UnaryFn`, `BinaryFn`, `Node` (immutable), `NodeState` (per evaluation), `ErrInfo`.
- `src/ops.rs`: per-op math as `impl Op`: `out_shape`, `eval`, `vjp`, `local_error`.
- `src/float_engine.rs`: `FloatEngine` (arena + record/demand API on `NodeId`) and `Handle<'e>` (Rust-only operator sugar).

## Idea

- A central engine owns all float values. User code holds handles, not values.
- Operations are recorded into a graph at runtime instead of being evaluated eagerly.
- When a value is demanded externally (read, printed, compared, passed out of the engine), the engine:
  1. evaluates the subgraph cheaply (low-precision hardware formats),
  2. runs a reverse (adjoint) pass to estimate each node's contribution to the output error,
  3. recomputes the nodes that blow the error budget in a higher hardware format.
- Optionally emits per-node error traces for graphing/debugging without formal analysis.

Why lazy: eager upcasting can't work. Rounding error already incurred is not recovered by promoting later, and cancellation exposes error introduced upstream. Only once the whole graph is known can you tell where precision was actually needed.

## Design decisions

- **Lazy nodes, not Rust `async` futures.** Async is for I/O scheduling and adds overhead for nothing. Use an arena of nodes plus lightweight `Copy` handles with overloaded operators (JAX-tracing / PyTorch-lazy-tensor style).
- **Coarse tensor nodes.** One node per scalar op costs 10–100x the flop. Each node is an op on N-d tensors (rank 0 = scalar). Contiguous row-major storage, no strided views. Shape logic stays inside `ops.rs` / `shape.rs`.
- **Ops are not limited to elementwise.** A node can be any tensor function (e.g. `Matrix → Scalar`). Matrices of `NodeId` (one node per element) were considered and rejected because of per-node overhead.
- **Closed, hard-coded op set.** `Op` is a closed enum: `Input`, `Unary(UnaryFn, _)`, `Binary(BinaryFn, _, _)` (elementwise add/sub/mul/div, NumPy broadcasting), `Dot` (XLA `dot_general`: contracting + batch axes per side; covers matmul, batched matmul, tensordot, inner/outer products), `Sum { axes }`. Every variant has a hand-written VJP and local error bound, enforced by exhaustive `match`. A `TensorOp` trait for user-defined ops comes later, if ever.
- **Hardware formats only.** Precision choices are snapped to bf16 / f16 / f32 / f64 (maybe f128 via software). No arbitrary bit widths.
- **Inputs keep their f64 data** in `Op::Input` so promotion can always recompute from the original values.
- **Structure vs state.** `Node` (op + shape) is recorded once and never mutated. `NodeState` (value, adjoint, error) is rewritten per evaluation. They live in parallel arrays indexed by `NodeId`. The arena is append-only, so IDs are already in topological order: forward pass = forward scan, adjoint pass = reverse scan.
- **Core API on `NodeId`, not `Handle`.** `FloatEngine` methods take and return `NodeId`. `Handle<'e>` with overloaded operators is Rust-only sugar. This keeps a future C ABI a one-to-one wrapper.

## Error model

Derivatives and errors are one reverse pass. Treat node `i`'s own rounding as a perturbation `δᵢ` on its output. To first order:

```
err(y) ≈ Σᵢ ⟨ȳᵢ, δᵢ⟩  ≤  Σᵢ Σₖ |ȳᵢₖ| · |δᵢₖ|
```

`ȳᵢ` is node `i`'s adjoint (same shape as its value). Each op supplies a `vjp` and a `local_error`: an elementwise bound on `|δᵢ|`, its own rounding error given exact inputs. Input errors are handled by the adjoint pass, so no per-op condition number is needed (that would double-count).

| Op | `local_error` bound |
|---|---|
| Input (rounding to format) | `u · |x|` |
| Elementwise `Binary` / `Unary` | `u · |y|` (a few ulps for `Exp`/`Log`) |
| `Dot` (`k` = product of contracted dims) | `γₖ · (|A| ⋅ |B|)` |
| `Sum` (`n` = product of reduced dims) | `γₙ · Σ|xᵢ|` |

- Use `|A|·|B|`, not `|AB|`: cancellation (`|AB| ≪ |A|·|B|`) is exactly when error is large.
- Use the probabilistic constant `γₙ ≈ √n · u` (Higham–Mary), not worst-case `n · u`. Worst case says bf16 matmul with `k = 1000` is always useless, which is false. We choose precision; we don't certify.
- Call it "adjoint/sensitivity error", not "backward error" (that means something else in numerical analysis).
- First-order only. Invalid once `κ · u ≳ 1` (κ below ~250 in bf16).
- Scalar outputs only for milestone 1. Vector outputs need one reverse pass per element, a random seed, or a conservative `|J|ᵀ·1` pass. Undecided.
- Open: a separate accumulator format for `Dot` (bf16 inputs, f32 accumulate). If added, `Dot`'s bound uses `γₖ(u_acc)`.

Later, optional: shadow-evaluate sampled nodes in a higher format to measure `δᵢ` (trace mode, calibration); interval/ball arithmetic to certify the final output only.

## Known hard problems

- **Branches on float values** (`if x < 0.0`) force evaluation mid-graph, and a later higher-precision recompute can flip the comparison and invalidate the graph built after it. Options: CGAL-style filtered predicates (certify the comparison at whatever precision is needed before branching), or require comparisons to certify.
- **Loops grow the graph unboundedly.** Need checkpointing (as in reverse-mode AD) or loop-level nodes.
- **Per-node overhead** must be amortised by coarse ops.
- **`Det`, `Inv`, `Solve` are deferred.** The derivatives are easy (reuse the LU; `NodeState` would need to store factors). The error model is hard: forward error ≈ `κ(A) · n · ρ · u`, with κ only estimable and the pivot growth factor ρ only known after factoring. First-order breaks at κ ≳ 1/u, which is exactly when low precision fails. `Det` also over/underflows and has no VJP at singular matrices. The right mixed-precision `Solve` is iterative refinement (Carson–Higham), a different mechanism from per-node recompute, so it likely becomes its own built-in algorithm.

## Plan

1. Milestone 1: dense linear algebra with the fixed op set above, scalar outputs, model-based error only.
2. Success criterion: within tolerance of an all-f64 baseline while running mostly in f32/bf16.
3. Then: vector outputs, accumulator formats, trace mode, `Solve` via iterative refinement.
4. C++ support: `extern "C"` API over the `NodeId` core (cbindgen) plus a header-only C++ wrapper with RAII and operator overloading. No panics across FFI (`catch_unwind`, error codes); f64 in/out at the boundary; one engine per thread (the engine is `!Sync`). Built after milestone 1.
5. Control flow is the research problem after that.

## Prior art (read before designing)

- **Boehm's constructive reals** (Android calculator): lazy nodes, demand-driven precision propagated top-down. Same idea with arbitrary precision.
- **CGAL `Lazy_exact_nt`**: float-interval evaluation while recording a DAG; exact replay only when the interval can't certify the answer.
- **ADAPT (SC'18), CHEF-FP**: reverse-mode AD to estimate per-variable rounding-error contribution and choose mixed precision.
- **CADNA / PROMISE, Verificarlo**: dynamic error estimation and precision tuning for C/C++/Fortran (stochastic arithmetic / instrumentation).
- **Arb/FLINT, Shewchuk predicates, iRRAM**: ball arithmetic and recompute-on-failure.
- **FPTuner, Precimonious, Herbie**: static precision tuning / expression rewriting. Not what this is.

Novel part: lazy graph + adjoint error estimate + recompute restricted to hardware formats, in Rust.

## Working conventions

- Code changes go on a branch as a PR; never commit to `main`.
- Keep the owner informed of architecture: types, struct fields, function signatures, module layout. Function bodies matter less.
- Coding replies terse.
