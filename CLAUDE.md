# floating_point

Lazy floating-point runtime in Rust that picks precision per operation from an output error target.

## Status

Empty scaffold (`src/main.rs` returns "Not Implemented"). Everything below is design intent, not existing code.

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
- **Coarse nodes.** One node per scalar op costs 10–100x the flop. Nodes should be vector/tensor ops.
- **Hardware formats only.** Precision choices are snapped to bf16 / f16 / f32 / f64 (maybe f128 via software). No arbitrary bit widths.
- **Error estimate:** first-order adjoint propagation of rounding error,
  `err(y) ≈ Σᵢ |∂y/∂vᵢ| · |vᵢ| · uᵢ`, where `uᵢ` is the unit roundoff of the format node `i` ran in. Call it "adjoint/sensitivity error", not "backward error" (that means something else in numerical analysis). First-order only: unreliable near singularities.

## Known hard problems

- **Branches on float values** (`if x < 0.0`) force evaluation mid-graph, and a later higher-precision recompute can flip the comparison and invalidate the graph built after it. Options: CGAL-style filtered predicates (certify the comparison at whatever precision is needed before branching), or require comparisons to certify.
- **Loops grow the graph unboundedly.** Need checkpointing (as in reverse-mode AD) or loop-level nodes.
- **Per-node overhead** must be amortised by coarse ops.

## Plan

1. Narrow, branch-free domain first: dense linear algebra or a fixed-step ODE integrator.
2. Success criterion: within tolerance of an all-f64 baseline while running mostly in f32/bf16.
3. Control flow is the research problem after that.

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
