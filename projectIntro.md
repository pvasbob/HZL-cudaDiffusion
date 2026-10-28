# CUDA-Accelerated Diffusion Prediction and Temperature Control

## Project overview

This project develops a low-latency scientific computing application that predicts and controls the diffusion of coloring materials in glass. It combines numerical PDE simulation, GPU image processing, image-based state reconstruction, quantitative optimization, and predictive feedback control.

The application follows one evolving diffusion process. Successive images describe its current pattern. From each image, the software reconstructs the current material distribution, forecasts future evolution under candidate temperature controls, and selects an adjustment that moves the predicted outcome closer to a desired final pattern.

The central capability is **anticipation**: showing where the process is heading and adjusting temperature before an unwanted pattern develops.

The implementation targets an NVIDIA RTX 3080 using C++ and CUDA. Established libraries support standard functionality, while specialized numerical operations and measured performance opportunities motivate custom CUDA implementations.

## Physical motivation

Certain metal ions can enter glass during heat treatment. Their distribution, chemical state, and possible particle formation can influence coloration. Temperature-dependent transport therefore provides a basis for influencing the development of a material pattern.

The initial model represents a localized material deposit spreading through an effective thin glass layer. Its temperature distribution determines local diffusivity.

Three quantities remain distinct:

- The physical concentration field represented by the model.
- A synthetic image encoding that concentration.
- The optical appearance of real manufactured glass.

The first implementation addresses concentration evolution and synthetic image formation. Detailed chemical reactions, calibrated optical appearance, and three-dimensional penetration are future extensions.

Temperature changes material mobility. It is not modeled as an artificial force attracting material toward hotter locations.

## Objectives

The project has seven primary objectives:

1. Implement a numerically correct temperature-dependent diffusion solver.
2. Simulate substantially faster than the physical evolution being predicted.
3. Process successive images of one chronological diffusion process.
4. Reconstruct the current concentration from those images.
5. Automatically optimize temperature controls toward a target pattern.
6. Provide interactive visualization and quantitative diagnostics.
7. Establish reproducible CUDA benchmarks and evidence-based optimization comparisons.

The physical model is deliberately simplified, while the software follows production-quality engineering practices.

## Software-based experimental environment

The initial system is entirely self-contained. It generates:

- Initial material concentration.
- Initial temperature distribution.
- Desired final concentration patterns.
- Successive images of the evolving process.

No camera, thermal sensor, real glass photographs, oven communication, or physical heating hardware is required.

The observation interface is designed so that real image acquisition can replace synthetic observations later.

At this stage, the ongoing simulation is assumed to represent the process accurately. A separate virtual plant with different physics is not required. Model mismatch, measurement noise, and experimental parameter calibration remain optional extensions.

## Chronological process and observations

The application maintains one ongoing concentration field. Its next state depends on its present state and the temperature control applied during the next interval:

\[
c_{k+1}
=
F\left(c_k,T_k,\Delta t_{\mathrm{control}}\right).
\]

Here:

- \(c_k\) is the underlying concentration at observation time \(t_k\).
- \(T_k\) is the temperature map applied during the interval.
- \(\Delta t_{\mathrm{control}}\) is the physical duration until the next observation.

Each observed process state is rendered into a synthetic image:

\[
I_k=R(c_k).
\]

Image processing reconstructs an estimated concentration:

\[
\hat c_k=P(I_k).
\]

The controller uses this image-derived estimate, the target, and current control information to select the next temperature adjustment.

Successive snapshots are related states of the same process. They are not independent images, repeated initial conditions, or unrelated inputs. Each observation and forecast carries an explicit simulated timestamp.

The underlying concentration remains available for validation, but normal controller operation receives its concentration input through the image-processing path.

## Process advancement and forecasting

The numerical solver serves two separate roles.

| Role | Responsibility |
|---|---|
| Ongoing process | Advances the current concentration chronologically under the selected control |
| Candidate forecasting | Predicts possible futures from the latest reconstructed observation |

These roles may share solver code but use independent state storage.

Within one control decision, every candidate begins from the same reconstructed state and timestamp. Across decisions, the starting snapshot changes as the ongoing process advances.

Candidate evaluation does not overwrite the current process or advance its simulated time.

The next observation comes from advancing the ongoing process by one control interval. It is not obtained by adopting the entire forecast endpoint as the new current state.

## Temperature-control approximation

Temperature is a directly adjustable field stored in a floating-point array named `temp`.

For a flattened rectangular grid:

```cpp
temp[y * width + x]
```

represents the temperature at location `(x, y)`.

Changing an array element changes the assigned temperature at that point.

The initial model assumes:

- Pointwise temperature control.
- Configurable minimum and maximum temperatures.
- Immediate temperature response on the diffusion timescale.
- A fixed temperature map between control updates.

The immediate-response approximation assumes that thermal equilibration is much faster than diffusion.

Heater-power calculations, heat-transfer PDEs, and physical coupling between neighboring heating locations are excluded from the initial scope. Optional spatial regularization can discourage rough controls without introducing heater physics.

## Mathematical model

The starting model is two-dimensional temperature-dependent diffusion:

\[
\frac{\partial c}{\partial t}
=
\nabla\cdot\left(D(T)\nabla c\right).
\]

It describes an effective thin-layer concentration rather than resolving the full depth of real glass.

A configurable diffusivity model is:

\[
D(T)
=
D_0\exp\left(-\frac{E_a}{k_BT}\right).
\]

Temperature is expressed in Kelvin, with consistent units for all parameters. Default material parameters are illustrative until supported by experimental calibration.

The numerical formulation emphasizes:

- Correct treatment of spatially varying diffusivity.
- Conservative face fluxes.
- No-flux boundaries for the initial closed-domain model.
- Mass conservation.
- Nonnegativity.
- Stability and convergence.

The divergence-form operator is not silently replaced by \(D(T)\nabla^2c\), which omits terms when diffusivity varies spatially. Concentration clamping is not used to conceal an unstable discretization.

One material species is modeled initially. Reactions, additional species, depth transport, and calibrated optical models can be introduced later.

## Numerical fields and resource ownership

The implementation separates the following logical data:

| Field | Purpose |
|---|---|
| `conc` | Underlying ongoing concentration |
| `temp` | Applied temperature distribution |
| `target` | Desired endpoint concentration |
| Synthetic image | Rendered observation |
| Reconstructed concentration | Image-derived estimate of the current state |
| Forecast buffers | Candidate future concentration states |
| Candidate controls | Proposed temperature maps or schedules |

Storage reuse is permitted when lifetimes allow it, without corrupting process state or candidate independence.

Configuration documents grid dimensions, domain dimensions, spacing, units, concentration interpretation, image precision, channels, and strides. Image and device buffers are not assumed to be tightly packed.

## Synthetic imaging and reconstruction

Image processing is a computational stage of the feedback loop, rather than simply a means of displaying numerical arrays.

Each observation involves:

1. Rendering concentration into an image.
2. Processing that image.
3. Reconstructing concentration.
4. Supplying the reconstruction to prediction and control.

The initial renderer uses a known concentration-to-image mapping and sufficient precision to avoid unnecessary quantization error.

Measurement encoding is distinguished from display coloring. A noninvertible display color map is not treated as a unique concentration measurement.

Supported processing can include:

- Channel extraction and conversion.
- Domain masking.
- Local filtering.
- Concentration reconstruction.
- Image gradients where they serve a defined purpose.

At least one meaningful neighborhood operation, such as configurable Gaussian filtering, supports comparison between direct global-memory access and tiled shared-memory processing with halos.

Filtering remains optional or bypassable. Its effect on reconstruction and control is measured rather than silently treating a smoothed image as exact concentration.

Noise, illumination variation, and optical blur can later provide more challenging synthetic observations.

## Initial conditions and target patterns

Concentration presets include single deposits, multiple deposits, lines, regions, and editable painted patterns.

Temperature presets include uniform maps, gradients, localized hot regions, multiple hot regions, and editable distributions.

Initial control benchmarks use reachable targets generated by a known simulation:

1. A chosen initial concentration evolves under a known temperature schedule.
2. Its concentration at a defined endpoint becomes the target.
3. The process resets to the initial condition.
4. The controller attempts to approach the target without access to the generating schedule.

Designed and imported targets are later supported.

Arbitrary pattern formation is not guaranteed. Diffusion generally smooths concentration and imposes reachability constraints. With no-flux boundaries, compatible total mass is necessary but does not establish reachability.

Residual mismatch and unsuccessful optimization are reported explicitly.

## Predictive feedback control

The repeated control cycle is:

1. Obtain the latest synthetic image.
2. Process it and reconstruct concentration.
3. Record the observation timestamp.
4. Generate or optimize candidate temperature controls.
5. Forecast their outcomes from the reconstructed state.
6. Compare predictions with the target.
7. Select a control.
8. Apply its immediate temperature map.
9. Advance the process for one control interval.
10. Produce the next image.

A candidate initially consists of a temperature map held fixed over the forecast. Future versions may optimize sequences of maps.

Every prediction identifies its assumed future control.

This creates a moving view into the future: as the process advances, new observations refresh the state from which predictions and decisions are made.

## Quantitative optimization

A basic target objective is mean squared concentration error:

\[
E_{\mathrm{target}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left(
c_i^{\mathrm{predicted}}
-
c_i^{\mathrm{target}}
\right)^2.
\]

Additional terms may penalize temperature changes, heating, or spatial roughness. Temperature bounds constrain the feasible controls.

An understandable baseline optimizer establishes whether automatic control improves outcomes over fixed-temperature alternatives.

Although the model accepts a temperature at every grid point, an initial optimizer may use coarse tiles or spatial basis coefficients expanded into `temp`. Any reduced parameterization is documented.

Optimization strategy accounts for forecast cost. Naïve finite differences requiring one complete forecast per pixel are unsuitable for large control fields.

Established optimization libraries are evaluated before introducing a general-purpose optimizer from scratch. Problem-specific GPU objectives, sensitivities, or adjoints may still require custom implementation.

Iteration budgets, convergence measures, stopping reasons, and failures are visible in diagnostics.

## Time definitions and endpoint behavior

The system distinguishes:

| Quantity | Meaning |
|---|---|
| Integration time step | Numerical advancement within the PDE solver |
| Control interval | Physical duration between applied control updates |
| Prediction horizon | Physical duration covered by a forecast |
| Endpoint time | Time at which the desired outcome is evaluated |
| Computation time | Wall-clock duration needed to calculate results |

A five-second forecast predicts five physical seconds ahead; it does not wait five seconds to produce an answer.

Prediction horizons are configurable. Example experiments include five seconds, thirty seconds, two minutes, and ten minutes. The appropriate horizon is selected through accuracy, control, and performance experiments.

Reachable-target benchmarks define an endpoint explicitly. A shorter rolling-horizon objective must explain its relationship to that endpoint rather than assume that greedy intermediate target matching is optimal.

Stopping conditions are explicit. Reaching a target does not physically halt diffusion, and freezing the state is not presented as a physical result.

## Low-latency requirements

Forecasting must run substantially faster than the process being predicted.

The real-time factor is:

\[
\mathrm{RTF}
=
\frac{\text{physical duration simulated}}
{\text{wall-clock computation duration}}.
\]

For illustration, predicting five physical seconds in fifty milliseconds corresponds to 100× faster than real time. This is an example, not an RTX 3080 performance promise.

A five-second prediction taking ten seconds is outdated for the intended live-control use.

Performance measurement covers the full decision cycle:

- Image generation or input handling.
- Image processing and reconstruction.
- Required transfers.
- Candidate forecasts.
- Objective evaluation.
- Optimization.
- Temperature-update preparation.

The complete cycle is evaluated against the control-update deadline. Visualization cost and its impact on control latency are measured separately.

Results include median, average, high-percentile, and maximum observed latency; deadline misses; forecast real-time factor; and the numerical accuracy associated with each configuration.

## Simulation clock and wall clock

A deterministic development mode may pause simulated time while computing the next control. This supports debugging but does not prove real-time capability.

Simulated and wall-clock time remain separate. Actual computation is compared with the intended physical update interval.

Performance is not improved artificially by slowing the virtual process. Real-time pacing and deadline testing can be added once the basic loop works.

## Production-quality software and library use

Standard functionality uses established libraries when they provide an appropriate, maintainable solution.

Potential dependencies include:

| Capability | Candidate tools |
|---|---|
| Build configuration | CMake and CMake presets |
| Image I/O and standard processing | OpenCV |
| GPU image primitives | OpenCV CUDA or NVIDIA NPP |
| Device reductions and scans | CUB or Thrust |
| Dense GPU linear algebra | cuBLAS |
| Sparse GPU operations | cuSPARSE |
| Fourier transforms | cuFFT |
| Optimization | A suitable established optimization library |
| Visualization | OpenGL, a window/context library, and Dear ImGui |
| Testing | An established C++ test framework |
| Logging and configuration | Appropriate established libraries |
| Diagnostics and profiling | Compute Sanitizer and NVIDIA profiling tools |

These are candidates rather than mandatory dependencies. Libraries are included when the selected algorithm needs them.

Dependency selection considers installed functionality, compatibility, licensing, maintenance, and performance. OpenCV CUDA support is verified rather than assumed.

Custom CUDA remains appropriate for the diffusion operator, specialized reconstruction, fused processing, integration requirements, and controlled performance experiments.

Custom implementations have a defined purpose and a correctness reference. Library implementations provide application components and benchmark baselines where suitable.

## Engineering practices

The implementation uses modern, CUDA-compatible C++ with:

- RAII resource ownership.
- Explicit device-buffer, stream, event, and graphics-resource lifetimes.
- Configuration validation.
- Clear error propagation.
- CUDA API and launch checks.
- Appropriate asynchronous completion checks.
- Documented units and defaults.
- Reproducible random seeds.
- Structured diagnostics.
- Separation of numerical, control, imaging, and UI code.
- Graceful initialization and runtime failure handling.

Repeated control cycles avoid unnecessary allocation and hidden synchronization.

Hidden CPU fallbacks are not allowed to invalidate GPU measurements. Global mutable state and unnecessary abstraction are minimized.

## Build and reproducibility

The project provides:

- Reproducible CMake builds and presets.
- Explicit dependency versions or constraints.
- A documented dependency-management strategy.
- Headless and GUI build options.
- Test and benchmark targets.
- Appropriate RTX 3080 architecture configuration.
- Compiler and CUDA compatibility requirements.
- Example configurations and commands.
- Machine-readable benchmark results where practical.

CPU validation supports development without substituting for CUDA testing.

Unavailable GPU tests are reported as skipped, not passed. Continuous integration validates what its environment supports and identifies checks requiring the local GPU.

## CUDA performance engineering

Performance investigation spans rendering, image processing, reconstruction, PDE integration, objective evaluation, candidate forecasting, optimization, and visualization.

Relevant concerns include:

| Area | Topics |
|---|---|
| Memory | Coalescing, bandwidth, alignment, strides, cache behavior, layouts, shared-memory reuse, halos, bank conflicts, padding, buffer reuse, spilling |
| Execution | Block shapes, warp mapping, boundaries, divergence, registers, occupancy, synchronization, launches, fusion, CUDA graphs |
| Transfers | GPU-resident state, transfer reduction, pinned memory, asynchronous copies |
| Concurrency | Streams, event dependencies, overlap, batched candidate evaluation |
| Algorithms | Solver choice, integration-step count, precision, reductions, optimization forecast count, CPU/GPU division |
| Visualization | CUDA–OpenGL interoperability and avoiding unnecessary blocking or CPU round trips |

Investigation is comprehensive without forcing irrelevant techniques into the implementation. Important opportunities are implemented and measured, or documented as irrelevant or deferred.

## Alternative implementations and benchmarks

Comparisons include meaningful alternatives such as:

- Global-memory versus shared-memory stencils.
- Tile and block dimensions.
- Cooperative halo-loading strategies.
- Padded versus unpadded shared-memory layouts where relevant.
- Interleaved versus planar image channels.
- AoS, SoA, or AoSoA for genuine multi-component data.
- Separate versus fused kernels.
- Conventional launches versus CUDA graphs.
- Numerical solver alternatives.
- Precision alternatives.
- Library primitives versus custom implementations.

No unnecessary combination matrix is required. Shared memory, padding, or a particular layout is not assumed to win universally.

Workloads cover small through large images, rectangular domains, non-tile-aligned dimensions, different filter radii, forecast horizons, candidate counts, and temperature patterns.

Comparisons use equivalent operations and consistent accuracy requirements. Layout conversions, transfers, and setup are included when they occur in normal execution.

Experimental variants remain selectable for benchmarking even when the main application adopts a different default.

## Profiling methodology

Optimization follows a measured cycle:

1. Establish a correct baseline.
2. Measure numerical accuracy and performance.
3. Profile representative workloads.
4. Identify a bottleneck.
5. Apply a targeted change.
6. Recheck correctness.
7. Measure kernel and full-pipeline effects.
8. Accept or reject the change based on evidence.

CUDA events measure device execution. Wall-clock measurements cover complete cycles.

Nsight Systems supports timeline and synchronization analysis; Nsight Compute supports kernel analysis; Compute Sanitizer supports relevant memory and synchronization checks.

Initialization and warm-up are separated from steady-state measurements.

Benchmarks record hardware, software versions, compiler settings, model parameters, dimensions, and tolerances. Results are measured rather than inferred or invented.

## Visualization and interaction

The interface displays:

- Current synthetic image.
- Reconstructed concentration.
- Underlying concentration for validation.
- Applied temperature.
- Desired target.
- Forecast concentration.
- Error maps.

Diagnostics show timestamps, horizon, endpoint, control interval, latency, real-time factor, target error, reconstruction error, conservation error, and controller status.

Interaction supports initial-condition selection, target generation, temperature editing, manual forecasting, automatic control, pause, step, reset, numerical settings, implementation variants, and controlled-versus-uncontrolled comparisons.

Views use labeled scales and consistent concentration ranges.

## Validation strategy

### Numerical validation

Checks cover uniform-state preservation, mass conservation, nonnegativity, analytical or trusted-reference agreement, CPU/GPU agreement, refinement convergence, and correct temperature dependence.

### Image validation

Checks cover dimensions, channels, strides, boundaries, halo loading, odd image sizes, tiled-versus-reference agreement, reconstruction error, and optional filtering effects.

### Control validation

Checks cover chronological observations, applied-control effects, forecast isolation, candidate independence, temperature bounds, improvement against baselines, and honest behavior on unreachable targets.

### Reliability validation

Meaningful tests cover invalid configurations, resource lifecycle, and practical failure handling. Tests support numerical and behavioral guarantees rather than simply duplicating implementation details.

## Architecture

The application separates configuration, units, field ownership, pattern generation, process advancement, synthetic rendering, image processing, reconstruction, observations, solvers, forecasting, objectives, optimization, visualization, and diagnostics.

Interfaces preserve the distinction between underlying process state and observed state. A future real-image source can replace synthetic observations through the same observation boundary.

The CPU solver serves numerical validation; it is not a required second physical-process model.

Architecture remains modular without excessive abstraction before a working baseline exists.

## Development roadmap

| Milestone | Deliverables |
|---|---|
| Numerical foundation | CMake project, generated fields, CPU reference, CUDA solver, correctness checks, viewable output, forecast benchmarks |
| Successive image observations | Chronological snapshots, synthetic rendering, image processing, reconstruction, neighborhood operation, reference comparisons |
| Interactive prediction | Current/target/temperature/forecast views, temperature editing, forecasts from reconstructed observations |
| Automatic control | Reachable targets, objective evaluation, optimization, repeated image-based control |
| Performance engineering | Full-pipeline profiling, implementation comparisons, bottleneck optimization, workload studies, deadline evaluation |

Production practices and performance measurement apply throughout development.

Initial implementation begins with repository and environment assessment, dependency selection, and the smallest runnable numerical milestone. Later stages build on verified results.

The completed system demonstrates a low-latency, image-guided predictive controller: it observes an evolving simulated diffusion process, reconstructs the present, predicts possible futures, and adjusts temperature toward a desired endpoint, supported by reproducible numerical validation and CUDA performance evidence.
