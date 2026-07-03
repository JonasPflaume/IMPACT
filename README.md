# IMPACT

IMPACT is an augmented-Lagrangian / block-coordinate-descent solver for
**MPCCs** (Mathematical Programs with Complementarity Constraints). The code in
this repository is mainly organized around contact-implicit trajectory
optimization, but the solver can also be used on a generic MPCC.

<p align="center">
  <a href="https://jonaspflaume.github.io/impact_info/">
    <img src="https://img.shields.io/badge/Project%20Page-IMPACT-2563eb?style=for-the-badge&logo=githubpages&logoColor=white" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/2605.09127">
    <img src="https://img.shields.io/badge/arXiv-2605.09127-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="#citation">
    <img src="https://img.shields.io/badge/RSS-2026-1f883d?style=for-the-badge" alt="RSS 2026">
  </a>
</p>

<p align="center">
  <b><a href="https://jonaspflaume.github.io/impact_info/">IMPACT: An Implicit Active-Set Augmented Lagrangian for Fast Contact-Implicit Trajectory Optimization</a></b>
  <br>
  Jiayun Li, Dejian Gong, Georgia Chalvatzaki. RSS 2026. (<a href="#citation">cite below</a>)
</p>

<p align="center">
  <img src="resources/image35.gif" width="100%" alt="IMPACT contact-implicit trajectory optimization on the Push-T task">
</p>

The solver code is shared across all examples. A task supplies the symbolic
problem data, then a shooting builder assembles the MPCC passed to
`BCDAULASolver`. The CITO experiments in the paper use the multiple-shooting
front-end.

## Build

Dependencies: CMake ≥ 3.15, Eigen3, CasADi, BLAS/LAPACK. The Allegro demo also
needs MuJoCo and GLFW.

For the planar CITO experiments, the quickest path is the Docker image:

```bash
docker build -t impact .

# Runs the box-pushing multiple-shooting example with default arguments
docker run --rm -v "$PWD/results:/workspace/impact/results" impact

# Other CITO examples
docker run --rm -v "$PWD/results:/workspace/impact/results" impact push_t
docker run --rm -v "$PWD/results:/workspace/impact/results" impact cart
```

The Docker build disables the MuJoCo/Allegro target so the planar experiments do
not require MuJoCo or GLFW. Run `docker run --rm impact help` to see all wrapper
commands.

The container runs as root, so files written to the mounted `results/` directory
are root-owned on Linux hosts; add `--user "$(id -u):$(id -g)"` to the
`docker run` command if you want them owned by your user.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Smallest example
./build/experiments/toy_mpcc/toy_mpcc
```

To build the component tests, configure with `-DIMPACT_BUILD_TESTS=ON` and run
`ctest --test-dir build`.

## Layout

```
impact_solver/            solver library (target: impact)
  include/impact/
    stage_problem.h          common per-stage task interface
    mpcc_subproblem.h        buildMPCC(), the direct MPCC assembly entry point
    bcd_aula_solver.h        BCDAULASolver (outer AuLa + inner BCD)
    multiple_shooting.h      builder/front-end for multiple shooting
    single_shooting.h        builder/front-end for single shooting
    mpcc_stage.h, lcp_stage.h
    gauss_newton_solver.h, complementarity_projection.h, ...
experiments/              box / push_t / cart_transporter (multiple-shooting MPCC),
                          allegro (single-shooting LCP), toy_mpcc (generic MPCC)
penalty_solver/, relax_solver/   IPOPT / Scholtes baselines (same problem interface)
simulation/               MuJoCo wrapper (allegro)
resources/                allegro MuJoCo models
```

Pipeline: `StageProblem` → `build{Single,Multiple}Shooting` → `buildMPCC` →
`AulaSubproblem` → `BCDAULASolver`. Trajectory problems normally go through one
of the shooting builders. A non-trajectory MPCC can call `buildMPCC` directly.

## Experiments

**Planar tasks.** Pass the initial state followed by the goal state. By default,
the trajectory is written under `results/<task>/bcd_aula/`; most binaries also
accept an explicit output path as the final argument.

For the CITO experiments in the paper, use the multiple-shooting binaries:
`*_impact_multiple`. In this transcription, the state trajectory is free and
dynamics are enforced as defect constraints. Single shooting is also implemented
for comparison purposes.

```bash
# box pushing       state = [px, py, theta]
./build/experiments/box/box_impact_multiple                            0 0 0   0.1 0.1 1.0
# push-T            state = [px, py, theta]
./build/experiments/push_t/push_t_impact_multiple                     0 0 0   0.05 0.05 1.5708
# cart transporter  state = [x1, x2, x1dot, x2dot]   (horizon 300)
./build/experiments/cart_transporter/cart_transporter_impact_multiple  0 0 0 0   1 0 0 0
```

The `*_penalty` (IPOPT) and `*_relaxation` (Scholtes) binaries are baselines and
take the same arguments. The Python visualizers live next to each task:
`experiments/<task>/<task>_visual.py`.

**Allegro hand re-orientation.** This is the MuJoCo receding-horizon MPC example
and requires MuJoCo + GLFW:

```bash
./build/experiments/allegro/allegro_impact_single --object cube --seed 0 --render
./build/experiments/allegro/allegro_impact_single --object cube --seed 0 --json   # metrics, headless
```

`--object` ∈ {airplane, binoculars, bowl, bunny, camera, can, cube, cup, elephant,
flashlight, foambrick, light_bulb, mug, piggy_bank, rubber_duck, stick, teapot,
torus, water_bottle}. Other flags: `--seed <n>`, `--save-video <path>` (with
`--render`), `--horizon <h>`, `--max-inner-iters <n>`, `--no-saddle`.

## Solve an MPCC

Build the residuals symbolically with CasADi and pass them to `buildMPCC`.
The example below solves
`min (x1-0.5)^2 + (x2-0.5)^2  s.t.  0 ≤ x1 ⊥ x2 ≥ 0`.

```cpp
#include "impact/mpcc_subproblem.h"
#include "impact/bcd_aula_solver.h"
using casadi::SX;

SX z = SX::sym("z", 2);
impact::MPCCDescription d;
d.z = z;
d.cost = z - 0.5;              // objective = ||cost||^2
d.cost_is_linear = true;      // residual affine in z (quadratic objective)
// 0 <= z1 ⊥ z2 >= 0   (also: .Equality with .c=h(z),  .Inequality with .c=g(z))
d.constraints.push_back({impact::MPCCConstraint::Complementarity, "comp",
                         /*c=*/SX(), /*G=*/z(0), /*H=*/z(1),
                         /*scale=*/1.0, /*rho_init=*/1.0, /*tol=*/1e-8});

impact::MPCCSubproblem mpcc = impact::buildMPCC(d);
impact::BCDAULAConfig cfg;            // tweak rho_scale, tolerances, use_saddle, ...
Eigen::VectorXd z0(2); z0 << 0.6, 0.4;
impact::BCDAULAResult r = impact::BCDAULASolver().solve(*mpcc.sub, cfg, z0);
// r.z, r.objective_value, r.converged, r.complementarity_violation
```

`p` (runtime parameters) lets the same symbolic problem be re-solved with new data
(for example, contact Jacobians at each MPC step). Write the parameter vector at
`mpcc.off_p` before solving. See `experiments/toy_mpcc/toy_mpcc.cpp` for a full
runnable version, and the shooting front-ends (`MultipleShootingSolver` /
`SingleShootingSolver`) for trajectory tasks.

## License

This code is released under the MIT License. See `LICENSE`.

## Citation

```bibtex
@inproceedings{li2026impact,
  title        = {{IMPACT}: An Implicit Active-Set Augmented Lagrangian for
                  Fast Contact-Implicit Trajectory Optimization},
  author       = {Li, Jiayun and Gong, Dejian and Chalvatzaki, Georgia},
  booktitle    = {Proceedings of Robotics: Science and Systems ({RSS})},
  year         = {2026},
  note         = {To appear},
  eprint       = {2605.09127},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  doi          = {10.48550/arXiv.2605.09127}
}
```
