#pragma once

#include <Eigen/Core>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "impact/aula_subproblem.h"
#include "impact/saddle_layout.h"

namespace impact {

/**
 * @file subproblem_spec.h
 * @brief Language-neutral description of an assembled AuLa subproblem.
 *
 * `buildMPCC()` does two separable things: it *derives* the augmented-Lagrangian
 * residual rows symbolically, and it *wires* the resulting CasADi functions plus
 * their block metadata into an AulaSubproblem. Only the first half needs CasADi's
 * symbolic layer; the second half is bookkeeping over offsets and dimensions.
 *
 * This header exposes the second half on its own. A front-end that has already
 * done the symbolic derivation elsewhere -- the Python modelling layer in
 * `impact/` does it with the CasADi Python API -- hands over the derived
 * functions in CasADi's own serialized form together with the offsets it chose,
 * and gets back the same AulaSubproblem the C++ builder would have produced.
 *
 * The functions travel as serialized strings (`casadi::Function::serialize()`)
 * rather than as `casadi::Function` objects so the boundary stays a plain byte
 * interface: nothing but CasADi's own format crosses it, and the binding layer
 * needs no knowledge of CasADi's C++ types.
 *
 * Nothing here re-derives anything. Row order, parameter offsets and saddle
 * layout are taken verbatim from the spec, so a spec that mirrors what
 * buildMPCC() emits produces a bit-identical subproblem; `parity_python.cpp`
 * pins that.
 */

/// One equality / inequality AuLa channel, as laid out by the front-end.
struct DualBlockSpec {
    std::string name;
    bool inequality = false;  // false: equality channel
    int dim = 0;
    double scale = 1.0;
    double rho_init = 1.0;
    double tol = 1e-5;
    int kappa_offset = 0;
    int rho_offset = 0;
    /// Serialized (z, p) -> scale * c(z, p). Used for the dual ascent and for the
    /// reported violation, so it carries `scale` already folded in.
    std::string eval_scaled;
};

/// One complementarity channel 0 <= G ⊥ H >= 0.
struct CompBlockSpec {
    std::string name;
    int dim = 0;
    double scale = 1.0;
    double rho_init = 1.0;
    double tol = 1e-5;
    int sG_offset = 0;
    int sH_offset = 0;
    int kappaG_offset = 0;
    int kappaH_offset = 0;
    int rho_offset = 0;
};

/**
 * @brief Everything needed to reconstitute an AulaSubproblem.
 *
 * `residual`, `jacobian` and `gh` are required; the rest are optional and an
 * empty string means "not emitted", matching the C++ builder's behaviour when the
 * corresponding flag is off.
 */
struct SubproblemSpec {
    int n_opt = 0;
    int n_params = 0;

    // Serialized CasADi functions, all with signature (z, p) -> ...
    std::string residual;      // -> r
    std::string jacobian;      // -> dr/dz (sparse)
    std::string gh;            // -> {G_all, H_all}
    std::string obj;           // -> ||cost||^2   (optional)
    std::string obj_grad;      // -> grad ||cost||^2 (optional; stagnation restarts)
    std::string stationarity;  // -> grad ||r||^2    (optional; stationarity check)

    SaddleLayout layout;
    std::vector<DualBlockSpec> dual_blocks;
    std::vector<CompBlockSpec> comp_blocks;

    // Termination options, mirroring MPCCDescription.
    bool check_stationarity = false;
    bool conditioned_complementarity = true;
    double stationarity_tol = 1e-5;
    int max_stagnation_restarts = 0;

    /// Constant entries of the parameter buffer written once at build time (the
    /// `rho_one` slot the nonlinear-cost saddle block reads, and any task data
    /// the front-end already knows). Applied in order.
    std::vector<std::pair<int, std::vector<double>>> param_values;
};

/**
 * @brief Reconstitute an AulaSubproblem from a spec.
 *
 * Throws std::invalid_argument if a required function is missing or a declared
 * offset would fall outside the parameter buffer.
 */
std::unique_ptr<AulaSubproblem> buildFromSpec(const SubproblemSpec& spec);

}  // namespace impact
