/**
 * @file bindings.cpp
 * @brief Python bindings for the IMPACT augmented-Lagrangian solver.
 *
 * The module exposes the *solver*, not the modelling layer. Everything symbolic
 * -- deriving the augmented-Lagrangian rows, their Jacobian, the complementarity
 * legs -- happens on the Python side with the CasADi Python API and arrives here
 * as serialized CasADi functions inside a `SubproblemSpec` (see
 * impact/subproblem_spec.h). What crosses the boundary is therefore CasADi's own
 * byte format plus a handful of integers, and no CasADi type appears in any
 * signature below.
 *
 * `Solver` is a class rather than a free function so a receding-horizon caller
 * can keep one instance across MPC steps: the Jacobian sparsity maps and the
 * saddle symbolic factorisation live in it and are reused whenever the structure
 * is unchanged.
 */

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// The only CasADi headers here, and neither declares a type: `config.h` for the
// version this module is compiled against, `casadi_meta.hpp` for the version of
// the libcasadi it ends up linked to. Both are strings the import guard reads.
#include <casadi/config.h>
#include <casadi/core/casadi_meta.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "impact/aula_config.h"
#include "impact/aula_solver.h"
#include "impact/aula_subproblem.h"
#include "impact/complementarity_projection.h"
#include "impact/saddle_layout.h"
#include "impact/subproblem_spec.h"

namespace py = pybind11;
using namespace impact;

namespace {

/// Reset every AuLa channel to its build-time seed. The single-shooting front-end
/// does this between MPC steps so a solve never inherits the previous step's
/// multipliers or its escalated penalties; exposing it keeps that behaviour
/// available to a Python driver that reuses one subproblem.
void resetAulaState(AulaSubproblem& sub) {
    for (DualBlock& b : sub.dualBlocks()) {
        b.kappa.setZero();
        b.rho = b.rho_init;
    }
    for (CompBlock& c : sub.compBlocks()) {
        c.kappaG.setZero();
        c.kappaH.setZero();
        c.sG.setZero();
        c.sH.setZero();
        c.rho = c.rho_init;
    }
    sub.syncParams();
}

}  // namespace

PYBIND11_MODULE(_impact_core, m) {
    m.doc() = "IMPACT solver core: augmented-Lagrangian MPCC solver with a "
              "block-coordinate-descent inner loop.";

    // ---------------------------------------------------------------- config --
    py::class_<AulaConfig>(m, "AulaConfig",
                           "Solver hyper-parameters. Field-for-field the C++ AulaConfig; the "
                           "defaults here are the C++ defaults.")
        .def(py::init<>())
        .def_readwrite("horizon", &AulaConfig::horizon)
        .def_readwrite("x_0", &AulaConfig::x_0)
        .def_readwrite("x_goal", &AulaConfig::x_goal)
        .def_readwrite("stage_cost_weight", &AulaConfig::stage_cost_weight)
        .def_readwrite("stage_state_cost_weight", &AulaConfig::stage_state_cost_weight)
        .def_readwrite("control_rate_weight", &AulaConfig::control_rate_weight)
        .def_readwrite("final_cost_weight", &AulaConfig::final_cost_weight)
        .def_readwrite("rho_fix_point_init", &AulaConfig::rho_fix_point_init)
        .def_readwrite("rho_dynamics_init", &AulaConfig::rho_dynamics_init)
        .def_readwrite("rho_eq_init", &AulaConfig::rho_eq_init)
        .def_readwrite("rho_ineq_init", &AulaConfig::rho_ineq_init)
        .def_readwrite("rho_comp_init", &AulaConfig::rho_comp_init)
        .def_readwrite("rho_max", &AulaConfig::rho_max)
        .def_readwrite("rho_scale", &AulaConfig::rho_scale)
        .def_readwrite("penalty_decrease_ratio", &AulaConfig::penalty_decrease_ratio)
        .def_readwrite("auto_rho_init", &AulaConfig::auto_rho_init)
        .def_readwrite("auto_rho_clip_min", &AulaConfig::auto_rho_clip_min)
        .def_readwrite("auto_rho_clip_max", &AulaConfig::auto_rho_clip_max)
        .def_readwrite("safeguard_factor", &AulaConfig::safeguard_factor)
        .def_readwrite("fix_point_scale", &AulaConfig::fix_point_scale)
        .def_readwrite("dynamics_scale", &AulaConfig::dynamics_scale)
        .def_readwrite("eq_scale", &AulaConfig::eq_scale)
        .def_readwrite("ineq_scale", &AulaConfig::ineq_scale)
        .def_readwrite("comp_scale", &AulaConfig::comp_scale)
        .def_readwrite("max_outer_iters", &AulaConfig::max_outer_iters)
        .def_readwrite("outer_tol_h", &AulaConfig::outer_tol_h)
        .def_readwrite("outer_tol_g", &AulaConfig::outer_tol_g)
        .def_readwrite("outer_tol_comp", &AulaConfig::outer_tol_comp)
        .def_readwrite("max_inner_iters", &AulaConfig::max_inner_iters)
        .def_readwrite("inner_tol_init", &AulaConfig::inner_tol_init)
        .def_readwrite("inner_tol_final", &AulaConfig::inner_tol_final)
        .def_readwrite("inner_tol_ramp_start", &AulaConfig::inner_tol_ramp_start)
        .def_readwrite("inner_tol_ramp_end", &AulaConfig::inner_tol_ramp_end)
        .def_readwrite("check_stationarity", &AulaConfig::check_stationarity)
        .def_readwrite("conditioned_complementarity", &AulaConfig::conditioned_complementarity)
        .def_readwrite("stationarity_tol", &AulaConfig::stationarity_tol)
        .def_readwrite("max_stagnation_restarts", &AulaConfig::max_stagnation_restarts)
        .def_readwrite("use_forcing_sequence", &AulaConfig::use_forcing_sequence)
        .def_readwrite("forcing_cap", &AulaConfig::forcing_cap)
        .def_readwrite("forcing_factor", &AulaConfig::forcing_factor)
        .def_readwrite("newton_max_iter", &AulaConfig::newton_max_iter)
        .def_readwrite("newton_tol", &AulaConfig::newton_tol)
        .def_readwrite("newton_step_tol", &AulaConfig::newton_step_tol)
        .def_readwrite("newton_regularization", &AulaConfig::newton_regularization)
        .def_readwrite("newton_lambda_min", &AulaConfig::newton_lambda_min)
        .def_readwrite("newton_lambda_max", &AulaConfig::newton_lambda_max)
        .def_readwrite("newton_max_damping_tries", &AulaConfig::newton_max_damping_tries)
        .def_readwrite("jit", &AulaConfig::jit)
        .def_readwrite("use_saddle", &AulaConfig::use_saddle)
        .def_readwrite("saddle_sigma_primal", &AulaConfig::saddle_sigma_primal)
        .def_readwrite("saddle_equilibrate_dual", &AulaConfig::saddle_equilibrate_dual)
        .def_readwrite("saddle_refinement_steps", &AulaConfig::saddle_refinement_steps)
        .def_readwrite("use_cmd_bounds", &AulaConfig::use_cmd_bounds)
        .def_readwrite("cmd_lower", &AulaConfig::cmd_lower)
        .def_readwrite("cmd_upper", &AulaConfig::cmd_upper)
        .def_readwrite("use_constant_state_init", &AulaConfig::use_constant_state_init)
        .def_readwrite("print_level", &AulaConfig::print_level);

    // ---------------------------------------------------------------- result --
    py::enum_<BCDAULAStatus>(m, "Status")
        .value("Converged", BCDAULAStatus::Converged)
        .value("MaxIterations", BCDAULAStatus::MaxIterations)
        .value("LinearAlgebraFailure", BCDAULAStatus::LinearAlgebraFailure);

    py::class_<ConstraintViolation>(m, "ConstraintViolation")
        .def_readonly("name", &ConstraintViolation::name)
        .def_readonly("violation", &ConstraintViolation::violation)
        .def_readonly("residual_violation", &ConstraintViolation::residual_violation)
        .def("__repr__", [](const ConstraintViolation& c) {
            return "<ConstraintViolation " + c.name + "=" + std::to_string(c.violation) + ">";
        });

    py::class_<AulaResult>(m, "AulaResult")
        .def_readonly("z", &AulaResult::z)
        .def_readonly("objective_value", &AulaResult::objective_value)
        .def_readonly("dynamics_violation", &AulaResult::dynamics_violation)
        .def_readonly("equality_violation", &AulaResult::equality_violation)
        .def_readonly("inequality_violation", &AulaResult::inequality_violation)
        .def_readonly("complementarity_violation", &AulaResult::complementarity_violation)
        .def_readonly("stationarity_violation", &AulaResult::stationarity_violation)
        .def_readonly("constraint_violations", &AulaResult::constraint_violations)
        .def_readonly("comp_neg_G", &AulaResult::comp_neg_G)
        .def_readonly("comp_neg_H", &AulaResult::comp_neg_H)
        .def_readonly("comp_support_G", &AulaResult::comp_support_G)
        .def_readonly("comp_support_H", &AulaResult::comp_support_H)
        .def_readonly("converged", &AulaResult::converged)
        .def_readonly("status", &AulaResult::status)
        .def_readonly("outer_iterations", &AulaResult::outer_iterations)
        .def_readonly("total_inner_iterations", &AulaResult::total_inner_iterations)
        .def_readonly("total_gn_iterations", &AulaResult::total_gn_iterations)
        .def_readonly("solve_time", &AulaResult::solve_time)
        .def_readonly("eval_time", &AulaResult::eval_time)
        .def_readonly("factor_time", &AulaResult::factor_time)
        .def_readonly("status_message", &AulaResult::status_message);

    // ------------------------------------------------------------ saddle/spec --
    py::class_<SaddleBlock>(m, "SaddleBlock")
        .def(py::init<>())
        .def(py::init([](int row_start, int count, int rho_param_offset) {
                 return SaddleBlock{row_start, count, rho_param_offset};
             }),
             py::arg("row_start"), py::arg("count"), py::arg("rho_param_offset"))
        .def_readwrite("row_start", &SaddleBlock::row_start)
        .def_readwrite("count", &SaddleBlock::count)
        .def_readwrite("rho_param_offset", &SaddleBlock::rho_param_offset);

    py::class_<SaddleLayout>(m, "SaddleLayout")
        .def(py::init<>())
        .def_readwrite("n_z", &SaddleLayout::n_z)
        .def_readwrite("n_cost", &SaddleLayout::n_cost)
        .def_readwrite("n_dual", &SaddleLayout::n_dual)
        .def_readwrite("blocks", &SaddleLayout::blocks);

    py::class_<DualBlockSpec>(m, "DualBlockSpec")
        .def(py::init<>())
        .def_readwrite("name", &DualBlockSpec::name)
        .def_readwrite("inequality", &DualBlockSpec::inequality)
        .def_readwrite("dim", &DualBlockSpec::dim)
        .def_readwrite("scale", &DualBlockSpec::scale)
        .def_readwrite("rho_init", &DualBlockSpec::rho_init)
        .def_readwrite("tol", &DualBlockSpec::tol)
        .def_readwrite("kappa_offset", &DualBlockSpec::kappa_offset)
        .def_readwrite("rho_offset", &DualBlockSpec::rho_offset)
        .def_readwrite("eval_scaled", &DualBlockSpec::eval_scaled);

    py::class_<CompBlockSpec>(m, "CompBlockSpec")
        .def(py::init<>())
        .def_readwrite("name", &CompBlockSpec::name)
        .def_readwrite("dim", &CompBlockSpec::dim)
        .def_readwrite("scale", &CompBlockSpec::scale)
        .def_readwrite("rho_init", &CompBlockSpec::rho_init)
        .def_readwrite("tol", &CompBlockSpec::tol)
        .def_readwrite("sG_offset", &CompBlockSpec::sG_offset)
        .def_readwrite("sH_offset", &CompBlockSpec::sH_offset)
        .def_readwrite("kappaG_offset", &CompBlockSpec::kappaG_offset)
        .def_readwrite("kappaH_offset", &CompBlockSpec::kappaH_offset)
        .def_readwrite("rho_offset", &CompBlockSpec::rho_offset);

    py::class_<SubproblemSpec>(m, "SubproblemSpec")
        .def(py::init<>())
        .def_readwrite("n_opt", &SubproblemSpec::n_opt)
        .def_readwrite("n_params", &SubproblemSpec::n_params)
        .def_readwrite("residual", &SubproblemSpec::residual)
        .def_readwrite("jacobian", &SubproblemSpec::jacobian)
        .def_readwrite("gh", &SubproblemSpec::gh)
        .def_readwrite("obj", &SubproblemSpec::obj)
        .def_readwrite("obj_grad", &SubproblemSpec::obj_grad)
        .def_readwrite("stationarity", &SubproblemSpec::stationarity)
        .def_readwrite("layout", &SubproblemSpec::layout)
        .def_readwrite("dual_blocks", &SubproblemSpec::dual_blocks)
        .def_readwrite("comp_blocks", &SubproblemSpec::comp_blocks)
        .def_readwrite("check_stationarity", &SubproblemSpec::check_stationarity)
        .def_readwrite("conditioned_complementarity", &SubproblemSpec::conditioned_complementarity)
        .def_readwrite("stationarity_tol", &SubproblemSpec::stationarity_tol)
        .def_readwrite("max_stagnation_restarts", &SubproblemSpec::max_stagnation_restarts)
        .def_readwrite("param_values", &SubproblemSpec::param_values);

    // ------------------------------------------------------------ subproblem --
    py::class_<AulaSubproblem>(m, "Subproblem",
                               "Assembled augmented-Lagrangian subproblem. Build one with "
                               "build_subproblem(spec); the AuLa dual state lives here and is "
                               "mutated by Solver.solve().")
        .def_property_readonly("num_opt", &AulaSubproblem::numOpt)
        .def("set_param_value", &AulaSubproblem::setParamValue, py::arg("offset"),
             py::arg("values"),
             "Write a fixed (non-AuLa) parameter sub-vector, e.g. a goal or per-step contact "
             "data.")
        .def("sync_params", &AulaSubproblem::syncParams,
             "Scatter the current dual/complementarity state into the parameter buffer.")
        .def("reset_aula_state", &resetAulaState,
             "Zero every multiplier and slack and restore each channel's build-time penalty.")
        .def("eval_gh",
             [](const AulaSubproblem& s, const Eigen::VectorXd& z) {
                 Eigen::VectorXd G, H;
                 s.evalGH(z, G, H);
                 return py::make_tuple(G, H);
             },
             py::arg("z"), "Evaluate the stacked complementarity legs (G, H) at z.")
        .def("eval_augmented_objective", &AulaSubproblem::evalAugmentedObjective, py::arg("z"))
        .def("eval_task_objective", &AulaSubproblem::evalTaskObjective, py::arg("z"))
        .def_property_readonly(
            "params",
            [](const AulaSubproblem& s) {
                const casadi::DM& p = s.params();
                return Eigen::VectorXd(
                    Eigen::Map<const Eigen::VectorXd>(p.ptr(), p.numel()));
            },
            "Current parameter buffer (copy).")
        .def_property_readonly("dual_block_names",
                               [](const AulaSubproblem& s) {
                                   std::vector<std::string> names;
                                   for (const DualBlock& b : s.dualBlocks())
                                       names.push_back(b.name);
                                   return names;
                               })
        .def_property_readonly("rho_values", [](const AulaSubproblem& s) {
            std::vector<std::pair<std::string, double>> out;
            for (const DualBlock& b : s.dualBlocks()) out.emplace_back(b.name, b.rho);
            for (const CompBlock& c : s.compBlocks()) out.emplace_back(c.name, c.rho);
            return out;
        });

    m.def("build_subproblem", &buildFromSpec, py::arg("spec"),
          "Reconstitute a Subproblem from serialized CasADi functions plus block metadata.");

    // ---------------------------------------------------------------- solver --
    py::class_<AulaSolver>(m, "Solver",
                           "Safeguarded augmented-Lagrangian solver. Reuse one instance across "
                           "MPC steps to keep the sparsity maps and symbolic factorisation.")
        .def(py::init<>())
        .def("solve",
             [](AulaSolver& self, AulaSubproblem& sub, const AulaConfig& config,
                const Eigen::VectorXd& z_init) {
                 // The solve is long-running and touches no Python object, so
                 // drop the GIL.
                 py::gil_scoped_release release;
                 return self.solve(sub, config, z_init);
             },
             py::arg("subproblem"), py::arg("config"), py::arg("z_init"));

    // ------------------------------------------------- complementarity kernels --
    // Exposed so the closed-form slack update is not re-derived in Python: two
    // spellings of the same formula pick different branches on exact ties (see
    // complementarity_projection.h).
    m.def("project_complementarity",
          [](const Eigen::VectorXd& G, const Eigen::VectorXd& H, const Eigen::VectorXd& kappaG,
             const Eigen::VectorXd& kappaH, double rho, double scale) {
              Eigen::VectorXd sG, sH;
              projectComplementarity(G, H, kappaG, kappaH, rho, scale, sG, sH);
              return py::make_tuple(sG, sH);
          },
          py::arg("G"), py::arg("H"), py::arg("kappaG"), py::arg("kappaH"), py::arg("rho"),
          py::arg("scale"));

#ifdef IMPACT_VERSION
    m.attr("__version__") = IMPACT_VERSION;
#else
    m.attr("__version__") = "0.0.0";
#endif

    // Which CasADi this extension is really talking to. The two are not the same
    // question: `casadi_build_version` is the header it was compiled against,
    // baked in at build time, while `casadi_runtime_version` is answered by the
    // libcasadi the dynamic loader actually bound it to -- which is whichever
    // copy reached the process first, because every CasADi 3.x exports the same
    // SONAME. Python compares both against the `casadi` module it imported (see
    // the preamble in impact/__init__.py): the problem crosses this boundary as
    // CasADi's own serialized functions, so a disagreement here is a solve that
    // dies inside `Function::deserialize` -- or worse, does not.
    m.attr("casadi_build_version") = CASADI_VERSION_STRING;
    m.attr("casadi_runtime_version") = casadi::CasadiMeta::version();
}
