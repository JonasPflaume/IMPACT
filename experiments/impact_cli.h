#pragma once

/**
 * @file impact_cli.h
 * @brief Optional trailing CLI flags shared by the IMPACT experiment drivers.
 *
 * Every driver keeps its own task-specific positional arguments and its own tuned
 * defaults; this header only adds the knobs the accuracy A/B protocol needs, and
 * only ever overrides a config field when the corresponding flag is present. A
 * driver invoked the way it always was therefore behaves exactly as it always did.
 *
 * Two measurement traps live here rather than in the sweep scripts, because
 * getting them wrong silently produces numbers that look algorithmic but are not:
 *
 *  1. The reported stationarity is the same quantity the inner Gauss-Newton stops
 *     on. Leaving `newton_tol` at its default while asking for a 1e-8 certificate
 *     makes *both* inner solvers appear to floor near the default, for reasons
 *     that have nothing to do with either algorithm. So `--stat-tol` also tightens
 *     `newton_tol` and the inner stagnation tolerance, unless they are given
 *     explicitly.
 *  2. Classifying complementarity index sets needs an active-set threshold, and at
 *     these accuracy levels the threshold would decide the verdict by itself. The
 *     drivers therefore report the tolerance-free support residuals that
 *     AulaResult computes instead.
 */

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "impact/aula_config.h"

namespace impact_cli {

struct Options {
    double stat_tol = -1.0;    // > 0 turns the stationarity check on
    double tol = -1.0;         // outer feasibility tolerance (h / g / comp)
    double newton_tol = -1.0;  // GN gradient stop tolerance
    double inner_tol = -1.0;   // late-outer inner stagnation tolerance
    double rho_max = -1.0;     // penalty cap
    int max_outer = -1;
    int print_level = -1;
    // Force the classical normal-equations X-step. The saddle backend is the
    // default; being able to A/B the two linear-algebra backends is worth a flag.
    bool no_saddle = false;
};

inline void printUsage(const char* prog, const char* positional) {
    std::cerr << "Usage: " << prog << " " << positional << " [output_file] [flags]\n"
                 "  --stat-tol <v>        require ||grad L_A||_inf < v to converge\n"
                 "  --tol <v>             outer feasibility tolerance (h, g, comp)\n"
                 "  --newton-tol <v>      Gauss-Newton gradient tolerance\n"
                 "  --inner-tol <v>       late-outer inner stagnation tolerance\n"
                 "  --rho-max <v>         penalty cap\n"
                 "  --max-outer <n>       outer iteration budget\n"
                 "  --no-saddle           classical normal-equations X-step\n"
                 "  --quiet               suppress the per-outer trace\n";
}

/// Index of the first `--flag` at or after `start`, else argc. Drivers parse
/// their positional arguments against this instead of argc, so trailing flags can
/// be appended to any existing invocation without disturbing the positionals.
inline int firstFlagIndex(int argc, char* argv[], int start = 1) {
    for (int i = start; i < argc; ++i)
        if (argv[i][0] == '-' && argv[i][1] == '-') return i;
    return argc;
}

/// Parse flags from argv[start..argc). Returns false on an unknown flag or a
/// missing value; the caller should then print usage and exit.
inline bool parseFlags(int argc, char* argv[], int start, Options& opt) {
    for (int i = start; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](const char* name) -> double {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << name << " needs a value\n";
                std::exit(1);
            }
            return std::atof(argv[++i]);
        };
        if (a == "--no-saddle") opt.no_saddle = true;
        else if (a == "--quiet") opt.print_level = 0;
        else if (a == "--stat-tol") opt.stat_tol = value("--stat-tol");
        else if (a == "--tol") opt.tol = value("--tol");
        else if (a == "--newton-tol") opt.newton_tol = value("--newton-tol");
        else if (a == "--inner-tol") opt.inner_tol = value("--inner-tol");
        else if (a == "--rho-max") opt.rho_max = value("--rho-max");
        else if (a == "--max-outer") opt.max_outer = static_cast<int>(value("--max-outer"));
        else {
            std::cerr << "Error: unknown flag " << a << "\n";
            return false;
        }
    }
    return true;
}

/// Apply the parsed flags on top of a driver's tuned defaults.
inline void apply(const Options& opt, impact::AulaConfig& config) {
    if (opt.stat_tol > 0.0) {
        config.check_stationarity = true;
        config.stationarity_tol = opt.stat_tol;
        config.newton_tol = 0.1 * opt.stat_tol;                 // trap 1
        config.inner_tol_final = 1e-3 * opt.stat_tol;
        config.inner_tol_init = std::max(config.inner_tol_init, config.inner_tol_final);
    }
    if (opt.tol > 0.0) {
        config.outer_tol_h = opt.tol;
        config.outer_tol_g = opt.tol;
        config.outer_tol_comp = opt.tol;
    }
    if (opt.newton_tol > 0.0) config.newton_tol = opt.newton_tol;
    if (opt.inner_tol > 0.0) {
        config.inner_tol_final = opt.inner_tol;
        config.inner_tol_init = std::max(config.inner_tol_init, opt.inner_tol);
    }
    if (opt.rho_max > 0.0) config.rho_max = opt.rho_max;
    if (opt.max_outer > 0) config.max_outer_iters = opt.max_outer;
    if (opt.no_saddle) config.use_saddle = false;
    if (opt.print_level >= 0) config.print_level = opt.print_level;
}

inline std::string plannerTag(const Options&, const char* bcd_tag) { return bcd_tag; }

inline void printSettings(const Options&, const impact::AulaConfig& config) {
    std::cout << "Inner solver: BCD"
              << ", stat_tol=" << (config.check_stationarity ? config.stationarity_tol : 0.0)
              << ", tol=" << config.outer_tol_comp << ", newton_tol=" << config.newton_tol
              << ", inner_tol_final=" << config.inner_tol_final
              << ", max_outer=" << config.max_outer_iters
              << ", backend=" << (config.use_saddle ? "saddle" : "normal-equations")
              << std::endl;
}

/// The `RESULT key=value ...` line the sweep scripts parse. `Solution` is any
/// shooting solution struct carrying the fields below.
template <typename Solution>
inline void printResultLine(const std::string& mode, const Solution& s, double goal_err) {
    std::cout << std::scientific << std::setprecision(6) << "RESULT"
              << " mode=" << mode << " converged=" << (s.converged ? 1 : 0)
              << " objective=" << s.objective_value << " goal_err=" << goal_err
              << " dynamics=" << s.dynamics_violation
              << " comp_prod=" << s.complementarity_violation
              << " neg_G=" << s.comp_neg_G << " neg_H=" << s.comp_neg_H
              << " supp_G=" << s.comp_support_G << " supp_H=" << s.comp_support_H
              << " stationarity=" << s.stationarity_violation
              << " outer=" << s.outer_iterations << " inner=" << s.total_inner_iterations
              << " gn=" << s.total_gn_iterations << " time=" << s.solve_time
              << std::endl;
}

}  // namespace impact_cli
