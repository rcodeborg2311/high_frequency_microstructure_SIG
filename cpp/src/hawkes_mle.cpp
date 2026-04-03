/**
 * @file hawkes_mle.cpp
 * @brief Hawkes process MLE with O(N) recursive likelihood and Nelder-Mead.
 *
 * Reference: Bacry, Mastromatteo, Muzy (2015), "Hawkes Processes in Finance",
 * Market Microstructure and Liquidity 1(1):1550005.
 */

#include "hawkes_mle.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace hfm {

namespace {

/// Convert (log_mu, log_alpha, log_beta) → HawkesMLE::Params
HawkesMLE::Params from_log(const std::array<double, 3>& x) {
    return {std::exp(x[0]), std::exp(x[1]), std::exp(x[2])};
}

/// Convert HawkesMLE::Params → (log_mu, log_alpha, log_beta)
std::array<double, 3> to_log(const HawkesMLE::Params& p) {
    return {std::log(p.mu), std::log(p.alpha), std::log(p.beta)};
}

}  // namespace

// ── O(N) Recursive R computation ────────────────────────────────────────────

std::vector<double>
HawkesMLE::compute_R(const std::vector<double>& times,
                      double beta) const noexcept {
    const int N = static_cast<int>(times.size());
    std::vector<double> R(N, 0.0);

    // R_1 = 0 (no past events at the first arrival).
    // R_i = (R_{i-1} + 1) · exp(−β · (t_i − t_{i-1}))
    // Bacry et al. (2015), eq. 6, p. 8.
    for (int i = 1; i < N; ++i) {
        const double dt = times[i] - times[i - 1];
        R[i] = (R[i - 1] + 1.0) * std::exp(-beta * dt);
    }
    return R;
}

// ── Log-likelihood (O(N)) ────────────────────────────────────────────────────

double HawkesMLE::log_likelihood(const Params& p,
                                  const std::vector<double>& times,
                                  double T) const noexcept {
    if (times.empty() || p.mu <= 0.0 || p.alpha <= 0.0 || p.beta <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }

    const std::vector<double> R = compute_R(times, p.beta);
    const int N = static_cast<int>(times.size());

    // L(μ,α,β) = −μT − (α/β)·Σᵢ[1 − exp(−β(T−tᵢ))] + Σᵢ log(μ + α·Rᵢ)
    // Bacry et al. (2015), eq. 5, p. 8.
    double ll = -p.mu * T;

    // Compensator integral term: (α/β) · Σᵢ (1 − exp(−β(T−tᵢ)))
    double comp = 0.0;
    for (int i = 0; i < N; ++i) {
        comp += 1.0 - std::exp(-p.beta * (T - times[i]));
    }
    ll -= (p.alpha / p.beta) * comp;

    // Sum of log intensities
    for (int i = 0; i < N; ++i) {
        const double intensity = p.mu + p.alpha * R[i];
        if (intensity <= 0.0) return -std::numeric_limits<double>::infinity();
        ll += std::log(intensity);
    }

    return ll;
}

// ── Conditional intensity λ(t) ───────────────────────────────────────────────

double HawkesMLE::intensity_at(const Params& p,
                                const std::vector<double>& times,
                                double t) const noexcept {
    double contrib = 0.0;
    for (const double ti : times) {
        if (ti >= t) break;
        contrib += std::exp(-p.beta * (t - ti));
    }
    return p.mu + p.alpha * contrib;
}

// ── KS test on residual process ──────────────────────────────────────────────

double HawkesMLE::ks_test(const Params& p,
                           const std::vector<double>& times,
                           double T) const noexcept {
    // Compute compensated times Λ(t_i) = ∫_0^{t_i} λ(s) ds
    // For Hawkes: Λ(t_i) = μ·t_i + α·Σ_{j<i}[ (1 − exp(−β(t_i − t_j))) / β ]
    //                     = μ·t_i + (α/β)·Σ_{j≤i} (1 − exp(−β(t_i − t_j)))
    // We use the recursive approach for efficiency.
    const int N = static_cast<int>(times.size());
    if (N < 2) return 1.0;

    // Cumulative compensator at each event
    std::vector<double> lambda_t(N);
    const std::vector<double> R = compute_R(times, p.beta);

    // Integral of intensity up to t_i:
    // ∫_0^{t_i} λ(s)ds = μ·t_i + (α/β) · Σ_{j≤i} (1 − exp(−β(t_i−t_j)))
    // Using the recursive sum S_i = Σ_{j≤i} (1 − exp(−β(t_i−t_j))):
    // S_i = (S_{i-1} + 1) · (1 − exp(−β(t_i−t_{i-1}))) ... not trivial.
    // Use direct computation for the KS test (called rarely):
    for (int i = 0; i < N; ++i) {
        double integral = p.mu * times[i];
        for (int j = 0; j <= i; ++j) {
            integral += (p.alpha / p.beta) * (1.0 - std::exp(-p.beta * (times[i] - times[j])));
        }
        lambda_t[i] = integral;
    }
    (void)R;  // Only used for log-likelihood; not needed here.

    // Inter-compensator intervals should be Exp(1)
    std::vector<double> gaps(N - 1);
    for (int i = 0; i < N - 1; ++i) {
        gaps[i] = lambda_t[i + 1] - lambda_t[i];
    }
    std::sort(gaps.begin(), gaps.end());

    // KS statistic against Exp(1) CDF: F(x) = 1 − exp(−x)
    double ks = 0.0;
    const int M = static_cast<int>(gaps.size());
    for (int i = 0; i < M; ++i) {
        const double empirical = static_cast<double>(i + 1) / M;
        const double theoretical = 1.0 - std::exp(-gaps[i]);
        ks = std::max(ks, std::abs(empirical - theoretical));
    }
    return ks;
}

// ── Nelder-Mead simplex (no external dependencies) ───────────────────────────

std::pair<HawkesMLE::Params, bool>
HawkesMLE::nelder_mead(const std::function<double(const Params&)>& objective,
                        Params initial,
                        int    max_iter,
                        double tol) const {
    // Work in log-parameter space to enforce positivity:
    // x = [log(μ), log(α), log(β)]
    constexpr int N = 3;  // number of parameters

    // N-M coefficients (standard values)
    constexpr double ALPHA_NM = 1.0;   // reflection
    constexpr double GAMMA_NM = 2.0;   // expansion
    constexpr double RHO_NM   = 0.5;   // contraction
    constexpr double SIGMA_NM = 0.5;   // shrink

    // Objective in log space
    auto obj_log = [&](const std::array<double, N>& x) -> double {
        return objective(from_log(x));
    };

    // Initialize simplex: N+1 vertices
    std::array<std::array<double, N>, N + 1> simplex;
    simplex[0] = to_log(initial);

    for (int i = 0; i < N; ++i) {
        simplex[i + 1] = simplex[0];
        // Perturb each dimension by ±0.5 in log space
        simplex[i + 1][i] += (std::abs(simplex[0][i]) > 1e-10) ? 0.5 : 0.1;
    }

    // Evaluate at all vertices
    std::array<double, N + 1> fval;
    for (int i = 0; i <= N; ++i) {
        fval[i] = obj_log(simplex[i]);
    }

    bool converged = false;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Sort by function value
        std::array<int, N + 1> idx{0, 1, 2, 3};
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return fval[a] < fval[b]; });

        // Check convergence: diameter of simplex
        double diam = 0.0;
        for (int i = 1; i <= N; ++i) {
            for (int j = 0; j < N; ++j) {
                diam = std::max(diam, std::abs(simplex[idx[i]][j] - simplex[idx[0]][j]));
            }
        }
        if (diam < tol) {
            converged = true;
            break;
        }

        // Centroid of all but the worst
        std::array<double, N> centroid{};
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                centroid[j] += simplex[idx[i]][j] / N;
            }
        }

        const auto& worst = simplex[idx[N]];

        // Reflection
        std::array<double, N> xr;
        for (int j = 0; j < N; ++j) {
            xr[j] = centroid[j] + ALPHA_NM * (centroid[j] - worst[j]);
        }
        const double fr = obj_log(xr);

        if (fr < fval[idx[0]]) {
            // Expansion
            std::array<double, N> xe;
            for (int j = 0; j < N; ++j) {
                xe[j] = centroid[j] + GAMMA_NM * (xr[j] - centroid[j]);
            }
            const double fe = obj_log(xe);
            if (fe < fr) {
                simplex[idx[N]] = xe;
                fval[idx[N]]    = fe;
            } else {
                simplex[idx[N]] = xr;
                fval[idx[N]]    = fr;
            }
        } else if (fr < fval[idx[N - 1]]) {
            simplex[idx[N]] = xr;
            fval[idx[N]]    = fr;
        } else {
            // Contraction
            const bool outside = fr < fval[idx[N]];
            std::array<double, N> xc;
            if (outside) {
                for (int j = 0; j < N; ++j) {
                    xc[j] = centroid[j] + RHO_NM * (xr[j] - centroid[j]);
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    xc[j] = centroid[j] + RHO_NM * (worst[j] - centroid[j]);
                }
            }
            const double fc = obj_log(xc);

            if (fc < (outside ? fr : fval[idx[N]])) {
                simplex[idx[N]] = xc;
                fval[idx[N]]    = fc;
            } else {
                // Shrink
                for (int i = 1; i <= N; ++i) {
                    for (int j = 0; j < N; ++j) {
                        simplex[idx[i]][j] = simplex[idx[0]][j]
                            + SIGMA_NM * (simplex[idx[i]][j] - simplex[idx[0]][j]);
                    }
                    fval[idx[i]] = obj_log(simplex[idx[i]]);
                }
            }
        }
    }

    // Return best vertex
    int best = 0;
    for (int i = 1; i <= N; ++i) {
        if (fval[i] < fval[best]) best = i;
    }

    return {from_log(simplex[best]), converged};
}

// ── Public fit ───────────────────────────────────────────────────────────────

HawkesMLE::Result
HawkesMLE::fit(const std::vector<double>& event_times,
               double T,
               Params initial) const {
    if (event_times.size() < 3) {
        throw std::invalid_argument("HawkesMLE::fit requires at least 3 events");
    }

    // Negative log-likelihood with stationarity penalty
    const auto neg_ll = [&](const Params& p) -> double {
        if (p.mu <= 0.0 || p.alpha <= 0.0 || p.beta <= 0.0) {
            return 1e30;
        }
        if (p.branching_ratio() >= 0.999) {
            // Soft penalty pushing toward stationarity
            return 1e30 * p.branching_ratio();
        }
        const double ll = log_likelihood(p, event_times, T);
        if (!std::isfinite(ll)) return 1e30;
        return -ll;
    };

    auto [best_params, converged] = nelder_mead(neg_ll, initial);

    // Ensure stationarity
    if (!best_params.stationary()) {
        best_params.alpha = 0.99 * best_params.beta;
    }

    Result result;
    result.params         = best_params;
    result.log_likelihood = log_likelihood(best_params, event_times, T);
    result.n_events       = static_cast<int>(event_times.size());
    result.T              = T;
    result.converged      = converged;

    return result;
}

}  // namespace hfm
