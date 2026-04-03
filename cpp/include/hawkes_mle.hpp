#pragma once
/**
 * @file hawkes_mle.hpp
 * @brief Hawkes process MLE with O(N) recursive log-likelihood computation.
 *
 * Models order-arrival clustering via a self-exciting point process.
 *
 * References:
 *   Bacry, Mastromatteo, Muzy (2015), "Hawkes Processes in Finance",
 *   Market Microstructure and Liquidity 1(1):1550005.
 *   (Intensity definition: eq. 1, p. 4; MLE: eq. 5, p. 8.)
 *
 * Intensity:
 *   λ(t) = μ + α · Σ_{t_i < t} exp(−β(t − t_i))
 *
 * Log-likelihood (eq. 5, Bacry et al. 2015):
 *   L(μ,α,β) = −μT − (α/β)·Σᵢ[1 − exp(−β(T−tᵢ))] + Σᵢ log(μ + α·Rᵢ)
 *
 * Recursive O(N) computation of Rᵢ (eq. 6, Bacry et al. 2015):
 *   R₁ = 0
 *   Rᵢ = (Rᵢ₋₁ + 1) · exp(−β · (tᵢ − tᵢ₋₁))
 *
 * Optimization: Nelder-Mead simplex on log-transformed parameters
 * (enforces μ > 0, α > 0, β > 0 without explicit box constraints).
 */

#include <functional>
#include <vector>

namespace hfm {

/**
 * @brief Maximum-likelihood estimator for a univariate Hawkes process.
 *
 * All public methods are const (no mutable state); objects can be reused
 * across instruments / sessions.
 */
class HawkesMLE {
public:
    // ── Parameter / result types ──────────────────────────────────────────────

    /// Hawkes process parameters (μ, α, β).
    struct Params {
        double mu    = 0.5;  ///< Baseline arrival intensity (events/sec).
        double alpha = 0.3;  ///< Excitation amplitude.
        double beta  = 1.0;  ///< Decay rate (1/sec).

        /**
         * @brief Branching ratio n = α/β (Bacry et al. 2015, eq. 2, p. 5).
         *
         * Fraction of arrivals attributable to self-excitation.
         * Must be < 1 for the process to be stationary.
         */
        [[nodiscard]] double branching_ratio() const noexcept {
            return (beta > 0.0) ? alpha / beta : 1e30;
        }

        /// Returns true when the process is stationary (α/β < 1).
        [[nodiscard]] bool stationary() const noexcept {
            return branching_ratio() < 1.0;
        }

        /// log(2)/β — the half-life of excitation in seconds.
        [[nodiscard]] double excitation_halflife() const noexcept {
            return (beta > 0.0) ? 0.693147180559945 / beta : 1e30;
        }
    };

    /// Full fit result including diagnostics.
    struct Result {
        Params params;

        double log_likelihood      = 0.0;   ///< Maximized log-likelihood.
        int    n_events            = 0;     ///< Number of events used.
        double T                   = 0.0;   ///< Observation window length (sec).
        bool   converged           = false; ///< Did Nelder-Mead converge?

        /// μ — exogenous (baseline) arrival rate.
        [[nodiscard]] double baseline_rate() const noexcept { return params.mu; }

        /// α/β — fraction of arrivals that are self-excited.
        [[nodiscard]] double endogenous_fraction() const noexcept {
            return params.branching_ratio();
        }
    };

    // ── Core API ──────────────────────────────────────────────────────────────

    /**
     * @brief Fit Hawkes model to event times via MLE.
     *
     * Uses a Nelder-Mead simplex on log-transformed parameters to ensure
     * positivity without box constraints. Stationarity (α/β < 1) is enforced
     * via a large penalty in the objective.
     *
     * @param event_times  Sorted vector of event times (seconds from start).
     *                     Must have at least 3 events.
     * @param T            Total observation window [0, T] in seconds.
     * @param initial      Initial parameter guess (optional).
     * @return             Fit result including parameters and diagnostics.
     */
    [[nodiscard]] Result
    fit(const std::vector<double>& event_times,
        double T,
        Params initial = {0.5, 0.3, 1.0}) const;

    /**
     * @brief Evaluate log-likelihood at given parameters via O(N) recursion.
     *
     * Implements eq. 5 from Bacry, Mastromatteo & Muzy (2015), p. 8.
     * The recursive Rᵢ formula (eq. 6, p. 8) runs in O(N) time.
     *
     * @param p      Parameters (μ, α, β).
     * @param times  Sorted event times.
     * @param T      Observation window.
     * @return       Log-likelihood value (higher = better).
     */
    [[nodiscard]] double
    log_likelihood(const Params& p,
                   const std::vector<double>& times,
                   double T) const noexcept;

    /**
     * @brief Compute the recursive Rᵢ sequence (O(N)).
     *
     * Implements eq. 6 (Bacry et al. 2015):
     *   R₁ = 0
     *   Rᵢ = (Rᵢ₋₁ + 1) · exp(−β · (tᵢ − tᵢ₋₁))
     *
     * @param times  Sorted event times.
     * @param beta   Decay rate.
     * @return       Vector R of size equal to times.size().
     */
    [[nodiscard]] std::vector<double>
    compute_R(const std::vector<double>& times, double beta) const noexcept;

    /**
     * @brief Conditional intensity λ(t) at a specific query time t.
     *
     * λ(t) = μ + α · Σ_{t_i < t} exp(−β(t − t_i))
     *
     * Uses the recursive R representation for efficiency.
     *
     * @param p      Parameters.
     * @param times  Sorted event times.
     * @param t      Query time (must be ≥ 0).
     * @return       λ(t) ≥ 0.
     */
    [[nodiscard]] double
    intensity_at(const Params& p,
                 const std::vector<double>& times,
                 double t) const noexcept;

    /**
     * @brief Kolmogorov–Smirnov test on the compensated residual process.
     *
     * Under the fitted model the compensated times Λ(tᵢ) = ∫₀^{tᵢ} λ(s)ds
     * should form a standard Poisson process on [0, ∫₀ᵀ λ(s)ds].
     * The inter-arrival times Λ(t_{i+1}) − Λ(t_i) should be Exp(1).
     *
     * @param p      Fitted parameters.
     * @param times  Sorted event times.
     * @param T      Observation window.
     * @return       KS statistic ∈ [0, 1] (well-calibrated if < 0.05 at 95%).
     */
    [[nodiscard]] double
    ks_test(const Params& p,
            const std::vector<double>& times,
            double T) const noexcept;

private:
    /**
     * @brief Nelder-Mead simplex minimization (no external dependencies).
     *
     * Operates on log-transformed parameters [log(μ), log(α), log(β)] to
     * enforce positivity naturally.
     *
     * @param objective  Callable (Params) → double (value to minimize).
     * @param initial    Starting parameter values.
     * @param max_iter   Maximum iterations.
     * @param tol        Convergence tolerance (simplex diameter).
     * @return           Best Params found and convergence flag.
     */
    [[nodiscard]] std::pair<Params, bool>
    nelder_mead(const std::function<double(const Params&)>& objective,
                Params initial,
                int    max_iter = 1000,
                double tol      = 1e-8) const;
};

}  // namespace hfm
