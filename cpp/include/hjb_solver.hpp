#pragma once
/**
 * @file hjb_solver.hpp
 * @brief HJB PDE optimal quoting engine (Cartea-Jaimungal-Penalva 2015).
 *
 * Solves the Hamilton-Jacobi-Bellman equation for a market maker who maximizes
 * expected CARA utility of terminal wealth subject to inventory risk.
 *
 * Reference:
 *   Cartea, Jaimungal, Penalva (2015), "Algorithmic and High-Frequency
 *   Trading", Cambridge University Press, Chapter 4.
 *
 * Value function decomposition (CJP §4.2):
 *   V(t, x, q, s) = x + qs + h(t, q)
 *
 * HJB for h(t, q) (derived from CJP eq. 4.21, p. 93):
 *   ∂h/∂t = φq²σ²
 *           − (A / (k·e)) · exp(k · (h(t,q+1) − h(t,q)))   [bid contribution]
 *           − (A / (k·e)) · exp(k · (h(t,q−1) − h(t,q)))   [ask contribution]
 *
 * Terminal condition:
 *   h(T, q) = −κ · q²
 *
 * Optimal spreads from the value function (unconstrained, CJP eq. 4.24, p. 95):
 *   δ*_bid(t, q) = max(0, 1/k − [h(t,q+1) − h(t,q)])
 *   δ*_ask(t, q) = max(0, 1/k − [h(t,q−1) − h(t,q)])
 *
 * Closed-form Avellaneda-Stoikov approximation (large-k limit,
 * Avellaneda & Stoikov (2008), Quantitative Finance 8(3):217–224, eq. 17):
 *   δ*_bid(t,q) = 1/k + (γσ²/2)·(T−t) + γσ²·(q − ½)·(T−t)/k
 *   δ*_ask(t,q) = 1/k + (γσ²/2)·(T−t) − γσ²·(q + ½)·(T−t)/k
 */

#include <utility>
#include <vector>

namespace hfm {

/**
 * @brief Numerical HJB solver returning a full quoting policy table.
 *
 * The solver is stateless; call solve() as many times as needed with
 * different Config objects.
 */
class HJBSolver {
public:
    // ── Configuration ─────────────────────────────────────────────────────────

    /// Solver parameters matching Avellaneda-Stoikov / CJP notation.
    struct Config {
        double T     = 1.0 / 6.5;  ///< Trading session length in years (≈ 6.5 h).
        int    Q_max = 10;          ///< Maximum |inventory| in shares.
        int    Nt    = 100;         ///< Number of time steps for backward Euler.
        double sigma = 0.001;       ///< Mid-price diffusion per unit time.
        double phi   = 0.01;        ///< Running inventory penalty (CARA risk aversion φ).
        double kappa = 0.01;        ///< Terminal inventory penalty κ.
        double A     = 1.0;         ///< Fill-rate constant in Λ(δ) = A·exp(−k·δ).
        double k     = 1.5;         ///< Fill-rate decay (order-book depth parameter).
        double gamma = 0.01;        ///< Risk aversion γ = φ for CARA utility.
    };

    // ── Quoting policy table ───────────────────────────────────────────────────

    /**
     * @brief Full quoting policy: optimal bid and ask spreads for every (t, q).
     *
     * Dimensions:
     *   bid_spread[t_idx][q + Q_max]  (t_idx ∈ [0, Nt), q ∈ [−Q_max, Q_max])
     *   ask_spread[t_idx][q + Q_max]
     */
    struct QuotingPolicy {
        std::vector<std::vector<double>> bid_spread;  ///< Outer: time, inner: inventory.
        std::vector<std::vector<double>> ask_spread;
        int    Q_max = 10;
        int    Nt    = 100;
        double T     = 1.0 / 6.5;
        double k     = 1.5;

        /**
         * @brief Look up the optimal bid and ask spread at time t and inventory q.
         *
         * @param t  Current time t ∈ [0, T].
         * @param q  Current inventory q ∈ [−Q_max, Q_max].
         * @return   {δ*_bid, δ*_ask} (both ≥ 0).
         */
        [[nodiscard]] std::pair<double, double>
        get_quotes(double t, int q) const noexcept;
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Construct solver with default configuration.
    HJBSolver();

    /// Construct solver with the given configuration.
    explicit HJBSolver(Config cfg);

    // ── Core API ──────────────────────────────────────────────────────────────

    /**
     * @brief Solve the HJB PDE via backward Euler and return the quoting policy.
     *
     * Runs in O(Nt × Q_max) time.  The value function h(t, q) is computed
     * backward from t = T to t = 0, then optimal spreads are extracted.
     *
     * @return  Full quoting policy table.
     */
    [[nodiscard]] QuotingPolicy solve() const;

    /**
     * @brief Closed-form Avellaneda-Stoikov approximation (large-k limit).
     *
     * From Avellaneda & Stoikov (2008), eq. 17, for a single (t, q) point.
     *
     * @param t  Current time.
     * @param q  Current inventory.
     * @return   {δ*_bid_cf, δ*_ask_cf}.
     */
    [[nodiscard]] std::pair<double, double>
    closed_form_quotes(double t, int q) const noexcept;

    /**
     * @brief Compare the numerical HJB policy to the closed-form approximation.
     *
     * Solves the HJB numerically, evaluates the closed-form at the same
     * grid points, and returns the maximum absolute difference across all (t, q).
     *
     * @return  max_{t,q} | δ*_numerical − δ*_closed_form |.
     */
    [[nodiscard]] double compare_to_closed_form() const;

    /// Access the current configuration.
    [[nodiscard]] const Config& config() const noexcept { return cfg_; }

private:
    Config cfg_;

    /// Compute the h(t, q) value function grid via backward Euler.
    /// Returns a 2-D array of size (Nt+1) × (2·Q_max+1).
    [[nodiscard]] std::vector<std::vector<double>> solve_value_function() const;
};

}  // namespace hfm
