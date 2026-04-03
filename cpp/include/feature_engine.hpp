#pragma once
/**
 * @file feature_engine.hpp
 * @brief Real-time LOB feature computation: OFI, spread, imbalance, depth.
 *
 * Implements the multi-level Order Flow Imbalance (OFI) signal from:
 *   Cont, Kukanov, Stoikov (2014), "The Price Impact of Order Book Events",
 *   Journal of Financial Econometrics 12(1):47-88.
 *
 * OFI event definitions (eqs. 2–3, p. 51):
 *   e_n(bid) = V_n^b · 1{P_n^b ≥ P_{n-1}^b}
 *            − V_n^b · 1{P_n^b < P_{n-1}^b}
 *            − V_{n-1}^b · 1{P_n^b ≤ P_{n-1}^b}
 *
 *   e_n(ask) = −V_n^a · 1{P_n^a ≤ P_{n-1}^a}
 *            + V_n^a · 1{P_n^a > P_{n-1}^a}
 *            + V_{n-1}^a · 1{P_n^a ≥ P_{n-1}^a}
 *
 *   OFI_n = e_n(bid) + e_n(ask)
 *
 * Normalized rolling OFI over window W:
 *   ŌFIP(W) = Σ OFI_n / (Σ |OFI_n| + ε)  ∈ [−1, +1]
 */

#include "lob_parser.hpp"

#include <deque>
#include <optional>
#include <vector>

namespace hfm {

/**
 * @brief Computes LOB features from consecutive snapshot pairs.
 *
 * The engine keeps internal state (rolling window, previous snapshot) so
 * callers can push snapshots one at a time and receive updated signals.
 */
class FeatureEngine {
public:
    /// Default number of LOB levels used for multi-level OFI.
    static constexpr int DEFAULT_LEVELS = 5;

    /// Small regularization constant to avoid division by zero.
    static constexpr double EPS = 1e-9;

    /// Per-level OFI decomposition (bid component, ask component, total).
    struct OFIComponents {
        double bid  = 0.0;   ///< e_n(bid): bid side contribution.
        double ask  = 0.0;   ///< e_n(ask): ask side contribution.
        double total = 0.0;  ///< OFI_n = e_n(bid) + e_n(ask).
    };

    /**
     * @brief Construct a FeatureEngine.
     * @param window   Rolling window size (ticks) for normalized OFI.
     * @param n_levels Number of LOB levels to include in multi-level OFI.
     */
    explicit FeatureEngine(int window = 100, int n_levels = DEFAULT_LEVELS);

    // ── Stateless level-specific computation ─────────────────────────────────

    /**
     * @brief Compute the OFI event at a specific LOB level.
     *
     * Implements eqs. 2–3 from Cont, Kukanov & Stoikov (2014), p. 51.
     *
     * @param prev   Snapshot at time n−1.
     * @param curr   Snapshot at time n.
     * @param level  LOB level index (0 = best bid/ask).
     * @return       Decomposed OFI components at the requested level.
     */
    [[nodiscard]] OFIComponents
    compute_ofi_level(const LOBSnapshot& prev,
                      const LOBSnapshot& curr,
                      int level = 0) const noexcept;

    /**
     * @brief Compute multi-level OFI (equal-weighted sum over levels).
     *
     * Aggregates OFI across levels 0..n_levels-1 with equal weights.
     * PCA-based weighting is available via the Python research layer.
     *
     * @param prev     Snapshot at time n−1.
     * @param curr     Snapshot at time n.
     * @param n_levels Number of levels to aggregate.
     * @return         Raw (unnormalized) multi-level OFI.
     */
    [[nodiscard]] double
    compute_multi_level_ofi(const LOBSnapshot& prev,
                             const LOBSnapshot& curr,
                             int n_levels = DEFAULT_LEVELS) const noexcept;

    // ── Stateful streaming interface ──────────────────────────────────────────

    /**
     * @brief Push a new snapshot and return the current normalized OFI.
     *
     * On the first call (no previous snapshot available) returns 0.0.
     *
     * @param snapshot  The latest LOB snapshot.
     * @return          ŌFIP(W) ∈ [−1, +1].
     */
    double update(const LOBSnapshot& snapshot);

    /**
     * @brief Compute the full feature vector from two consecutive snapshots.
     *
     * Feature vector layout (dimension = 2·n_levels + 4):
     *   [0..n_levels-1]        : per-level OFI (unnormalized)
     *   [n_levels..2n_levels-1]: per-level volume imbalance (bid-ask)/(bid+ask)
     *   [2*n_levels]           : normalized OFI (rolling)
     *   [2*n_levels+1]         : quoted spread
     *   [2*n_levels+2]         : mid-price return (curr - prev) / prev
     *   [2*n_levels+3]         : best-bid OBI
     *
     * @param prev  Snapshot at time n−1.
     * @param curr  Snapshot at time n.
     * @return      Feature vector of size 2*n_levels + 4.
     */
    [[nodiscard]] std::vector<double>
    feature_vector(const LOBSnapshot& prev,
                   const LOBSnapshot& curr) const;

    /// Current normalized rolling OFI (last computed by update()).
    [[nodiscard]] double normalized_ofi() const noexcept;

    /// Number of ticks in the rolling window.
    [[nodiscard]] int window() const noexcept { return window_; }

    /// Feature vector dimension.
    [[nodiscard]] int feature_dim() const noexcept { return 2 * n_levels_ + 4; }

    /// Reset internal state (rolling window, previous snapshot).
    void reset() noexcept;

private:
    int  window_;
    int  n_levels_;

    std::optional<LOBSnapshot> prev_snapshot_;
    std::deque<double>         ofi_history_;  ///< Rolling OFI values.
    double                     running_ofi_sum_abs_ = 0.0;
    double                     running_ofi_sum_     = 0.0;
};

}  // namespace hfm
