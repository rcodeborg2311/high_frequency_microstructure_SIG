/**
 * @file feature_engine.cpp
 * @brief Real-time LOB feature computation (OFI, spread, imbalance).
 *
 * Reference: Cont, Kukanov, Stoikov (2014), JFE 12(1):47-88, eqs. 2–3, p. 51.
 */

#include "feature_engine.hpp"

#include <cmath>
#include <stdexcept>

namespace hfm {

// ── Constructor ─────────────────────────────────────────────────────────────

FeatureEngine::FeatureEngine(int window, int n_levels)
    : window_(window), n_levels_(n_levels) {
    if (window_ <= 0)   throw std::invalid_argument("window must be > 0");
    if (n_levels_ <= 0) throw std::invalid_argument("n_levels must be > 0");
    if (n_levels_ > 10) throw std::invalid_argument("n_levels must be ≤ 10");
}

// ── Per-level OFI ───────────────────────────────────────────────────────────

FeatureEngine::OFIComponents
FeatureEngine::compute_ofi_level(const LOBSnapshot& prev,
                                  const LOBSnapshot& curr,
                                  int level) const noexcept {
    // Cont, Kukanov & Stoikov (2014), eq. 2 (bid) and eq. 3 (ask), p. 51.
    const double pb = prev.bids[level].price;
    const double vb_prev = static_cast<double>(prev.bids[level].volume);
    const double cb = curr.bids[level].price;
    const double vb_curr = static_cast<double>(curr.bids[level].volume);

    const double pa = prev.asks[level].price;
    const double va_prev = static_cast<double>(prev.asks[level].volume);
    const double ca = curr.asks[level].price;
    const double va_curr = static_cast<double>(curr.asks[level].volume);

    // Bid component: e_n(bid)
    //   = V_n^b · 1{P_n^b ≥ P_{n-1}^b}
    //   − V_n^b · 1{P_n^b < P_{n-1}^b}
    //   − V_{n-1}^b · 1{P_n^b ≤ P_{n-1}^b}
    const double bid_up   = (cb >= pb) ? vb_curr : 0.0;
    const double bid_down = (cb <  pb) ? vb_curr : 0.0;
    const double bid_gone = (cb <= pb) ? vb_prev : 0.0;
    const double e_bid    = bid_up - bid_down - bid_gone;

    // Ask component: e_n(ask)
    //   = −V_n^a · 1{P_n^a ≤ P_{n-1}^a}
    //   + V_n^a · 1{P_n^a > P_{n-1}^a}
    //   + V_{n-1}^a · 1{P_n^a ≥ P_{n-1}^a}
    const double ask_down = (ca <= pa) ? va_curr : 0.0;
    const double ask_up   = (ca >  pa) ? va_curr : 0.0;
    const double ask_gone = (ca >= pa) ? va_prev : 0.0;
    const double e_ask    = -ask_down + ask_up + ask_gone;

    OFIComponents result;
    result.bid   = e_bid;
    result.ask   = e_ask;
    result.total = e_bid + e_ask;
    return result;
}

// ── Multi-level OFI ─────────────────────────────────────────────────────────

double FeatureEngine::compute_multi_level_ofi(const LOBSnapshot& prev,
                                               const LOBSnapshot& curr,
                                               int n_levels) const noexcept {
    double total = 0.0;
    const int levels = std::min(n_levels, std::min(prev.n_levels, curr.n_levels));
    for (int l = 0; l < levels; ++l) {
        total += compute_ofi_level(prev, curr, l).total;
    }
    return total;
}

// ── Streaming update ─────────────────────────────────────────────────────────

double FeatureEngine::update(const LOBSnapshot& snapshot) {
    if (!prev_snapshot_.has_value()) {
        prev_snapshot_ = snapshot;
        ofi_history_.push_back(0.0);
        running_ofi_sum_abs_ = 0.0;
        running_ofi_sum_     = 0.0;
        return 0.0;
    }

    const double raw_ofi = compute_multi_level_ofi(*prev_snapshot_, snapshot, n_levels_);

    // Maintain rolling window
    ofi_history_.push_back(raw_ofi);
    running_ofi_sum_     += raw_ofi;
    running_ofi_sum_abs_ += std::abs(raw_ofi);

    if (static_cast<int>(ofi_history_.size()) > window_) {
        const double evicted = ofi_history_.front();
        ofi_history_.pop_front();
        running_ofi_sum_     -= evicted;
        running_ofi_sum_abs_ -= std::abs(evicted);
    }

    prev_snapshot_ = snapshot;

    // Normalized OFI ∈ [−1, +1]
    const double denom = running_ofi_sum_abs_ + EPS;
    return running_ofi_sum_ / denom;
}

// ── Full feature vector ──────────────────────────────────────────────────────

std::vector<double>
FeatureEngine::feature_vector(const LOBSnapshot& prev,
                               const LOBSnapshot& curr) const {
    const int dim = feature_dim();
    std::vector<double> feat(dim, 0.0);

    // Features 0..n_levels-1: per-level OFI (unnormalized)
    for (int l = 0; l < n_levels_; ++l) {
        feat[l] = compute_ofi_level(prev, curr, l).total;
    }

    // Features n_levels..2n_levels-1: per-level volume imbalance (bid-ask)/(bid+ask)
    for (int l = 0; l < n_levels_; ++l) {
        const double vb = static_cast<double>(curr.bids[l].volume);
        const double va = static_cast<double>(curr.asks[l].volume);
        const double denom = vb + va + EPS;
        feat[n_levels_ + l] = (vb - va) / denom;
    }

    // Feature 2*n_levels: rolling normalized OFI (from state, if available)
    feat[2 * n_levels_] = (running_ofi_sum_abs_ > EPS)
                              ? running_ofi_sum_ / (running_ofi_sum_abs_ + EPS)
                              : 0.0;

    // Feature 2*n_levels+1: quoted spread
    feat[2 * n_levels_ + 1] = curr.spread();

    // Feature 2*n_levels+2: mid-price return
    const double mp   = curr.mid_price();
    const double mp0  = prev.mid_price();
    feat[2 * n_levels_ + 2] = (mp0 > EPS) ? (mp - mp0) / mp0 : 0.0;

    // Feature 2*n_levels+3: best-bid OBI
    feat[2 * n_levels_ + 3] = curr.obi();

    return feat;
}

// ── Accessors ────────────────────────────────────────────────────────────────

double FeatureEngine::normalized_ofi() const noexcept {
    const double denom = running_ofi_sum_abs_ + EPS;
    return running_ofi_sum_ / denom;
}

void FeatureEngine::reset() noexcept {
    prev_snapshot_.reset();
    ofi_history_.clear();
    running_ofi_sum_abs_ = 0.0;
    running_ofi_sum_     = 0.0;
}

}  // namespace hfm
