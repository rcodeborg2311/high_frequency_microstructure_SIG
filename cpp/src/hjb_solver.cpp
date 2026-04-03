/**
 * @file hjb_solver.cpp
 * @brief HJB PDE optimal quoting engine implementation.
 *
 * Reference: Cartea, Jaimungal, Penalva (2015), CUP, Chapter 4.
 * Avellaneda & Stoikov (2008), Quantitative Finance 8(3):217-224.
 */

#include "hjb_solver.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace hfm {

// ── Constructors ─────────────────────────────────────────────────────────────

HJBSolver::HJBSolver() : cfg_({}) {}
HJBSolver::HJBSolver(Config cfg) : cfg_(cfg) {}

// ── QuotingPolicy::get_quotes ────────────────────────────────────────────────

std::pair<double, double>
HJBSolver::QuotingPolicy::get_quotes(double t, int q) const noexcept {
    const int qi = q + Q_max;  // inventory index into [0, 2*Q_max]

    if (qi < 0 || qi >= 2 * Q_max + 1) {
        return {std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()};
    }

    // Map t ∈ [0, T] to time index
    const double frac = t / T;
    const int t_idx = std::min(static_cast<int>(frac * Nt), Nt - 1);

    return {bid_spread[t_idx][qi], ask_spread[t_idx][qi]};
}

// ── Value function via backward Euler ───────────────────────────────────────

std::vector<std::vector<double>> HJBSolver::solve_value_function() const {
    const int Q_max = cfg_.Q_max;
    const int Nt    = cfg_.Nt;
    const int nQ    = 2 * Q_max + 1;
    const double dt = cfg_.T / Nt;

    // h[t_idx][q + Q_max]: value function minus linear inventory term.
    // h(T, q) = −κ·q²  (CJP eq. 4.18, terminal condition, p. 92)
    std::vector<std::vector<double>> h(Nt + 1, std::vector<double>(nQ, 0.0));

    // Terminal condition: h(T, q) = −κ·q²
    for (int qi = 0; qi < nQ; ++qi) {
        const int q  = qi - Q_max;
        h[Nt][qi]    = -cfg_.kappa * static_cast<double>(q * q);
    }

    // A/(k·e): pre-computed constant (Avellaneda-Stoikov fill-rate optimization)
    const double A_over_ke = cfg_.A / (cfg_.k * std::exp(1.0));

    // Backward induction: t from Nt-1 down to 0
    for (int t = Nt - 1; t >= 0; --t) {
        for (int qi = 0; qi < nQ; ++qi) {
            const int q = qi - Q_max;

            // Bid contribution: if q < Q_max (can increase inventory)
            double A_bid = 0.0;
            if (qi + 1 < nQ) {
                const double delta_h_bid = h[t + 1][qi + 1] - h[t + 1][qi];
                A_bid = A_over_ke * std::exp(cfg_.k * delta_h_bid);
            }

            // Ask contribution: if q > -Q_max (can decrease inventory)
            double A_ask = 0.0;
            if (qi - 1 >= 0) {
                const double delta_h_ask = h[t + 1][qi - 1] - h[t + 1][qi];
                A_ask = A_over_ke * std::exp(cfg_.k * delta_h_ask);
            }

            // Backward Euler update:
            // ∂h/∂t = φq²σ² − A_bid_opt − A_ask_opt
            const double phi_term = cfg_.phi
                                    * static_cast<double>(q * q)
                                    * cfg_.sigma * cfg_.sigma;

            h[t][qi] = h[t + 1][qi] + dt * (phi_term - A_bid - A_ask);
        }
    }

    return h;
}

// ── Main solve() ─────────────────────────────────────────────────────────────

HJBSolver::QuotingPolicy HJBSolver::solve() const {
    const int Q_max = cfg_.Q_max;
    const int Nt    = cfg_.Nt;
    const int nQ    = 2 * Q_max + 1;

    const auto h = solve_value_function();

    QuotingPolicy policy;
    policy.Q_max = Q_max;
    policy.Nt    = Nt;
    policy.T     = cfg_.T;
    policy.k     = cfg_.k;
    policy.bid_spread.assign(Nt, std::vector<double>(nQ, 0.0));
    policy.ask_spread.assign(Nt, std::vector<double>(nQ, 0.0));

    for (int t = 0; t < Nt; ++t) {
        for (int qi = 0; qi < nQ; ++qi) {
            // Bid spread: δ*_bid = max(0, 1/k − Δh_bid)
            if (qi + 1 < nQ) {
                const double delta_h = h[t][qi + 1] - h[t][qi];
                policy.bid_spread[t][qi] = std::max(0.0, 1.0 / cfg_.k - delta_h);
            } else {
                // At maximum inventory: quote extremely wide bid (discourage buying more)
                policy.bid_spread[t][qi] = 1e6;
            }

            // Ask spread: δ*_ask = max(0, 1/k − Δh_ask)
            if (qi - 1 >= 0) {
                const double delta_h = h[t][qi - 1] - h[t][qi];
                policy.ask_spread[t][qi] = std::max(0.0, 1.0 / cfg_.k - delta_h);
            } else {
                // At minimum inventory: quote extremely wide ask (discourage selling more)
                policy.ask_spread[t][qi] = 1e6;
            }
        }
    }

    return policy;
}

// ── Closed-form Avellaneda-Stoikov approximation ─────────────────────────────

std::pair<double, double>
HJBSolver::closed_form_quotes(double t, int q) const noexcept {
    // Avellaneda & Stoikov (2008), eq. 17 (inventory-adjusted):
    //   δ*_bid(t,q) = 1/k + (γσ²/2)·(T−t) + γσ²·(q − ½)·(T−t)/k
    //   δ*_ask(t,q) = 1/k + (γσ²/2)·(T−t) − γσ²·(q + ½)·(T−t)/k
    const double tau    = cfg_.T - t;  // time remaining
    const double base   = 1.0 / cfg_.k;
    const double risk   = cfg_.gamma * cfg_.sigma * cfg_.sigma * tau;
    const double inv_adj = risk / cfg_.k;

    const double bid = base + 0.5 * risk + (static_cast<double>(q) - 0.5) * inv_adj;
    const double ask = base + 0.5 * risk - (static_cast<double>(q) + 0.5) * inv_adj;

    return {std::max(0.0, bid), std::max(0.0, ask)};
}

// ── Comparison to closed form ─────────────────────────────────────────────────

double HJBSolver::compare_to_closed_form() const {
    const auto policy = solve();
    const int  Q_max  = cfg_.Q_max;
    const int  Nt     = cfg_.Nt;
    const double dt   = cfg_.T / Nt;

    double max_diff = 0.0;

    for (int t_idx = 0; t_idx < Nt; ++t_idx) {
        const double t = t_idx * dt;
        for (int qi = 0; qi < 2 * Q_max + 1; ++qi) {
            const int q = qi - Q_max;

            const auto [cf_bid, cf_ask] = closed_form_quotes(t, q);

            const double num_bid = policy.bid_spread[t_idx][qi];
            const double num_ask = policy.ask_spread[t_idx][qi];

            // Skip boundary inventory levels (infinite spreads are expected)
            if (num_bid > 1e5 || num_ask > 1e5) continue;

            max_diff = std::max(max_diff, std::abs(num_bid - cf_bid));
            max_diff = std::max(max_diff, std::abs(num_ask - cf_ask));
        }
    }

    return max_diff;
}

}  // namespace hfm
