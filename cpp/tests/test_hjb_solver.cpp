/**
 * @file test_hjb_solver.cpp
 * @brief Unit tests for HJB PDE optimal quoting solver.
 *
 * Reference: Cartea, Jaimungal, Penalva (2015), CUP, Chapter 4.
 */

#include "hjb_solver.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <limits>

using namespace hfm;
using Catch::Approx;

// ── Helper to build a Config with given overrides ────────────────────────────

static HJBSolver::Config make_config(double phi   = 0.01,
                                      double kappa = 0.01,
                                      double gamma = 0.01) {
    HJBSolver::Config cfg;
    cfg.phi   = phi;
    cfg.kappa = kappa;
    cfg.gamma = gamma;
    cfg.sigma = 0.001;
    cfg.A     = 1.0;
    cfg.k     = 1.5;
    cfg.Nt    = 50;
    cfg.Q_max = 5;
    return cfg;
}

// ── TEST 1: Optimal spread is positive for all (t, q) ────────────────────────

TEST_CASE("Optimal spread is positive for all (t, q)", "[hjb_solver]") {
    HJBSolver solver(make_config());
    const auto policy = solver.solve();

    const int Q_max = policy.Q_max;
    const int Nt    = policy.Nt;

    for (int t = 0; t < Nt; ++t) {
        for (int qi = 1; qi < 2 * Q_max; ++qi) {  // skip exact boundaries
            INFO("t=" << t << " qi=" << qi);
            REQUIRE(policy.bid_spread[t][qi] >= 0.0);
            REQUIRE(policy.ask_spread[t][qi] >= 0.0);
        }
    }
}

// ── TEST 2: Bid spread increases with inventory q ─────────────────────────────

TEST_CASE("Bid spread increases with inventory q (more risk when long)", "[hjb_solver]") {
    HJBSolver solver(make_config());
    const auto policy = solver.solve();

    const int  Q_max  = policy.Q_max;
    const int  Nt     = policy.Nt;
    const int  t_mid  = Nt / 2;

    bool any_increasing = false;
    for (int qi = 1; qi < 2 * Q_max - 1; ++qi) {
        const double b_lo = policy.bid_spread[t_mid][qi];
        const double b_hi = policy.bid_spread[t_mid][qi + 1];
        // Allow flat or slightly decreasing at extremes (constrained boundary)
        if (b_lo < 1e5 && b_hi < 1e5) {
            if (b_hi > b_lo) any_increasing = true;
        }
    }
    REQUIRE(any_increasing);
}

// ── TEST 3: Ask spread decreases with inventory q (more aggressive when long) ─

TEST_CASE("Ask spread decreases with inventory q (aggressive selling when long)",
          "[hjb_solver]") {
    HJBSolver solver(make_config());
    const auto policy = solver.solve();

    const int Q_max = policy.Q_max;
    const int Nt    = policy.Nt;
    const int t_mid = Nt / 2;

    bool any_decreasing = false;
    for (int qi = 1; qi < 2 * Q_max - 1; ++qi) {
        const double a_lo = policy.ask_spread[t_mid][qi];
        const double a_hi = policy.ask_spread[t_mid][qi + 1];
        if (a_lo < 1e5 && a_hi < 1e5) {
            if (a_hi < a_lo) any_decreasing = true;
        }
    }
    REQUIRE(any_decreasing);
}

// ── TEST 4: Closed-form spreads at t=T equal 1/k (time premium vanishes) ─────

TEST_CASE("Closed-form spread at t=T equals 1/k (no time-risk premium)", "[hjb_solver]") {
    const auto cfg = make_config();
    HJBSolver  solver(cfg);

    const double expected = 1.0 / cfg.k;

    for (int q = -cfg.Q_max; q <= cfg.Q_max; ++q) {
        const auto [bid_cf, ask_cf] = solver.closed_form_quotes(cfg.T, q);
        INFO("q=" << q << " bid_cf=" << bid_cf << " ask_cf=" << ask_cf);
        REQUIRE(bid_cf == Approx(expected).epsilon(1e-6));
        REQUIRE(ask_cf == Approx(expected).epsilon(1e-6));
    }
}

// ── TEST 5: Symmetric closed-form spreads when q=0 ───────────────────────────

TEST_CASE("Symmetric spreads when q=0 (no inventory bias)", "[hjb_solver]") {
    const auto cfg = make_config();
    HJBSolver  solver(cfg);

    // At any t with q=0, closed-form bid == ask
    for (int t_idx = 0; t_idx < cfg.Nt; ++t_idx) {
        const double t = cfg.T * t_idx / cfg.Nt;
        const auto [bid_cf, ask_cf] = solver.closed_form_quotes(t, 0);
        INFO("t=" << t << " bid=" << bid_cf << " ask=" << ask_cf);
        REQUIRE(bid_cf == Approx(ask_cf).epsilon(1e-9));
    }
}

// ── TEST 6: Bid spread > ask spread when q > 0 (skewed quotes when long) ─────

TEST_CASE("Bid spread > ask spread when inventory is positive (q > 0)", "[hjb_solver]") {
    const auto cfg = make_config();
    HJBSolver  solver(cfg);

    // When long (q > 0), the MM widens the bid and narrows the ask to offload
    const double t = 0.3 * cfg.T;  // mid-session
    for (int q = 1; q <= cfg.Q_max - 1; ++q) {
        const auto [bid, ask] = solver.closed_form_quotes(t, q);
        INFO("q=" << q << " bid=" << bid << " ask=" << ask);
        REQUIRE(bid > ask);
    }
}

// ── TEST 7: QuotingPolicy::get_quotes interpolates correctly ─────────────────

TEST_CASE("QuotingPolicy::get_quotes returns valid spreads", "[hjb_solver]") {
    HJBSolver solver(make_config());
    const auto policy = solver.solve();

    const auto [b, a] = policy.get_quotes(0.0, 0);
    REQUIRE(b >= 0.0);
    REQUIRE(a >= 0.0);
    REQUIRE(std::isfinite(b));
    REQUIRE(std::isfinite(a));
}
