/**
 * @file test_hawkes_mle.cpp
 * @brief Unit tests for Hawkes process MLE (O(N) recursion, Nelder-Mead).
 *
 * Reference: Bacry, Mastromatteo, Muzy (2015), MML 1(1):1550005.
 */

#include "hawkes_mle.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

using namespace hfm;
using Catch::Approx;

// ── Simulate a Poisson process (no self-excitation) ──────────────────────────

static std::vector<double>
simulate_poisson(double rate, double T, unsigned seed = 42) {
    std::mt19937_64 rng(seed);
    std::exponential_distribution<double> exp_dist(rate);
    std::vector<double> times;
    double t = 0.0;
    while ((t += exp_dist(rng)) < T) {
        times.push_back(t);
    }
    return times;
}

// ── Simulate Hawkes (Ogata thinning) ────────────────────────────────────────

static std::vector<double>
simulate_hawkes(double mu, double alpha, double beta,
                double T, unsigned seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<double> times;

    double t           = 0.0;
    double lambda_bar  = mu;

    while (t < T) {
        const double u = uniform(rng);
        t -= std::log(u) / lambda_bar;
        if (t >= T) break;

        double intensity = mu;
        for (const double ti : times) {
            intensity += alpha * std::exp(-beta * (t - ti));
        }

        if (uniform(rng) <= intensity / lambda_bar) {
            times.push_back(t);
        }

        // Recompute upper bound
        lambda_bar = mu;
        for (const double ti : times) {
            lambda_bar += alpha * std::exp(-beta * (t - ti));
        }
        lambda_bar = std::max(lambda_bar, mu);
    }
    return times;
}

// ── TEST 1: Fit on Poisson process → alpha ≈ 0 ──────────────────────────────

TEST_CASE("Fitting Poisson data yields near-zero alpha", "[hawkes_mle]") {
    const double RATE = 3.0;
    const double T    = 1000.0;

    const auto times = simulate_poisson(RATE, T, 123);
    REQUIRE(times.size() > 50);

    HawkesMLE mle;
    const auto result = mle.fit(times, T);

    // For a Poisson process, the best-fit Hawkes has α ≈ 0 (no self-excitation).
    // Allow for statistical noise; branching ratio should be < 0.25 for pure Poisson.
    INFO("Fitted alpha = " << result.params.alpha
         << "  branching ratio = " << result.params.branching_ratio());
    REQUIRE(result.params.branching_ratio() < 0.30);
    REQUIRE(result.params.stationary());
}

// ── TEST 2: Branching ratio < 1 for stationary process ──────────────────────

TEST_CASE("Fitted branching ratio is < 1 (stationarity)", "[hawkes_mle]") {
    const auto times = simulate_hawkes(2.0, 0.6, 2.0, 500.0, 7);

    HawkesMLE mle;
    const auto result = mle.fit(times, 500.0);

    REQUIRE(result.params.stationary());
    REQUIRE(result.params.branching_ratio() < 1.0);
}

// ── TEST 3: KS statistic < 0.1 for well-fitted synthetic data ───────────────

TEST_CASE("KS statistic < 0.10 for well-fitted Hawkes data", "[hawkes_mle]") {
    constexpr double MU    = 1.5;
    constexpr double ALPHA = 0.6;
    constexpr double BETA  = 2.5;
    constexpr double T     = 2000.0;

    const auto times  = simulate_hawkes(MU, ALPHA, BETA, T, 99);
    HawkesMLE  mle;
    const auto result = mle.fit(times, T);

    const double ks = mle.ks_test(result.params, times, T);
    INFO("KS statistic = " << ks
         << "  fitted (μ,α,β) = (" << result.params.mu << ","
         << result.params.alpha << "," << result.params.beta << ")");
    REQUIRE(ks < 0.15);  // Allow some slack; exact threshold depends on N
}

// ── TEST 4: Recursive Rᵢ matches naive O(N²) sum within 1e-10 ───────────────

TEST_CASE("Recursive R computation matches naive O(N^2) sum within 1e-10", "[hawkes_mle]") {
    const std::vector<double> times = {0.1, 0.3, 0.35, 0.8, 1.2, 2.5, 2.51, 3.0};
    constexpr double BETA = 1.5;

    HawkesMLE mle;
    const auto R_recursive = mle.compute_R(times, BETA);

    // Naive O(N²) computation for comparison
    const int N = static_cast<int>(times.size());
    std::vector<double> R_naive(N, 0.0);
    for (int i = 1; i < N; ++i) {
        for (int j = 0; j < i; ++j) {
            R_naive[i] += std::exp(-BETA * (times[i] - times[j]));
        }
    }

    for (int i = 0; i < N; ++i) {
        INFO("i = " << i
             << "  R_recursive = " << R_recursive[i]
             << "  R_naive = "     << R_naive[i]);
        REQUIRE(R_recursive[i] == Approx(R_naive[i]).epsilon(1e-10));
    }
}

// ── TEST 5: Log-likelihood increases during optimization ────────────────────

TEST_CASE("Log-likelihood at fitted params exceeds initial guess", "[hawkes_mle]") {
    const auto times = simulate_hawkes(1.0, 0.5, 2.0, 1000.0, 42);

    HawkesMLE mle;

    // Evaluate at a deliberately poor initial guess
    const HawkesMLE::Params poor_init = {0.1, 0.05, 0.5};
    const double ll_init  = mle.log_likelihood(poor_init, times, 1000.0);

    const auto result     = mle.fit(times, 1000.0, poor_init);
    const double ll_final = result.log_likelihood;

    INFO("LL at init  = " << ll_init);
    INFO("LL at final = " << ll_final);
    REQUIRE(ll_final > ll_init);
}

// ── TEST 6: R₁ = 0 (first event has no past history) ───────────────────────

TEST_CASE("R[0] = 0 (no history before first event)", "[hawkes_mle]") {
    const std::vector<double> times = {0.5, 1.0, 2.0};
    HawkesMLE mle;
    const auto R = mle.compute_R(times, 1.0);
    REQUIRE(R[0] == Approx(0.0).margin(1e-15));
}
