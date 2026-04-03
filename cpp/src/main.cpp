/**
 * @file main.cpp
 * @brief HF Microstructure Platform demo: stream synthetic LOB, compute signals,
 *        fit Hawkes process, solve HJB, print optimal quote table.
 */

#include "feature_engine.hpp"
#include "hawkes_mle.hpp"
#include "hjb_solver.hpp"
#include "lob_parser.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// ── Synthetic LOB generator ──────────────────────────────────────────────────

static std::vector<hfm::LOBSnapshot>
generate_synthetic_lob(int n_steps, unsigned seed = 42) {
    std::mt19937_64                   rng(seed);
    std::normal_distribution<double>  gauss(0.0, 1.0);
    std::lognormal_distribution<double> log_spread(std::log(0.0002), 0.25);
    std::lognormal_distribution<double> log_vol(std::log(500.0), 0.4);

    std::vector<hfm::LOBSnapshot> snaps;
    snaps.reserve(n_steps);

    double mid = 100.0;
    const double sigma_step = 0.0001;  // price diffusion per tick

    for (int i = 0; i < n_steps; ++i) {
        // GBM mid price
        mid *= std::exp(sigma_step * gauss(rng));

        const double half_spread = std::max(0.0001, log_spread(rng) * 0.5);

        hfm::LOBSnapshot snap;
        snap.time     = i * 0.1;  // 100 ms per tick
        snap.n_levels = 5;

        for (int l = 0; l < 5; ++l) {
            const double depth_factor = std::exp(-0.5 * l);
            const double vol = std::max(1.0, log_vol(rng) * depth_factor);
            snap.bids[l].price  = mid - half_spread * (1 + 0.5 * l);
            snap.bids[l].volume = static_cast<long>(vol);
            snap.asks[l].price  = mid + half_spread * (1 + 0.5 * l);
            snap.asks[l].volume = static_cast<long>(vol * std::exp(-0.5 * l));
        }

        snaps.push_back(snap);
    }
    return snaps;
}

// ── Synthetic Hawkes event times ─────────────────────────────────────────────

static std::vector<double>
generate_hawkes_events(double mu, double alpha, double beta,
                        double T, unsigned seed = 42) {
    // Ogata's thinning algorithm for Hawkes simulation
    std::mt19937_64             rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::exponential_distribution<double>  exp_dist(1.0);

    std::vector<double> times;
    double t = 0.0;
    double lambda_bar = mu;  // upper bound on intensity

    while (t < T) {
        t += exp_dist(rng) / lambda_bar;
        if (t >= T) break;

        // Compute exact intensity at t
        double intensity = mu;
        for (const double ti : times) {
            intensity += alpha * std::exp(-beta * (t - ti));
        }

        // Accept with probability intensity / lambda_bar
        if (uniform(rng) <= intensity / lambda_bar) {
            times.push_back(t);
        }

        // Update upper bound
        lambda_bar = mu;
        for (const double ti : times) {
            lambda_bar += alpha * std::exp(-beta * (t - ti));
        }
        lambda_bar = std::max(lambda_bar, mu);
    }

    return times;
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "  HF Market Microstructure Signal Platform  ─  Demo\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    // ── 1. Generate synthetic LOB and compute OFI ──────────────────────────
    constexpr int N_STEPS = 2000;
    const auto snaps = generate_synthetic_lob(N_STEPS);

    hfm::FeatureEngine eng(100, 5);
    std::vector<double> ofi_series;
    ofi_series.reserve(N_STEPS);

    for (const auto& snap : snaps) {
        ofi_series.push_back(eng.update(snap));
    }

    // Rolling statistics on OFI
    double ofi_sum = 0.0, ofi_sum2 = 0.0;
    for (double v : ofi_series) { ofi_sum += v; ofi_sum2 += v * v; }
    const double ofi_mean = ofi_sum / N_STEPS;
    const double ofi_std  = std::sqrt(ofi_sum2 / N_STEPS - ofi_mean * ofi_mean);

    std::cout << "── Order Flow Imbalance (OFI) ──────────────────────────────\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Ticks processed : " << N_STEPS << "\n";
    std::cout << "  OFI mean        : " << ofi_mean << "\n";
    std::cout << "  OFI std dev     : " << ofi_std  << "\n";
    std::cout << "  OFI range       : [";
    const auto [mn, mx] = std::minmax_element(ofi_series.begin(), ofi_series.end());
    std::cout << *mn << ", " << *mx << "]\n\n";

    // ── 2. Hawkes MLE ────────────────────────────────────────────────────────
    // Generate synthetic events with known parameters for validation
    constexpr double TRUE_MU    = 2.0;
    constexpr double TRUE_ALPHA = 0.8;
    constexpr double TRUE_BETA  = 3.0;
    constexpr double T_WINDOW   = 500.0;

    const auto event_times = generate_hawkes_events(TRUE_MU, TRUE_ALPHA, TRUE_BETA, T_WINDOW);

    std::cout << "── Hawkes Process MLE ──────────────────────────────────────\n";
    std::cout << "  True   (μ,α,β) : (" << TRUE_MU << ", " << TRUE_ALPHA
              << ", " << TRUE_BETA << ")\n";
    std::cout << "  Events generated: " << event_times.size() << "\n";

    hfm::HawkesMLE fitter;
    const auto result = fitter.fit(event_times, T_WINDOW);

    std::cout << std::setprecision(4);
    std::cout << "  Fitted (μ,α,β) : (" << result.params.mu << ", "
              << result.params.alpha << ", " << result.params.beta << ")\n";
    std::cout << "  Branching ratio : " << result.params.branching_ratio() << "\n";
    std::cout << "  Excit. half-life: " << result.params.excitation_halflife() << " s\n";
    std::cout << "  Log-likelihood  : " << result.log_likelihood << "\n";
    std::cout << "  Converged       : " << (result.converged ? "yes" : "no") << "\n";
    const double ks = fitter.ks_test(result.params, event_times, T_WINDOW);
    std::cout << "  KS statistic    : " << ks
              << (ks < 0.05 ? "  ✓ (< 0.05)" : "  ✗ (> 0.05)") << "\n\n";

    // ── 3. HJB Optimal Quoting ────────────────────────────────────────────────
    hfm::HJBSolver::Config hjb_cfg;
    hjb_cfg.sigma = 0.001;
    hjb_cfg.phi   = 0.01;
    hjb_cfg.kappa = 0.01;
    hjb_cfg.A     = 1.0;
    hjb_cfg.k     = 1.5;
    hjb_cfg.gamma = hjb_cfg.phi;

    hfm::HJBSolver solver(hjb_cfg);
    const auto policy = solver.solve();

    std::cout << "── HJB Optimal Quotes — t = T/2 ────────────────────────────\n";
    std::cout << std::setw(8)  << "q"
              << std::setw(14) << "δ*_bid"
              << std::setw(14) << "δ*_ask"
              << std::setw(14) << "half-spread"
              << std::setw(14) << "skew\n";
    std::cout << std::string(64, '-') << "\n";

    const double t_half = hjb_cfg.T * 0.5;
    for (int q : {-5, -3, 0, 3, 5}) {
        const auto [bid, ask] = policy.get_quotes(t_half, q);
        std::cout << std::setw(8)  << q
                  << std::setw(14) << bid
                  << std::setw(14) << ask
                  << std::setw(14) << (bid + ask) * 0.5
                  << std::setw(14) << (ask - bid) << "\n";
    }

    const double max_diff = solver.compare_to_closed_form();
    std::cout << "\n  Max |numerical − closed-form| : " << max_diff << "\n\n";

    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "  Demo complete.\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";

    return 0;
}
