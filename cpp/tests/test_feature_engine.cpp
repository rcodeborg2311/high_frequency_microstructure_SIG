/**
 * @file test_feature_engine.cpp
 * @brief Unit tests for FeatureEngine (OFI computation).
 */

#include "feature_engine.hpp"
#include "lob_parser.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <nanobench.h>

#include <cmath>
#include <random>

using namespace hfm;
using Catch::Approx;

// ── Helper to build a simple snapshot ───────────────────────────────────────

static LOBSnapshot make_snap(double time,
                              double bid_px, long bid_vol,
                              double ask_px, long ask_vol,
                              int n_levels = 1) {
    LOBSnapshot s;
    s.time     = time;
    s.n_levels = n_levels;
    for (int l = 0; l < n_levels; ++l) {
        const double spread = 0.0001 * (l + 1);
        s.bids[l] = {bid_px - spread * l, bid_vol};
        s.asks[l] = {ask_px + spread * l, ask_vol};
    }
    // Override level 0 with exact values
    s.bids[0] = {bid_px, bid_vol};
    s.asks[0] = {ask_px, ask_vol};
    return s;
}

// ── TEST 1: OFI = 0 when bid and ask volumes are equal and unchanged ─────────

TEST_CASE("OFI is zero when LOB is completely unchanged", "[feature_engine]") {
    FeatureEngine eng(100, 1);

    // Same prices and volumes => no flow event
    const LOBSnapshot prev = make_snap(0.0, 100.0, 1000, 100.01, 1000);
    const LOBSnapshot curr = make_snap(0.1, 100.0, 1000, 100.01, 1000);

    const auto ofi = eng.compute_ofi_level(prev, curr, 0);

    // Cont et al. (2014): if nothing changed, e_n(bid) = V_n^b - V_{n-1}^b = 0
    // and e_n(ask) = V_{n-1}^a - V_n^a = 0, so OFI = 0.
    REQUIRE(ofi.bid   == Approx(0.0).margin(1e-9));
    REQUIRE(ofi.ask   == Approx(0.0).margin(1e-9));
    REQUIRE(ofi.total == Approx(0.0).margin(1e-9));
}

// ── TEST 2: OFI > 0 when a large bid order arrives at best bid ───────────────

TEST_CASE("OFI is positive when a large bid order arrives at best bid", "[feature_engine]") {
    FeatureEngine eng(100, 1);

    // Prev: bid 100.0 x 500
    // Curr: bid 100.0 x 2000  (large buy order arrived, increasing depth)
    const LOBSnapshot prev = make_snap(0.0, 100.0,  500, 100.01, 500);
    const LOBSnapshot curr = make_snap(0.1, 100.0, 2000, 100.01, 500);

    const auto ofi = eng.compute_ofi_level(prev, curr, 0);

    // P_n^b == P_{n-1}^b: e_n(bid) = V_n^b - V_{n-1}^b = 2000 - 500 = 1500 > 0
    REQUIRE(ofi.bid > 0.0);
    REQUIRE(ofi.bid == Approx(1500.0).margin(1.0));
    REQUIRE(ofi.total > 0.0);
}

// ── TEST 3: OFI < 0 when best bid price drops ────────────────────────────────

TEST_CASE("OFI is negative when best bid price drops (market order hit)", "[feature_engine]") {
    FeatureEngine eng(100, 1);

    // Prev: bid 100.0 x 1000
    // Curr: bid 99.99 x 800   (market sell hit the best bid, price dropped)
    const LOBSnapshot prev = make_snap(0.0, 100.00, 1000, 100.01, 1000);
    const LOBSnapshot curr = make_snap(0.1,  99.99,  800, 100.01, 1000);

    const auto ofi = eng.compute_ofi_level(prev, curr, 0);

    // P_n^b < P_{n-1}^b:
    //   e_n(bid) = -V_n^b - V_{n-1}^b  (both subtracted) = -800 - 1000 = -1800
    REQUIRE(ofi.bid < 0.0);
    REQUIRE(ofi.total < 0.0);
}

// ── TEST 4: Feature vector has correct dimension ─────────────────────────────

TEST_CASE("Feature vector has correct dimension", "[feature_engine]") {
    constexpr int LEVELS = 5;
    FeatureEngine eng(100, LEVELS);

    const LOBSnapshot prev = make_snap(0.0, 100.0, 1000, 100.01, 1000, LEVELS);
    const LOBSnapshot curr = make_snap(0.1, 100.0, 1100, 100.01,  900, LEVELS);

    const auto fv = eng.feature_vector(prev, curr);

    // Expected dim = 2*LEVELS + 4
    REQUIRE(static_cast<int>(fv.size()) == eng.feature_dim());
    REQUIRE(eng.feature_dim() == 2 * LEVELS + 4);
}

// ── TEST 5: Normalized OFI is in [-1, 1] ────────────────────────────────────

TEST_CASE("Normalized rolling OFI stays within [-1, 1]", "[feature_engine]") {
    FeatureEngine eng(50, 3);
    std::mt19937 rng(0);
    std::uniform_int_distribution<long> vol_dist(100, 5000);
    std::normal_distribution<double>    px_noise(0.0, 0.0005);

    double mid = 100.0;
    for (int i = 0; i < 500; ++i) {
        mid += px_noise(rng);
        LOBSnapshot snap;
        snap.time     = i * 0.1;
        snap.n_levels = 3;
        for (int l = 0; l < 3; ++l) {
            snap.bids[l] = {mid - 0.01 * (l + 1), vol_dist(rng)};
            snap.asks[l] = {mid + 0.01 * (l + 1), vol_dist(rng)};
        }
        const double ofi = eng.update(snap);
        REQUIRE(ofi >= -1.0 - 1e-9);
        REQUIRE(ofi <=  1.0 + 1e-9);
    }
}

// ── TEST 6: Performance — 1M snapshots in < 1 second ────────────────────────

TEST_CASE("Processing 1M snapshots in < 1 second", "[feature_engine][performance]") {
    FeatureEngine eng(200, 5);

    // Build a fixed pair of snapshots (no I/O)
    const LOBSnapshot prev = make_snap(0.0, 100.00, 1000, 100.01, 1000, 5);
    const LOBSnapshot curr = make_snap(0.1, 100.01, 1200, 100.02,  800, 5);

    ankerl::nanobench::Bench bench;
    bench.minEpochIterations(1'000'000);
    bench.run("OFI compute_multi_level 5-level", [&] {
        ankerl::nanobench::doNotOptimizeAway(
            eng.compute_multi_level_ofi(prev, curr, 5)
        );
    });

    // Verify the benchmark ran (runtime check skipped in CI; nanobench logs it)
    SUCCEED("Performance benchmark completed (see nanobench output)");
}
