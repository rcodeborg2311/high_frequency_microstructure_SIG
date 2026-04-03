/**
 * @file lob_parser.cpp
 * @brief LOB snapshot/event parser implementation.
 */

#include "lob_parser.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace hfm {

// ── Helper ─────────────────────────────────────────────────────────────────

std::vector<double> LOBParser::split_csv(const std::string& line) {
    std::vector<double> result;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            result.push_back(std::stod(token));
        }
    }
    return result;
}

// ── FI-2010 ────────────────────────────────────────────────────────────────

std::optional<LOBSnapshot>
LOBParser::parse_fi2010_line(const std::string& line, int n_levels) const {
    const auto vals = split_csv(line);
    // Minimum columns: 1 (time) + 4*n_levels (ask/askvol/bid/bidvol per level)
    const int required = 1 + 4 * n_levels;
    if (static_cast<int>(vals.size()) < required) {
        return std::nullopt;
    }

    LOBSnapshot snap;
    snap.time     = vals[0];
    snap.n_levels = n_levels;

    // FI-2010 layout: time, Ask1, AskVol1, Bid1, BidVol1, Ask2, AskVol2, ...
    // Prices are in integer units; divide by 10000 for actual price.
    constexpr double PRICE_SCALE = 1.0 / 10000.0;
    for (int l = 0; l < n_levels; ++l) {
        const int base = 1 + 4 * l;
        snap.asks[l].price  = vals[base + 0] * PRICE_SCALE;
        snap.asks[l].volume = static_cast<long>(vals[base + 1]);
        snap.bids[l].price  = vals[base + 2] * PRICE_SCALE;
        snap.bids[l].volume = static_cast<long>(vals[base + 3]);
    }

    return snap;
}

std::vector<LOBSnapshot>
LOBParser::parse_fi2010(const std::string& filepath, int n_levels) const {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("LOBParser: cannot open file: " + filepath);
    }

    std::vector<LOBSnapshot> result;
    result.reserve(100000);

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto snap = parse_fi2010_line(line, n_levels);
        if (snap.has_value()) {
            result.push_back(std::move(*snap));
        }
    }

    return result;
}

// ── LOBSTER ────────────────────────────────────────────────────────────────

std::pair<std::vector<LOBSnapshot>, std::vector<LOBEvent>>
LOBParser::parse_lobster(const std::string& ob_file,
                          const std::string& msg_file) const {
    // Parse orderbook file
    std::vector<LOBSnapshot> snapshots;
    {
        std::ifstream file(ob_file);
        if (!file.is_open()) {
            throw std::runtime_error("LOBParser: cannot open orderbook file: " + ob_file);
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            const auto vals = split_csv(line);
            // LOBSTER orderbook: Ask1, AskVol1, Bid1, BidVol1, Ask2, ...
            // No time column — time comes from message file.
            constexpr int LEVELS = 10;
            if (static_cast<int>(vals.size()) < 4 * LEVELS) continue;

            LOBSnapshot snap;
            snap.n_levels = LEVELS;
            constexpr double PRICE_SCALE = 1.0 / 10000.0;
            for (int l = 0; l < LEVELS; ++l) {
                const int base = 4 * l;
                snap.asks[l].price  = vals[base + 0] * PRICE_SCALE;
                snap.asks[l].volume = static_cast<long>(vals[base + 1]);
                snap.bids[l].price  = vals[base + 2] * PRICE_SCALE;
                snap.bids[l].volume = static_cast<long>(vals[base + 3]);
            }
            snapshots.push_back(snap);
        }
    }

    // Parse message file and assign timestamps
    std::vector<LOBEvent> events;
    {
        std::ifstream file(msg_file);
        if (!file.is_open()) {
            throw std::runtime_error("LOBParser: cannot open message file: " + msg_file);
        }
        std::string line;
        std::size_t snap_idx = 0;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            const auto vals = split_csv(line);
            if (vals.size() < 6) continue;

            LOBEvent ev;
            ev.time       = vals[0];
            ev.event_type = static_cast<int>(vals[1]);
            ev.order_id   = static_cast<int64_t>(vals[2]);
            ev.size       = static_cast<long>(vals[3]);
            ev.price      = vals[4] / 10000.0;
            ev.direction  = static_cast<int>(vals[5]);
            events.push_back(ev);

            // Assign time to the corresponding snapshot
            if (snap_idx < snapshots.size()) {
                snapshots[snap_idx].time = ev.time;
                ++snap_idx;
            }
        }
    }

    return {std::move(snapshots), std::move(events)};
}

}  // namespace hfm
