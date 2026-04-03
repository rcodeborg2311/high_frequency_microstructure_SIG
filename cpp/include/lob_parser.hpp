#pragma once
/**
 * @file lob_parser.hpp
 * @brief Level-2 Order Book snapshot/event parser.
 *
 * Supports:
 *  - FI-2010 10-level LOB CSV format (Helsinki stock exchange, 5 stocks, 9 days)
 *  - LOBSTER orderbook + message format
 *
 * Reference: FI-2010 dataset, Ntakaris et al. (2018), "Benchmark dataset for
 * mid-price forecasting of limit order book data with machine learning methods".
 */

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace hfm {

/// Single price/volume pair at one LOB level.
struct Level {
    double price  = 0.0;  ///< Price in native units (divide FI-2010 integers by 10000).
    long   volume = 0;    ///< Resting quantity at this price level.
};

/// One complete LOB snapshot (up to 10 levels each side).
struct LOBSnapshot {
    double time    = 0.0;   ///< Seconds from midnight (exchange time).
    int    n_levels = 10;   ///< Number of populated levels.

    std::array<Level, 10> bids{};  ///< Bid levels; bids[0] = best bid.
    std::array<Level, 10> asks{};  ///< Ask levels; asks[0] = best ask.

    /// Mid-price = (best_bid + best_ask) / 2.
    [[nodiscard]] double mid_price() const noexcept {
        return (bids[0].price + asks[0].price) * 0.5;
    }

    /// Quoted spread = best_ask − best_bid.
    [[nodiscard]] double spread() const noexcept {
        return asks[0].price - bids[0].price;
    }

    /// Order-book imbalance at level 0 ∈ [−1, +1].
    [[nodiscard]] double obi() const noexcept {
        const double vb = static_cast<double>(bids[0].volume);
        const double va = static_cast<double>(asks[0].volume);
        const double denom = vb + va;
        return (denom > 0.0) ? (vb - va) / denom : 0.0;
    }
};

/// One LOBSTER message event (trades, submissions, cancellations).
struct LOBEvent {
    double  time       = 0.0;  ///< Event timestamp in seconds.
    int     event_type = 0;    ///< 1=submit, 2=cancel, 3=delete, 4=exec_visible, 5=exec_hidden
    int64_t order_id   = 0;
    long    size       = 0;
    double  price      = 0.0;
    int     direction  = 0;  ///< +1 = buy, −1 = sell
};

/**
 * @brief Parses LOB data from files or raw CSV lines.
 *
 * Usage:
 * @code
 *   LOBParser parser;
 *   auto snaps = parser.parse_fi2010("NOKIA_...orderbook_10.csv");
 * @endcode
 */
class LOBParser {
public:
    /**
     * @brief Parse an FI-2010 orderbook CSV file.
     *
     * FI-2010 column layout (1-indexed):
     *   1:  Time (seconds from midnight)
     *   2,3:  Ask1, AskVol1    (best ask price / volume, integer price units)
     *   4,5:  Bid1, BidVol1
     *   ...repeated for 10 levels...
     *
     * @param filepath  Path to the CSV file.
     * @param n_levels  Number of LOB levels to populate (default 10, max 10).
     * @return          Vector of snapshots in chronological order.
     */
    [[nodiscard]] std::vector<LOBSnapshot>
    parse_fi2010(const std::string& filepath, int n_levels = 10) const;

    /**
     * @brief Parse a LOBSTER orderbook file paired with a messages file.
     *
     * @param ob_file   Path to the *_orderbook_N.csv file.
     * @param msg_file  Path to the *_message_N.csv file.
     * @return          Pair of (snapshots, events) in chronological order.
     */
    [[nodiscard]] std::pair<std::vector<LOBSnapshot>, std::vector<LOBEvent>>
    parse_lobster(const std::string& ob_file, const std::string& msg_file) const;

    /**
     * @brief Parse a single FI-2010 CSV line into a snapshot.
     *
     * Useful for streaming / line-by-line processing without loading the
     * entire file into memory.
     *
     * @param line      Raw CSV line (comma-separated).
     * @param n_levels  Levels to populate.
     * @return          Populated snapshot, or std::nullopt on parse error.
     */
    [[nodiscard]] std::optional<LOBSnapshot>
    parse_fi2010_line(const std::string& line, int n_levels = 10) const;

private:
    /// Split a CSV line into tokens.
    [[nodiscard]] static std::vector<double>
    split_csv(const std::string& line);
};

}  // namespace hfm
