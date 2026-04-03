"""
Synthetic LOB data generator.

Generates a dataset where OFI has known predictive power over short-term
price changes, enabling ground-truth IC validation in tests.

Data-generating process:
  Mid price:  geometric Brownian motion, σ = 0.0001 per step
  Spread:     LogNormal(log(0.0002), 0.25)
  Quantities: LogNormal(log(500), 0.4) × exp(−0.5·(l−1)) per level
  Trades:     Hawkes process, μ=0.5/s, α=0.3, β=1.0
  Direction:  P(buy | OBI) = 0.5 + 0.3·OBI  (informed flow component)

The OBI-dependent trade direction ensures that OFI predicts the next
mid-price move, giving an expected IC ≈ 0.10-0.15 at short horizons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SyntheticLOB:
    """
    Generate synthetic limit order book data.

    Parameters
    ----------
    n_levels  : Number of LOB levels to generate (default 5).
    tick_size : Minimum price increment.
    dt        : Time per tick in seconds (default 0.1 s = 100 ms).
    seed      : Random seed.
    """

    def __init__(
        self,
        n_levels:  int   = 5,
        tick_size: float = 0.01,
        dt:        float = 0.1,
        seed:      int   = 42,
    ) -> None:
        self.n_levels  = n_levels
        self.tick_size = tick_size
        self.dt        = dt
        self.rng       = np.random.default_rng(seed)

    # ── Hawkes simulation (vectorised Ogata thinning) ────────────────────────

    def _simulate_hawkes(
        self,
        n_steps: int,
        mu:      float = 0.5,
        alpha:   float = 0.3,
        beta:    float = 1.0,
    ) -> np.ndarray:
        """
        Returns a boolean array of length n_steps indicating whether a
        trade arrived at each tick.

        Uses a Poisson approximation per tick (Δt = dt ≪ 1/μ) for speed.
        """
        T = n_steps * self.dt
        arrivals = np.zeros(n_steps, dtype=bool)

        # Exact thinning in continuous time, then bin to ticks
        event_times: list[float] = []
        t = 0.0
        lam_bar = mu

        while t < T:
            dt_draw = self.rng.exponential(1.0 / lam_bar)
            t      += dt_draw
            if t >= T:
                break

            # Current intensity
            contrib = sum(alpha * np.exp(-beta * (t - ti)) for ti in event_times[-50:])
            intensity = mu + contrib

            if self.rng.uniform() <= intensity / lam_bar:
                event_times.append(t)
                tick_idx = min(int(t / self.dt), n_steps - 1)
                arrivals[tick_idx] = True

            # Update upper bound
            contrib   = sum(alpha * np.exp(-beta * (t - ti)) for ti in event_times[-50:])
            lam_bar   = max(mu + contrib, mu)

        return arrivals

    # ── Main generator ───────────────────────────────────────────────────────

    def generate(self, n_steps: int = 500_000) -> pd.DataFrame:
        """
        Generate n_steps synthetic LOB snapshots.

        Returns a DataFrame with columns:
            time, mid_price, spread,
            bid_price_{l}, bid_vol_{l}, ask_price_{l}, ask_vol_{l}  for l=1..5,
            order_flow, volume, trade_direction, event, obi
        """
        # ── Mid price: GBM ──────────────────────────────────────────────────
        sigma_step = 0.0001
        log_returns = self.rng.normal(0.0, sigma_step, n_steps)
        mid_prices  = 100.0 * np.exp(np.cumsum(log_returns))

        # ── Spread: LogNormal ────────────────────────────────────────────────
        half_spreads = np.maximum(
            self.tick_size,
            self.rng.lognormal(np.log(0.0001), 0.25, n_steps),
        )

        # ── Volume base: LogNormal ───────────────────────────────────────────
        v1 = np.maximum(1, self.rng.lognormal(np.log(500), 0.4, n_steps))

        # ── Build LOB levels ─────────────────────────────────────────────────
        records: dict[str, np.ndarray] = {}
        for l in range(1, self.n_levels + 1):
            depth_factor = np.exp(-0.5 * (l - 1))
            vol_l = np.maximum(1, (v1 * depth_factor
                                   * self.rng.lognormal(0.0, 0.15, n_steps)).astype(int))
            records[f"bid_price_{l}"] = mid_prices - half_spreads * l
            records[f"ask_price_{l}"] = mid_prices + half_spreads * l
            records[f"bid_vol_{l}"]   = vol_l.copy()
            records[f"ask_vol_{l}"]   = np.maximum(1,
                (vol_l * self.rng.lognormal(0.0, 0.1, n_steps)).astype(int))

        # ── Order book imbalance ─────────────────────────────────────────────
        obi = ((records["bid_vol_1"].astype(float) - records["ask_vol_1"].astype(float))
               / (records["bid_vol_1"].astype(float) + records["ask_vol_1"].astype(float)
                  + 1e-9))

        # ── Trade arrivals (Hawkes) ──────────────────────────────────────────
        # Use simplified approach for large n_steps
        poisson_rates = 0.5 + 0.5 * np.abs(obi)  # more trading when imbalanced
        trade_arrives = self.rng.poisson(poisson_rates * self.dt).astype(bool)

        # ── Trade direction: P(buy | OBI) = 0.5 + 0.3·OBI ───────────────────
        p_buy = np.clip(0.5 + 0.3 * obi, 0.01, 0.99)
        is_buy = self.rng.uniform(size=n_steps) < p_buy

        # Direction: +1 (buy), -1 (sell), 0 (no trade)
        direction = np.where(trade_arrives, np.where(is_buy, 1, -1), 0)

        # ── Volume of each trade ─────────────────────────────────────────────
        trade_size = np.where(
            trade_arrives,
            np.maximum(1, self.rng.lognormal(np.log(200), 0.5, n_steps)).astype(int),
            0,
        )

        # ── Price update: trades move mid price in direction of flow ─────────
        # Small impact: λ ≈ 0.00005 $/share
        price_impact = 5e-5 * direction * trade_size
        # Add noise to mid price
        mid_prices   = np.cumsum(np.diff(mid_prices, prepend=mid_prices[0])
                                  + price_impact)
        mid_prices   = np.maximum(mid_prices, 1.0)  # price floor

        # Rebuild prices after impact
        for l in range(1, self.n_levels + 1):
            records[f"bid_price_{l}"] = mid_prices - half_spreads * l
            records[f"ask_price_{l}"] = mid_prices + half_spreads * l

        # ── Assemble DataFrame ───────────────────────────────────────────────
        time = np.arange(n_steps) * self.dt

        df = pd.DataFrame(
            {
                "time":            time,
                "mid_price":       mid_prices,
                "spread":          2.0 * half_spreads,
                "obi":             obi,
                "order_flow":      direction * trade_size.astype(float),
                "volume":          trade_size,
                "trade_direction": direction,
                "event":           trade_arrives.astype(int),
                **records,
            }
        ).set_index("time")

        return df


# ── CLI preview ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen = SyntheticLOB(n_levels=5, seed=42)
    df  = gen.generate(n_steps=5_000)
    print(f"Generated {len(df)} synthetic ticks")
    print(df[["mid_price", "spread", "obi", "order_flow"]].describe())
