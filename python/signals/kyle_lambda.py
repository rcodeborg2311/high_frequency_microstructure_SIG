"""
Kyle's lambda — rolling price impact coefficient.

Reference:
    Kyle (1985), "Continuous Auctions and Insider Trading",
    Econometrica 53(6):1315-1335.

Linear price impact model (Kyle 1985, eq. 5, p. 1319):
    ΔP = λ · q + ε

where q is signed order flow (positive = net buying) and ΔP is the
price change.

Closed-form OLS estimator (rolling window of T observations):
    λ̂ = Σ(q_t · ΔP_t) / Σ(q_t²)

Intraday pattern: λ is high at open, falls at midday, rises at close (U-shaped).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ofi import BaseSignal


class KyleLambdaSignal(BaseSignal):
    """
    Rolling Kyle's lambda via closed-form OLS on (order_flow, price_change) pairs.

    Parameters
    ----------
    window : int
        Rolling window size in ticks (default 100).

    References
    ----------
    Kyle (1985), Econometrica 53(6):1315-1335, eq. 5, p. 1319.
    """

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self._EPS   = 1e-12

    def name(self) -> str:
        return f"KyleLambda_W{self.window}"

    # ── Core computation ─────────────────────────────────────────────────────

    def compute_from_arrays(
        self,
        order_flows: np.ndarray,
        price_changes: np.ndarray,
    ) -> np.ndarray:
        """
        Rolling Kyle's lambda via cumulative-sum O(N) trick.

        λ̂(t) = Σ_{s=t-W+1}^{t} q_s·ΔP_s  /  Σ_{s=t-W+1}^{t} q_s²

        Parameters
        ----------
        order_flows   : 1-D array of signed order flow (positive = net buy).
        price_changes : 1-D array of mid-price changes, aligned with order_flows.

        Returns
        -------
        np.ndarray : Rolling λ̂, NaN for first (window-1) ticks.
        """
        n   = len(order_flows)
        qp  = order_flows * price_changes
        q2  = order_flows ** 2

        # Rolling sums using cumulative sums (O(N) amortized)
        cum_qp = np.cumsum(qp)
        cum_q2 = np.cumsum(q2)

        # Window sums: sum[t-W+1..t] = cum[t] - cum[t-W]
        pad = np.zeros(self.window)

        roll_qp = cum_qp - np.concatenate([np.zeros(self.window), cum_qp[:-self.window]])
        roll_q2 = cum_q2 - np.concatenate([np.zeros(self.window), cum_q2[:-self.window]])

        # For the first (window-1) observations, just use the cumulative sum
        for i in range(min(self.window - 1, n)):
            roll_qp[i] = cum_qp[i]
            roll_q2[i] = cum_q2[i]

        lambda_vals = np.where(roll_q2 > self._EPS, roll_qp / roll_q2, np.nan)
        lambda_vals[: self.window - 1] = np.nan  # insufficient history

        return lambda_vals

    def compute(self, snapshot: pd.DataFrame) -> pd.Series:
        """
        Compute rolling Kyle's lambda from a LOB snapshot DataFrame.

        Parameters
        ----------
        snapshot : DataFrame with columns 'mid_price' and 'order_flow',
                   time-indexed.

                   If 'order_flow' is absent, it is approximated as the
                   product of trade direction (+1/-1) and volume, or as
                   the signed mid-price change when neither is available.

        Returns
        -------
        pd.Series : Rolling λ̂ (units: $/share), same index as snapshot.
        """
        prices = snapshot["mid_price"].values
        dp     = np.diff(prices, prepend=prices[0])  # ΔP at each tick

        # Determine order flow proxy
        if "order_flow" in snapshot.columns:
            q = snapshot["order_flow"].values.astype(float)
        elif "trade_direction" in snapshot.columns and "volume" in snapshot.columns:
            q = (snapshot["trade_direction"].values.astype(float)
                 * snapshot["volume"].values.astype(float))
        else:
            # Fallback: use signed mid-price change as order flow proxy
            q = np.sign(dp)

        lambda_vals = self.compute_from_arrays(q, dp)

        return pd.Series(lambda_vals, index=snapshot.index, name=self.name())

    # ── Intraday diagnostics ─────────────────────────────────────────────────

    def intraday_lambda(
        self,
        lambda_series: pd.Series,
        bins: int = 13,
    ) -> pd.Series:
        """
        Average λ by time-of-day bin.

        Expects lambda_series to have a DatetimeIndex.
        Returns average λ per bin, suitable for U-shape visualization.

        Parameters
        ----------
        lambda_series : Output of compute(), DatetimeIndex required.
        bins          : Number of equal-length time bins in the trading day.

        Returns
        -------
        pd.Series : Mean λ per time bin, indexed by bin start time.
        """
        if not isinstance(lambda_series.index, pd.DatetimeIndex):
            return lambda_series.groupby(
                pd.cut(np.arange(len(lambda_series)), bins=bins)
            ).mean()

        seconds = (lambda_series.index.hour * 3600
                   + lambda_series.index.minute * 60
                   + lambda_series.index.second)
        bin_edges = np.linspace(seconds.min(), seconds.max(), bins + 1)
        bins_arr  = np.digitize(seconds, bin_edges[:-1]) - 1

        result = {}
        for b in range(bins):
            mask = bins_arr == b
            vals = lambda_series[mask].dropna()
            if len(vals) > 0:
                result[b] = float(vals.mean())

        return pd.Series(result)
