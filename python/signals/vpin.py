"""
VPIN (Volume-Synchronized Probability of Informed Trading).

Reference:
    Easley, López de Prado, O'Hara (2012), "Flow Toxicity and Liquidity
    in a High-Frequency World", Review of Financial Studies 25(5):1457-1493.

Bulk volume classification (eq. 11, p. 1467):
    V_buy(bucket) = V_bucket · Φ(ΔP_bucket / σ_ΔP)
    V_sell(bucket) = V_bucket − V_buy(bucket)

VPIN (eq. 12, p. 1468):
    VPIN = (1/n) · Σ_{τ=t-n+1}^{t} |V_buy(τ) − V_sell(τ)| / V_bucket

where n = 50 equal-volume buckets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .ofi import BaseSignal


class VPINSignal(BaseSignal):
    """
    VPIN computed over rolling equal-volume buckets.

    Parameters
    ----------
    bucket_size : int
        Target volume per bucket (calibrate to daily_volume / 50).
    n_buckets   : int
        Number of trailing buckets for the VPIN estimate (default 50).

    References
    ----------
    Easley, López de Prado & O'Hara (2012), RFS 25(5):1457-1493.
    """

    def __init__(self, bucket_size: int = 10_000, n_buckets: int = 50) -> None:
        self.bucket_size = bucket_size
        self.n_buckets   = n_buckets

    def name(self) -> str:
        return f"VPIN_bs{self.bucket_size}_nb{self.n_buckets}"

    # ── Core bucket filling ──────────────────────────────────────────────────

    def _fill_buckets(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        sigma_dp: float,
    ) -> list[float]:
        """
        Fill equal-volume buckets and compute |V_buy − V_sell| / V_bucket
        for each completed bucket.

        Implements the bulk volume classification from Easley et al. (2012).
        """
        buckets: list[float] = []
        current_vol  = 0.0
        current_dp   = 0.0   # cumulative ΔP within current bucket

        n = len(prices)
        for i in range(1, n):
            vol = float(volumes[i])
            dp  = float(prices[i] - prices[i - 1])

            # Distribute vol into the current bucket
            while vol > 1e-9:
                space = self.bucket_size - current_vol
                fill  = min(vol, space)

                # Weight the ΔP contribution by the volume fraction
                current_dp  += dp * (fill / self.bucket_size)
                current_vol += fill
                vol         -= fill

                # Bucket complete
                if current_vol >= self.bucket_size - 1e-9:
                    # Bulk volume classification (Easley et al. 2012, eq. 11)
                    z_score = current_dp / sigma_dp if sigma_dp > 1e-12 else 0.0
                    v_buy   = self.bucket_size * float(norm.cdf(z_score))
                    v_sell  = self.bucket_size - v_buy
                    toxicity = abs(v_buy - v_sell) / self.bucket_size
                    buckets.append(toxicity)
                    current_vol = 0.0
                    current_dp  = 0.0

        return buckets

    # ── Public signal computation ────────────────────────────────────────────

    def compute_from_arrays(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
    ) -> np.ndarray:
        """
        Compute VPIN at bucket resolution.

        Parameters
        ----------
        prices  : 1-D array of mid-prices (or trade prices).
        volumes : 1-D array of per-tick volumes.

        Returns
        -------
        np.ndarray : VPIN value for each completed bucket.
        """
        dp = np.diff(prices)
        if len(dp) == 0:
            return np.array([])

        sigma_dp = float(np.std(dp)) if np.std(dp) > 0 else 1e-6
        buckets  = self._fill_buckets(prices, volumes, sigma_dp)

        if not buckets:
            return np.array([])

        # Rolling mean of |V_buy − V_sell| / V_bucket over n_buckets
        bucket_arr = np.array(buckets)
        vpin_vals  = np.full(len(bucket_arr), np.nan)
        for i in range(len(bucket_arr)):
            start           = max(0, i - self.n_buckets + 1)
            vpin_vals[i]    = np.mean(bucket_arr[start : i + 1])

        return vpin_vals

    def compute(self, snapshot: pd.DataFrame) -> pd.Series:
        """
        Compute tick-level VPIN by forward-filling bucket completions.

        Parameters
        ----------
        snapshot : DataFrame with columns 'mid_price' and 'volume',
                   time-indexed.

        Returns
        -------
        pd.Series : VPIN ∈ [0, 1], same index as snapshot.
        """
        prices  = snapshot["mid_price"].values
        volumes = snapshot["volume"].values if "volume" in snapshot.columns \
                  else np.ones(len(snapshot)) * self.bucket_size

        dp       = np.diff(prices)
        sigma_dp = float(np.std(dp)) if len(dp) > 1 else 1e-6

        # Track bucket completions and their tick positions
        n = len(prices)
        result   = np.full(n, np.nan)

        current_vol = 0.0
        current_dp  = 0.0
        bucket_list: list[float] = []
        vpin_current = 0.5  # default

        for i in range(1, n):
            vol = float(volumes[i])
            dp_i = float(prices[i] - prices[i - 1])

            while vol > 1e-9:
                space = self.bucket_size - current_vol
                fill  = min(vol, space)
                current_dp  += dp_i * (fill / self.bucket_size)
                current_vol += fill
                vol         -= fill

                if current_vol >= self.bucket_size - 1e-9:
                    z      = current_dp / sigma_dp if sigma_dp > 1e-12 else 0.0
                    v_buy  = self.bucket_size * float(norm.cdf(z))
                    toxicity = abs(v_buy - (self.bucket_size - v_buy)) / self.bucket_size
                    bucket_list.append(toxicity)
                    start   = max(0, len(bucket_list) - self.n_buckets)
                    vpin_current = float(np.mean(bucket_list[start:]))
                    current_vol = 0.0
                    current_dp  = 0.0

            result[i] = vpin_current

        result[0] = result[1] if n > 1 else 0.5

        return pd.Series(result, index=snapshot.index, name=self.name())
