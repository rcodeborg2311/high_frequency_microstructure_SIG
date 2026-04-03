"""
Order Flow Imbalance (OFI) signal.

Reference:
    Cont, Kukanov, Stoikov (2014), "The Price Impact of Order Book Events",
    Journal of Financial Econometrics 12(1):47-88.

OFI event definitions (eqs. 2-3, p. 51):
    e_n(bid) = V_n^b · 1{P_n^b ≥ P_{n-1}^b}
             − V_n^b · 1{P_n^b < P_{n-1}^b}
             − V_{n-1}^b · 1{P_n^b ≤ P_{n-1}^b}

    e_n(ask) = −V_n^a · 1{P_n^a ≤ P_{n-1}^a}
             + V_n^a · 1{P_n^a > P_{n-1}^a}
             + V_{n-1}^a · 1{P_n^a ≥ P_{n-1}^a}

    OFI_n = e_n(bid) + e_n(ask)

Normalized rolling OFI (ŌFIP):
    ŌFIP(W) = Σ_{n=t-W+1}^{t} OFI_n / (Σ |OFI_n| + ε)  ∈ [−1, +1]
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseSignal(ABC):
    """Abstract base class for all alpha signals."""

    @abstractmethod
    def compute(self, snapshot: pd.DataFrame) -> pd.Series:
        """
        Compute the signal from a LOB snapshot DataFrame.

        Parameters
        ----------
        snapshot : DataFrame with columns bid_price_{l}, bid_vol_{l},
                   ask_price_{l}, ask_vol_{l} for each level l, indexed by time.

        Returns
        -------
        pd.Series : Time-indexed signal values.
        """

    @abstractmethod
    def name(self) -> str:
        """Signal identifier."""


class OFISignal(BaseSignal):
    """
    Multi-level Order Flow Imbalance signal.

    Parameters
    ----------
    levels : int
        Number of LOB levels to aggregate (default 5).
    window : int
        Rolling normalization window in ticks (default 100).

    References
    ----------
    Cont, Kukanov & Stoikov (2014), JFE 12(1):47-88, eqs. 2-3, p. 51.
    """

    def __init__(self, levels: int = 5, window: int = 100) -> None:
        self.levels = levels
        self.window = window
        self._EPS   = 1e-9

    def name(self) -> str:
        return f"OFI_L{self.levels}_W{self.window}"

    # ── Low-level per-level computation ─────────────────────────────────────

    @staticmethod
    def _ofi_event(
        pb: np.ndarray, vb: np.ndarray,   # prev bid price/vol
        cb: np.ndarray, cvb: np.ndarray,  # curr bid price/vol
        pa: np.ndarray, va: np.ndarray,   # prev ask price/vol
        ca: np.ndarray, cva: np.ndarray,  # curr ask price/vol
    ) -> np.ndarray:
        """
        Vectorized OFI event computation for one LOB level.

        Implements Cont et al. (2014), eqs. 2-3, p. 51.

        Returns
        -------
        np.ndarray : OFI values, one per row.
        """
        # Bid component:
        #   +cvb  when cb >= pb (bid price rose or held — positive flow)
        #   −cvb  when cb <  pb (bid price fell — subtract current volume)
        #   −vb   when cb <= pb (prev volume removed when price fell/held)
        bid_up   = np.where(cb >= pb, cvb, 0.0)
        bid_down = np.where(cb <  pb, cvb, 0.0)
        bid_gone = np.where(cb <= pb,  vb, 0.0)
        e_bid    = bid_up - bid_down - bid_gone

        # Ask component:
        #   −cva  when ca <= pa (ask price fell — negative flow)
        #   +cva  when ca >  pa (ask price rose — positive flow)
        #   +va   when ca >= pa (prev ask volume still there)
        ask_down = np.where(ca <= pa, cva, 0.0)
        ask_up   = np.where(ca >  pa, cva, 0.0)
        ask_gone = np.where(ca >= pa,  va, 0.0)
        e_ask    = -ask_down + ask_up + ask_gone

        return e_bid + e_ask

    def compute(self, snapshot: pd.DataFrame) -> pd.Series:
        """
        Compute normalized rolling OFI from a LOB snapshot DataFrame.

        Parameters
        ----------
        snapshot : DataFrame indexed by time with columns:
            bid_price_{l}, bid_vol_{l}, ask_price_{l}, ask_vol_{l}
            for l ∈ {1, ..., levels}.

        Returns
        -------
        pd.Series : ŌFIP(W) ∈ [−1, +1], same index as snapshot.
        """
        df    = snapshot
        n     = len(df)
        total_ofi = np.zeros(n)

        for l in range(1, self.levels + 1):
            bp_col  = f"bid_price_{l}"
            bv_col  = f"bid_vol_{l}"
            ap_col  = f"ask_price_{l}"
            av_col  = f"ask_vol_{l}"

            # Fall back gracefully if higher levels are missing
            if bp_col not in df.columns:
                break

            pb  = df[bp_col].values
            vb  = df[bv_col].values.astype(float)
            pa  = df[ap_col].values
            va  = df[av_col].values.astype(float)

            # Lagged values (prev snapshot)
            pb_prev = np.roll(pb, 1); pb_prev[0] = pb[0]
            vb_prev = np.roll(vb, 1); vb_prev[0] = vb[0]
            pa_prev = np.roll(pa, 1); pa_prev[0] = pa[0]
            va_prev = np.roll(va, 1); va_prev[0] = va[0]

            ofi_l = self._ofi_event(pb_prev, vb_prev, pb, vb,
                                     pa_prev, va_prev, pa, va)
            # First tick has no prev → set to 0
            ofi_l[0] = 0.0
            total_ofi += ofi_l

        # Rolling normalization: ŌFIP(W) = Σ OFI / (Σ |OFI| + ε)
        ofi_s = pd.Series(total_ofi, index=df.index)
        roll_sum     = ofi_s.rolling(window=self.window, min_periods=1).sum()
        roll_abs_sum = ofi_s.abs().rolling(window=self.window, min_periods=1).sum()

        normalized = roll_sum / (roll_abs_sum + self._EPS)
        return normalized.clip(-1.0, 1.0).rename(self.name())

    # ── Convenience: price impact regression ────────────────────────────────

    def price_impact_beta(
        self,
        ofi: pd.Series,
        mid_prices: pd.Series,
        horizon: int = 10,
    ) -> dict:
        """
        OLS regression: ΔP_{mid} = α + β·ŌFIP + ε.

        Cont et al. (2014) report R² of 5-15% at 1-minute horizons.

        Parameters
        ----------
        ofi        : Normalized OFI series (output of compute()).
        mid_prices : Mid-price series aligned with ofi.
        horizon    : Forward return horizon in ticks.

        Returns
        -------
        dict with keys: alpha, beta, r_squared, n_obs.
        """
        fwd_ret = mid_prices.shift(-horizon) - mid_prices
        common  = ofi.dropna().index.intersection(fwd_ret.dropna().index)
        x = ofi.loc[common].values
        y = fwd_ret.loc[common].values

        if len(x) < 10:
            return {"alpha": float("nan"), "beta": float("nan"),
                    "r_squared": float("nan"), "n_obs": 0}

        # OLS via normal equations
        x_dm  = x - x.mean()
        b     = (x_dm * y).sum() / (x_dm ** 2).sum()
        a     = y.mean() - b * x.mean()

        y_hat = a + b * x
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2    = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {"alpha": float(a), "beta": float(b),
                "r_squared": float(r2), "n_obs": len(x)}
