"""
Signal combination: IC-weighted, PCA, and Ridge regression methods.

References:
    Grinold & Kahn (2000), "Active Portfolio Management", 2nd ed., Chapter 6.
    Cont, Kukanov & Stoikov (2014), JFE 12(1):47-88.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class SignalCombiner:
    """
    Combine multiple alpha signals into a single composite signal.

    Three combination methods:
    ─────────────────────────────────────────────────────────────────
    1. ``ic_weighted`` (default):
          w_i = IC_i / Σ_j IC_j  (positive ICs only)
          combined = Σ w_i · Z(signal_i)   Z = z-score normalization

    2. ``pca``:
          Stack signals into matrix X (T × N_signals).
          Take the first principal component (PC1) as combined signal.
          PC1 maximizes variance across all signals simultaneously.

    3. ``ridge``:
          Regress forward_returns on [signal_1, ..., signal_N].
          Coefficients from Ridge(alpha=1.0) = signal weights.
          Out-of-sample IC computed on hold-out period (last 20%).
    ─────────────────────────────────────────────────────────────────

    Parameters
    ----------
    method : str
        One of ``'ic_weighted'``, ``'pca'``, ``'ridge'``.
    ic_horizon : int
        Forward return horizon (ticks) for IC computation.
    """

    def __init__(
        self,
        method:     str = "ic_weighted",
        ic_horizon: int = 10,
    ) -> None:
        allowed = ("ic_weighted", "pca", "ridge")
        if method not in allowed:
            raise ValueError(f"method must be one of {allowed}")
        self.method     = method
        self.ic_horizon = ic_horizon

        self._weights:  Optional[dict[str, float]] = None
        self._pca:      Optional[PCA]              = None
        self._scaler:   Optional[StandardScaler]   = None
        self._ridge:    Optional[Ridge]             = None
        self._col_order: list[str]                 = []

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, signals: pd.DataFrame, returns: pd.Series) -> "SignalCombiner":
        """
        Fit the combiner to in-sample data.

        Parameters
        ----------
        signals : DataFrame, columns = signal names, rows = ticks.
        returns : Forward returns series aligned with signals.

        Returns
        -------
        self (for chaining).
        """
        self._col_order = list(signals.columns)

        if self.method == "ic_weighted":
            self._fit_ic_weighted(signals, returns)
        elif self.method == "pca":
            self._fit_pca(signals)
        elif self.method == "ridge":
            self._fit_ridge(signals, returns)

        return self

    def _fit_ic_weighted(self, signals: pd.DataFrame, returns: pd.Series) -> None:
        """IC-weighted combination (positive ICs only)."""
        ics: dict[str, float] = {}
        for col in signals.columns:
            sig  = signals[col].dropna()
            ret  = returns.reindex(sig.index).dropna()
            common = sig.index.intersection(ret.index)
            if len(common) < 10:
                continue
            ic, _ = pearsonr(sig.loc[common], ret.loc[common])
            ics[col] = ic

        # Keep only positive-IC signals
        pos_ics = {k: v for k, v in ics.items() if v > 0}
        if not pos_ics:
            # Fall back to equal weights if all ICs are negative
            pos_ics = {k: max(v, 1e-6) for k, v in ics.items()}

        total = sum(pos_ics.values())
        self._weights = {k: v / total for k, v in pos_ics.items()}

    def _fit_pca(self, signals: pd.DataFrame) -> None:
        """PCA combination: take the first principal component."""
        clean = signals.dropna()
        if len(clean) < 2:
            raise ValueError("Not enough non-NaN rows for PCA")
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(clean)
        self._pca = PCA(n_components=1)
        self._pca.fit(X_scaled)

    def _fit_ridge(self, signals: pd.DataFrame, returns: pd.Series) -> None:
        """Ridge regression combination."""
        clean = signals.dropna()
        ret   = returns.reindex(clean.index).dropna()
        common = clean.index.intersection(ret.index)
        X = clean.loc[common]
        y = ret.loc[common]
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._ridge = Ridge(alpha=1.0)
        self._ridge.fit(X_scaled, y.values)
        self._weights = dict(zip(signals.columns,
                                  self._ridge.coef_.tolist()))

    # ── Transform ────────────────────────────────────────────────────────────

    def transform(self, signals: pd.DataFrame) -> pd.Series:
        """
        Apply the fitted combiner to new signal data.

        Parameters
        ----------
        signals : DataFrame with the same columns used during fit().

        Returns
        -------
        pd.Series : Combined signal, same index as signals.
        """
        if not self._col_order:
            raise RuntimeError("Call fit() before transform()")

        if self.method == "pca":
            return self._transform_pca(signals)
        else:
            return self._transform_weighted(signals)

    def _transform_weighted(self, signals: pd.DataFrame) -> pd.Series:
        """Weighted combination of z-scored signals."""
        weights = self._weights or {}
        result  = pd.Series(0.0, index=signals.index)

        for col, w in weights.items():
            if col not in signals.columns:
                continue
            sig = signals[col].copy()
            # Z-score normalize
            mu  = sig.mean()
            std = sig.std()
            if std > 1e-10:
                sig = (sig - mu) / std
            result = result + w * sig.fillna(0.0)

        return result.rename("combined_signal")

    def _transform_pca(self, signals: pd.DataFrame) -> pd.Series:
        """Project signals onto PC1."""
        assert self._pca is not None and self._scaler is not None
        cols  = [c for c in self._col_order if c in signals.columns]
        clean = signals[cols].fillna(0.0)
        X_scaled = self._scaler.transform(clean)
        pc1 = self._pca.transform(X_scaled)[:, 0]
        return pd.Series(pc1, index=signals.index, name="combined_pca")

    # ── Evaluate ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        signals: pd.DataFrame,
        returns: pd.Series,
    ) -> dict:
        """
        Evaluate combined signal performance.

        Returns
        -------
        dict with keys:
            ic          – Pearson IC of combined vs forward returns.
            icir        – IC / std(IC) (rolling over 50-tick windows).
            sharpe      – Annualised Sharpe of the combined signal PnL.
            improvement – IC improvement over best single signal.
        """
        combined = self.transform(signals)

        # IC
        common = combined.dropna().index.intersection(returns.dropna().index)
        if len(common) < 10:
            return {"ic": np.nan, "icir": np.nan,
                    "sharpe": np.nan, "improvement": np.nan}

        ic_all, _ = pearsonr(combined.loc[common], returns.loc[common])

        # Rolling IC for ICIR
        roll_ic: list[float] = []
        W = 50
        idx_arr = np.array([combined.index.get_loc(i) for i in common])
        for i in range(W, len(common)):
            window_c = combined.iloc[idx_arr[i - W : i]]
            window_r = returns.reindex(combined.index[idx_arr[i - W : i]]).dropna()
            wc = window_c.reindex(window_r.index).dropna()
            wr = window_r.reindex(wc.index)
            if len(wc) >= 5:
                ic, _ = pearsonr(wc, wr)
                roll_ic.append(ic)

        icir = (np.mean(roll_ic) / np.std(roll_ic)
                if len(roll_ic) >= 2 and np.std(roll_ic) > 1e-10
                else np.nan)

        # Sharpe of combined signal (treated as daily PnL = signal * next_return)
        pnl = combined.loc[common] * returns.loc[common]
        sharpe = (pnl.mean() / pnl.std() * np.sqrt(252 * 390 * 6)
                  if pnl.std() > 1e-10 else np.nan)

        # Best single signal IC
        best_single_ic = 0.0
        for col in signals.columns:
            if col in signals:
                sg = signals[col].dropna()
                rt = returns.reindex(sg.index).dropna()
                c  = sg.index.intersection(rt.index)
                if len(c) >= 10:
                    ic_s, _ = pearsonr(sg.loc[c], rt.loc[c])
                    best_single_ic = max(best_single_ic, abs(ic_s))

        improvement = abs(ic_all) - best_single_ic

        return {
            "ic":          float(ic_all),
            "icir":        float(icir) if not np.isnan(icir) else np.nan,
            "sharpe":      float(sharpe) if not np.isnan(sharpe) else np.nan,
            "improvement": float(improvement),
        }
