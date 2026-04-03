"""
Hawkes process order-arrival model (Python, for research/validation).

Reference:
    Bacry, Mastromatteo, Muzy (2015), "Hawkes Processes in Finance",
    Market Microstructure and Liquidity 1(1):1550005.

Intensity (eq. 1, p. 4):
    λ(t) = μ + α · Σ_{t_i < t} exp(−β(t − t_i))

Log-likelihood (eq. 5, p. 8):
    L(μ,α,β) = −μT − (α/β)·Σᵢ[1 − exp(−β(T−tᵢ))] + Σᵢ log(μ + α·Rᵢ)

Recursive O(N) Rᵢ (eq. 6, p. 8):
    R₁ = 0
    Rᵢ = (Rᵢ₋₁ + 1) · exp(−β · (tᵢ − tᵢ₋₁))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kstest

from .ofi import BaseSignal


@dataclass
class HawkesParams:
    """Fitted Hawkes process parameters."""
    mu:    float  # baseline intensity (events/sec)
    alpha: float  # excitation amplitude
    beta:  float  # decay rate (1/sec)

    @property
    def branching_ratio(self) -> float:
        """α/β — fraction of events that are self-excited (Bacry et al. 2015)."""
        return self.alpha / self.beta if self.beta > 0 else float("inf")

    @property
    def stationary(self) -> bool:
        """True if branching ratio < 1 (process doesn't explode)."""
        return self.branching_ratio < 1.0

    @property
    def excitation_halflife(self) -> float:
        """log(2)/β — half-life of excitation effect in seconds."""
        return np.log(2.0) / self.beta if self.beta > 0 else float("inf")


@dataclass
class HawkesResult:
    """Full Hawkes MLE fit result."""
    params:          HawkesParams
    log_likelihood:  float
    n_events:        int
    T:               float
    converged:       bool
    ks_statistic:    float


class HawkesProcess:
    """
    Maximum-likelihood estimator and simulator for a univariate Hawkes process.

    Optimization via L-BFGS-B with log-transformed parameters to enforce
    positivity, plus stationarity penalty.
    """

    # ── O(N) Recursive R ────────────────────────────────────────────────────

    @staticmethod
    def compute_R(times: np.ndarray, beta: float) -> np.ndarray:
        """
        O(N) recursive computation of Rᵢ.

        Implements Bacry et al. (2015), eq. 6, p. 8:
            R₁ = 0
            Rᵢ = (Rᵢ₋₁ + 1) · exp(−β · (tᵢ − tᵢ₋₁))

        Parameters
        ----------
        times : Sorted event arrival times (seconds from start).
        beta  : Decay rate parameter.

        Returns
        -------
        np.ndarray : R values of same length as times.
        """
        N = len(times)
        R = np.zeros(N)
        for i in range(1, N):
            dt   = times[i] - times[i - 1]
            R[i] = (R[i - 1] + 1.0) * np.exp(-beta * dt)
        return R

    # ── Log-likelihood ───────────────────────────────────────────────────────

    @classmethod
    def log_likelihood(
        cls,
        params: HawkesParams,
        times:  np.ndarray,
        T:      float,
    ) -> float:
        """
        Evaluate log-likelihood at given parameters via O(N) recursion.

        Implements Bacry et al. (2015), eq. 5, p. 8:
            L = −μT − (α/β)·Σ[1 − exp(−β(T−tᵢ))] + Σ log(μ + α·Rᵢ)

        Parameters
        ----------
        params : Hawkes parameters.
        times  : Sorted event times.
        T      : Observation window length.

        Returns
        -------
        float : Log-likelihood (higher = better).
        """
        mu, alpha, beta = params.mu, params.alpha, params.beta
        if mu <= 0 or alpha <= 0 or beta <= 0:
            return -np.inf

        R = cls.compute_R(times, beta)
        N = len(times)

        # Term 1: −μT
        ll = -mu * T

        # Term 2: −(α/β)·Σᵢ[1 − exp(−β(T−tᵢ))]
        ll -= (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - times)))

        # Term 3: Σᵢ log(μ + α·Rᵢ)
        intensities = mu + alpha * R
        if np.any(intensities <= 0):
            return -np.inf
        ll += np.sum(np.log(intensities))

        return float(ll)

    # ── Conditional intensity ────────────────────────────────────────────────

    @classmethod
    def intensity_at(
        cls,
        params: HawkesParams,
        times:  np.ndarray,
        t:      float,
    ) -> float:
        """λ(t) = μ + α · Σ_{t_i < t} exp(−β(t − t_i))."""
        contrib = np.sum(np.exp(-params.beta * (t - times[times < t])))
        return params.mu + params.alpha * contrib

    # ── KS test ──────────────────────────────────────────────────────────────

    @classmethod
    def ks_test(
        cls,
        params: HawkesParams,
        times:  np.ndarray,
        T:      float,
    ) -> float:
        """
        KS test on the compensated residual process.

        Inter-event compensated intervals Λ(t_{i+1}) − Λ(t_i) should be Exp(1)
        under the fitted model. Returns the KS statistic.
        """
        N = len(times)
        if N < 5:
            return 1.0

        # Integrated intensity at each event: Λ(t_i) ≈ μ·t_i + (α/β)·Σ_{j≤i}(1−e^{−β(t_i−t_j)})
        R = cls.compute_R(times, params.beta)
        # Build cumulative compensator efficiently
        compensator = np.zeros(N)
        for i in range(N):
            comp = params.mu * times[i]
            # (α/β) · Σ_{j≤i} (1 − exp(−β(t_i − t_j)))
            past = times[: i + 1]
            comp += (params.alpha / params.beta) * np.sum(
                1.0 - np.exp(-params.beta * (times[i] - past))
            )
            compensator[i] = comp

        gaps = np.diff(compensator)
        gaps = gaps[gaps > 0]

        # KS test against Exp(1): CDF = 1 − exp(−x)
        stat, _ = kstest(gaps, "expon", args=(0, 1))
        return float(stat)

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(
        self,
        times: np.ndarray,
        T:     float,
        initial: Optional[HawkesParams] = None,
    ) -> HawkesResult:
        """
        Fit Hawkes process via L-BFGS-B on log-transformed parameters.

        Parameters
        ----------
        times   : Sorted event arrival times.
        T       : Observation window length (seconds).
        initial : Starting guess (default: μ=0.5, α=0.3, β=1.0).

        Returns
        -------
        HawkesResult with fitted parameters and diagnostics.
        """
        if len(times) < 3:
            raise ValueError("Need at least 3 events to fit Hawkes model")

        if initial is None:
            initial = HawkesParams(0.5, 0.3, 1.0)

        # Work in log-parameter space to enforce positivity
        x0 = np.array([np.log(initial.mu),
                        np.log(initial.alpha),
                        np.log(initial.beta)])

        def neg_ll(x: np.ndarray) -> float:
            mu, alpha, beta = np.exp(x)
            p = HawkesParams(mu, alpha, beta)
            if not p.stationary:
                # Soft stationarity penalty
                return 1e6 * (p.branching_ratio - 1.0) + 1e6
            ll = self.log_likelihood(p, times, T)
            return -ll if np.isfinite(ll) else 1e10

        result = minimize(
            neg_ll,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        fitted = HawkesParams(*np.exp(result.x))

        # Ensure stationarity
        if not fitted.stationary:
            fitted = HawkesParams(fitted.mu, 0.99 * fitted.beta, fitted.beta)

        ll     = self.log_likelihood(fitted, times, T)
        ks_val = self.ks_test(fitted, times, T)

        return HawkesResult(
            params         = fitted,
            log_likelihood = ll,
            n_events       = len(times),
            T              = T,
            converged      = result.success,
            ks_statistic   = ks_val,
        )

    # ── Simulation (Ogata thinning) ──────────────────────────────────────────

    @staticmethod
    def simulate(
        params: HawkesParams,
        T:      float,
        seed:   int = 42,
    ) -> np.ndarray:
        """
        Simulate Hawkes event times via Ogata's thinning algorithm.

        Parameters
        ----------
        params : Hawkes parameters (μ, α, β).
        T      : Simulation horizon (seconds).
        seed   : Random seed.

        Returns
        -------
        np.ndarray : Sorted event arrival times.
        """
        rng    = np.random.default_rng(seed)
        times: list[float] = []
        t = 0.0
        lam_bar = params.mu

        while t < T:
            dt = rng.exponential(1.0 / lam_bar)
            t += dt
            if t >= T:
                break

            # Exact intensity at t
            intensity = params.mu + params.alpha * sum(
                np.exp(-params.beta * (t - ti)) for ti in times
            )

            if rng.uniform() <= intensity / lam_bar:
                times.append(t)

            # Update upper bound
            lam_bar = params.mu + params.alpha * sum(
                np.exp(-params.beta * (t - ti)) for ti in times
            )
            lam_bar = max(lam_bar, params.mu)

        return np.array(times)


class HawkesSignal(BaseSignal):
    """
    Hawkes-based signal: rolling estimate of branching ratio and intensity.

    The branching ratio α/β measures the fraction of arrivals that are
    self-excited (endogenous). A high branching ratio signals momentum;
    a low ratio signals a more random (Poisson-like) flow.

    Parameters
    ----------
    fit_window : int
        Number of events in the rolling fit window.
    """

    def __init__(self, fit_window: int = 200) -> None:
        self.fit_window = fit_window
        self._process   = HawkesProcess()

    def name(self) -> str:
        return f"Hawkes_FW{self.fit_window}"

    def compute(self, snapshot: pd.DataFrame) -> pd.Series:
        """
        Compute rolling Hawkes branching ratio as a signal.

        Parameters
        ----------
        snapshot : DataFrame with columns 'time' (seconds) and 'event',
                   where 'event' is 1 when an order arrives, 0 otherwise.

        Returns
        -------
        pd.Series : Rolling branching ratio α/β ∈ [0, 1).
        """
        if "time" not in snapshot.columns:
            snapshot = snapshot.copy()
            snapshot["time"] = np.arange(len(snapshot)) * 0.1

        event_mask = snapshot.get("event",
                                   pd.Series(np.ones(len(snapshot)),
                                             index=snapshot.index))
        times = snapshot.loc[event_mask > 0, "time"].values

        if len(times) < self.fit_window:
            return pd.Series(np.nan, index=snapshot.index, name=self.name())

        result = np.full(len(snapshot), np.nan)

        # Rolling fit every 50 events
        for end in range(self.fit_window, len(times) + 1, 50):
            window_times = times[max(0, end - self.fit_window) : end]
            T_window = window_times[-1] - window_times[0]
            if T_window < 1e-3:
                continue
            shifted = window_times - window_times[0]
            try:
                fit = self._process.fit(shifted, T_window)
                # Find tick index corresponding to times[end-1]
                t_val = times[end - 1]
                idx   = np.searchsorted(snapshot["time"].values, t_val)
                if 0 <= idx < len(result):
                    result[idx] = fit.params.branching_ratio
            except Exception:
                pass

        return pd.Series(result, index=snapshot.index,
                          name=self.name()).ffill()
