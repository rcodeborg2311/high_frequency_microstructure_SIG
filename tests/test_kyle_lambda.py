"""
Unit tests for KyleLambdaSignal.

References:
    Kyle (1985), Econometrica 53(6):1315-1335.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.signals.kyle_lambda import KyleLambdaSignal
from python.data.synthetic import SyntheticLOB


# ── TEST 1: Lambda > 0 (price moves in direction of order flow) ───────────────

def test_kyle_lambda_positive():
    """
    λ should be positive: buying pressure moves prices up,
    selling pressure moves prices down (Kyle 1985, p. 1319).
    """
    rng = np.random.default_rng(42)
    n   = 2_000

    # Construct data where λ = 0.0001 exactly: ΔP = 0.0001·q + ε
    q  = rng.normal(0, 100, n)
    dp = 0.0001 * q + rng.normal(0, 0.0001 * 0.1, n)

    signal = KyleLambdaSignal(window=100)
    lam    = signal.compute_from_arrays(q, dp)

    valid  = lam[~np.isnan(lam)]
    assert len(valid) > 0
    # Mean lambda should be close to 0.0001 and positive
    mean_lam = float(np.mean(valid))
    assert mean_lam > 0, f"Expected λ > 0, got {mean_lam}"


# ── TEST 2: Rolling lambda is constant for constant price impact ──────────────

def test_kyle_lambda_constant_impact():
    """
    If ΔP = λ_true · q with no noise, rolling λ̂ should equal λ_true exactly.
    """
    n       = 500
    lam_true = 0.00025
    rng     = np.random.default_rng(7)

    q  = rng.normal(0, 200, n)
    dp = lam_true * q   # exact linear impact, no noise

    signal = KyleLambdaSignal(window=50)
    lam    = signal.compute_from_arrays(q, dp)

    # After the warm-up window, all estimates should equal lam_true
    valid = lam[50:]
    assert np.allclose(valid, lam_true, atol=1e-10), (
        f"Expected constant λ={lam_true}, got max deviation "
        f"{np.abs(valid - lam_true).max()}"
    )


# ── TEST 3: Lambda higher for illiquid (wide-spread) synthetic data ───────────

def test_kyle_lambda_higher_for_illiquid():
    """
    Illiquid instruments (fewer informed traders, less depth) exhibit
    higher price impact per unit flow.

    We test this by constructing two datasets with different impact coefficients.
    """
    n   = 2_000
    rng = np.random.default_rng(3)
    q   = rng.normal(0, 100, n)

    lam_liquid   = 0.00005
    lam_illiquid = 0.00050

    dp_liquid   = lam_liquid   * q + rng.normal(0, lam_liquid   * 5, n)
    dp_illiquid = lam_illiquid * q + rng.normal(0, lam_illiquid * 5, n)

    sig = KyleLambdaSignal(window=100)
    lam_l = sig.compute_from_arrays(q, dp_liquid)[100:]
    lam_i = sig.compute_from_arrays(q, dp_illiquid)[100:]

    mean_l = float(np.nanmean(lam_l))
    mean_i = float(np.nanmean(lam_i))

    assert mean_i > mean_l, (
        f"Expected illiquid λ ({mean_i:.6f}) > liquid λ ({mean_l:.6f})"
    )


# ── TEST 4: OLS formula matches sklearn LinearRegression within 1e-10 ─────────

def test_kyle_lambda_matches_sklearn():
    """
    The closed-form OLS estimator λ̂ = Σ(q·ΔP)/Σ(q²) must match
    sklearn.LinearRegression within 1e-10 (no intercept).
    """
    n   = 300
    rng = np.random.default_rng(99)
    q   = rng.normal(0, 100, n)
    dp  = 0.0002 * q + rng.normal(0, 0.00005, n)

    # Sklearn estimate (no intercept, full sample)
    lr     = LinearRegression(fit_intercept=False)
    lr.fit(q.reshape(-1, 1), dp)
    lam_sklearn = float(lr.coef_[0])

    # Closed-form (full window = all n points)
    lam_cf = float(np.sum(q * dp) / np.sum(q ** 2))

    assert abs(lam_cf - lam_sklearn) < 1e-10, (
        f"Closed-form ({lam_cf:.10f}) vs sklearn ({lam_sklearn:.10f}): "
        f"diff = {abs(lam_cf - lam_sklearn):.2e}"
    )


# ── TEST 5: Rolling lambda returns NaN for first (window-1) ticks ────────────

def test_kyle_lambda_nan_warmup():
    """First (window-1) ticks must be NaN (insufficient history)."""
    n      = 200
    window = 50
    rng    = np.random.default_rng(0)
    q      = rng.normal(0, 100, n)
    dp     = 0.0001 * q

    sig = KyleLambdaSignal(window=window)
    lam = sig.compute_from_arrays(q, dp)

    assert np.all(np.isnan(lam[:window - 1])), "Expected NaN for warm-up period"
    assert not np.isnan(lam[window]), "Expected valid value after warm-up"
