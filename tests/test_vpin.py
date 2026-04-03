"""
Unit tests for VPINSignal.

References:
    Easley, López de Prado & O'Hara (2012), RFS 25(5):1457-1493.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.signals.vpin import VPINSignal
from python.data.synthetic import SyntheticLOB


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_balanced_flow(n_buckets: int = 100, bucket_size: int = 1_000) -> tuple:
    """
    Balanced buy/sell flow: prices random-walk, volumes equal per tick.
    Expect VPIN ≈ 0.5 (equal toxicity).
    """
    rng    = np.random.default_rng(42)
    n      = n_buckets * bucket_size
    prices = np.cumsum(rng.normal(0, 0.001, n)) + 100.0
    vols   = np.ones(n) * 100.0  # constant volume per tick
    return prices, vols


def make_informed_flow(n: int = 20_000, bucket_size: int = 500) -> tuple:
    """
    Informed (one-directional) flow: prices trend strongly upward.
    Expect VPIN to be high (informed trading detected).

    Uses a large trend (0.1/tick) relative to noise (0.05/tick) so that
    within-bucket ΔP consistently exceeds σ_ΔP, pushing VPIN well above 0.5.
    """
    t      = np.arange(n)
    # Trend dominates noise: signal-to-noise ≈ 2:1 per tick.
    prices = 100.0 + 0.10 * t + np.random.default_rng(7).normal(0, 0.05, n)
    vols   = np.ones(n) * 200.0
    return prices, vols


# ── TEST 1: VPIN ∈ [0, 1] always ─────────────────────────────────────────────

def test_vpin_bounded():
    """VPIN must lie in [0, 1] at all bucket completions."""
    gen  = SyntheticLOB(seed=0)
    df   = gen.generate(n_steps=20_000)
    vpin = VPINSignal(bucket_size=2_000, n_buckets=20)

    prices = df["mid_price"].values
    vols   = np.ones(len(prices)) * 500.0

    vals = vpin.compute_from_arrays(prices, vols)
    assert len(vals) > 0, "No bucket completions"
    assert (vals >= 0.0 - 1e-9).all(), f"VPIN below 0: {vals.min()}"
    assert (vals <= 1.0 + 1e-9).all(), f"VPIN above 1: {vals.max()}"


# ── TEST 2: VPIN ≈ 0.5 for balanced buy/sell flow ────────────────────────────

def test_vpin_balanced_flow():
    """
    For a symmetric random walk with equal volumes, VPIN should be ≈ 0.5
    (Easley et al. 2012: VPIN ≈ 0.5 under balanced order flow).
    """
    prices, vols = make_balanced_flow(n_buckets=200, bucket_size=500)
    vpin = VPINSignal(bucket_size=500, n_buckets=50)
    vals = vpin.compute_from_arrays(prices, vols)

    assert len(vals) >= 50, "Need at least 50 buckets"
    mean_vpin = float(np.mean(vals[-50:]))
    # Allow generous tolerance: VPIN ≈ 0.5 ± 0.2 for balanced flow
    assert 0.2 <= mean_vpin <= 0.8, (
        f"Expected VPIN ≈ 0.5 for balanced flow, got {mean_vpin:.3f}"
    )


# ── TEST 3: VPIN increases during informed trading episodes ───────────────────

def test_vpin_high_during_informed_trading():
    """
    A strongly trending price series (one-directional order flow) should
    produce VPIN > 0.5 on average.
    """
    prices, vols = make_informed_flow(n=30_000, bucket_size=300)
    vpin = VPINSignal(bucket_size=300, n_buckets=30)
    vals = vpin.compute_from_arrays(prices, vols)

    assert len(vals) >= 20, "Need at least 20 buckets"
    mean_vpin = float(np.mean(vals))
    assert mean_vpin > 0.5, (
        f"Expected VPIN > 0.5 for informed (trending) flow, got {mean_vpin:.3f}"
    )


# ── TEST 4: Bucket filling logic is correct ───────────────────────────────────

def test_vpin_bucket_accumulation():
    """
    Buckets are filled to exactly bucket_size volumes.
    The number of completed buckets should match floor(total_volume / bucket_size).
    """
    bucket_size   = 1_000
    n_ticks       = 5_000
    vol_per_tick  = 100.0
    total_vol     = n_ticks * vol_per_tick
    expected_buckets = int(total_vol // bucket_size)

    rng    = np.random.default_rng(1)
    prices = np.cumsum(rng.normal(0, 0.001, n_ticks)) + 100.0
    vols   = np.full(n_ticks, vol_per_tick)

    vpin = VPINSignal(bucket_size=bucket_size, n_buckets=10)
    vals = vpin.compute_from_arrays(prices, vols)

    # Allow ±1 for floating-point boundary conditions at the last bucket.
    assert abs(len(vals) - expected_buckets) <= 1, (
        f"Expected {expected_buckets} buckets (±1), got {len(vals)}"
    )


# ── TEST 5: Tick-level VPIN series has correct index and range ────────────────

def test_vpin_compute_tick_level():
    """compute() returns a series with same index as input and values in [0, 1]."""
    gen = SyntheticLOB(seed=3)
    df  = gen.generate(n_steps=5_000)
    # Add volume column for compute()
    df  = df.copy()

    vpin = VPINSignal(bucket_size=1_000, n_buckets=10)
    sig  = vpin.compute(df)

    assert len(sig) == len(df), "Length mismatch"
    assert sig.index.equals(df.index), "Index mismatch"
    valid = sig.dropna()
    assert (valid >= 0.0 - 1e-9).all()
    assert (valid <= 1.0 + 1e-9).all()
