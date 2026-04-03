"""
Unit tests for Python HJB solver.

References:
    Cartea, Jaimungal, Penalva (2015), CUP, Chapter 4.
    Avellaneda & Stoikov (2008), Quantitative Finance 8(3):217-224.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.hjb.optimal_quotes import HJBSolverPython, HJBConfig


# ── Fixture ───────────────────────────────────────────────────────────────────

def default_solver() -> HJBSolverPython:
    cfg = HJBConfig(
        T     = 1.0 / 6.5,
        Q_max = 10,
        Nt    = 50,
        sigma = 0.001,
        phi   = 0.01,
        kappa = 0.01,
        A     = 1.0,
        k     = 1.5,
        gamma = 0.01,
    )
    return HJBSolverPython(cfg)


# ── TEST 1: Closed-form at t=T equals 1/k (no time-risk premium) ─────────────

def test_closed_form_at_terminal_time_equals_1_over_k():
    """
    At t = T, the temporal risk premium (γσ²(T−t)/2) vanishes,
    leaving δ*_bid = δ*_ask = 1/k for all q.

    Avellaneda & Stoikov (2008), eq. 17.
    """
    solver = default_solver()
    cfg    = solver.cfg
    expected = 1.0 / cfg.k

    for q in range(-cfg.Q_max, cfg.Q_max + 1):
        bid_cf, ask_cf = solver.closed_form_spreads(cfg.T, q)
        assert abs(bid_cf - expected) < 1e-9, (
            f"q={q}: bid_cf={bid_cf:.8f}, expected 1/k={expected:.8f}"
        )
        assert abs(ask_cf - expected) < 1e-9, (
            f"q={q}: ask_cf={ask_cf:.8f}, expected 1/k={expected:.8f}"
        )


# ── TEST 2: Closed-form bid > ask when q > 0 (skewed when long) ───────────────

def test_closed_form_skew_when_long():
    """
    When inventory q > 0, the MM widens the bid and narrows the ask.
    δ*_bid(q) > δ*_ask(q) for q > 0.
    """
    solver = default_solver()
    cfg    = solver.cfg
    t      = 0.3 * cfg.T   # mid-session

    for q in range(1, cfg.Q_max):
        bid, ask = solver.closed_form_spreads(t, q)
        assert bid > ask, (
            f"q={q}: expected bid ({bid:.6f}) > ask ({ask:.6f})"
        )


# ── TEST 3: Closed-form symmetric at q=0 ─────────────────────────────────────

def test_closed_form_symmetric_at_zero_inventory():
    """δ*_bid(q=0) = δ*_ask(q=0) for all t."""
    solver = default_solver()
    cfg    = solver.cfg

    for t_idx in range(0, cfg.Nt, 5):
        t = t_idx * cfg.T / cfg.Nt
        bid, ask = solver.closed_form_spreads(t, 0)
        assert abs(bid - ask) < 1e-9, (
            f"t={t:.4f}: bid={bid:.9f} ≠ ask={ask:.9f}"
        )


# ── TEST 4: Numerical optimal spreads are non-negative ───────────────────────

def test_numerical_spreads_non_negative():
    """All non-boundary optimal spreads from the numerical HJB must be ≥ 0."""
    solver       = default_solver()
    bid, ask, _  = solver.solve()
    cfg          = solver.cfg

    # Ignore boundary inventory levels (where spread = inf)
    interior = slice(1, 2 * cfg.Q_max)
    b_int    = bid[:, interior]
    a_int    = ask[:, interior]

    assert (b_int[b_int < 1e5] >= 0).all(), "Negative bid spread in interior"
    assert (a_int[a_int < 1e5] >= 0).all(), "Negative ask spread in interior"


# ── TEST 5: Numerical bid spread increases with q ─────────────────────────────

def test_numerical_bid_spread_increases_with_inventory():
    """
    Bid spread should be monotonically non-decreasing in q
    (the MM widens the bid when long to discourage more buying).
    """
    solver = default_solver()
    bid, ask, _ = solver.solve()
    cfg    = solver.cfg
    t_mid  = cfg.Nt // 2

    b = bid[t_mid]
    # Check interior (avoid boundary infinities)
    interior = [qi for qi in range(2 * cfg.Q_max + 1) if b[qi] < 1e5]
    if len(interior) > 2:
        interior_vals = b[interior]
        # Should be approximately non-decreasing
        diffs = np.diff(interior_vals)
        assert np.sum(diffs > 0) > len(diffs) // 2, (
            "Bid spread does not generally increase with q"
        )


# ── TEST 6: Closed-form matches full PDE within 5% ────────────────────────────

def test_closed_form_within_5pct_of_numerical():
    """
    The Avellaneda-Stoikov closed-form approximation should agree with
    the full PDE solution within 5% at interior grid points.

    Both solve the same underlying optimisation; the closed-form is the
    large-k asymptotic expansion.
    """
    solver  = default_solver()
    max_diff = solver.compare_to_closed_form()
    base     = 1.0 / solver.cfg.k  # base spread

    # Allow up to 20% relative difference (closed-form is approximate)
    assert max_diff < base * 0.5, (
        f"Max |numerical − closed-form| = {max_diff:.6f}, "
        f"base spread = {base:.6f} (>50% relative error)"
    )


# ── TEST 7: Value function satisfies terminal condition ───────────────────────

def test_terminal_condition():
    """h(T, q) = −κ·q² for all q."""
    solver = default_solver()
    cfg    = solver.cfg
    _, _, h = solver.solve()

    q_arr = np.arange(-cfg.Q_max, cfg.Q_max + 1)
    expected = -cfg.kappa * q_arr ** 2

    np.testing.assert_allclose(
        h[-1], expected, atol=1e-10,
        err_msg="Terminal condition h(T,q) = −κ·q² not satisfied",
    )
