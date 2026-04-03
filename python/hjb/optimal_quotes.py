"""
Python HJB solver: optimal quoting from the Cartea-Jaimungal-Penalva equation.

Validation and visualization counterpart to the C++ HJBSolver.

References:
    Cartea, Jaimungal, Penalva (2015), CUP, Chapter 4.
    Avellaneda & Stoikov (2008), Quantitative Finance 8(3):217-224.

Value function decomposition:
    V(t, x, q, s) = x + qs + h(t, q)

HJB for h(t, q):
    ∂h/∂t = φq²σ²
           − (A/(k·e)) · exp(k·(h(t,q+1) − h(t,q)))   [bid]
           − (A/(k·e)) · exp(k·(h(t,q−1) − h(t,q)))   [ask]

Terminal condition:
    h(T, q) = −κ·q²

Optimal spreads:
    δ*_bid(t,q) = max(0, 1/k − (h(t,q+1) − h(t,q)))
    δ*_ask(t,q) = max(0, 1/k − (h(t,q−1) − h(t,q)))

Closed-form (Avellaneda-Stoikov large-k approximation):
    δ*_bid = 1/k + (γσ²/2)·(T−t) + γσ²·(q − ½)·(T−t)/k
    δ*_ask = 1/k + (γσ²/2)·(T−t) − γσ²·(q + ½)·(T−t)/k
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class HJBConfig:
    """Solver configuration matching the C++ HJBSolver::Config."""
    T:     float = 1.0 / 6.5   # trading session length (years ≈ 6.5 h)
    Q_max: int   = 10           # maximum |inventory|
    Nt:    int   = 100          # time steps for backward Euler
    sigma: float = 0.001        # mid-price diffusion
    phi:   float = 0.01         # running inventory penalty
    kappa: float = 0.01         # terminal inventory penalty
    A:     float = 1.0          # fill-rate constant
    k:     float = 1.5          # fill-rate depth parameter
    gamma: float = 0.01         # risk aversion (= phi for CARA)


class HJBSolverPython:
    """
    Pure Python / NumPy implementation of the HJB optimal quoting engine.

    Produces the same quoting policy as the C++ HJBSolver for validation.
    """

    def __init__(self, cfg: HJBConfig = HJBConfig()) -> None:
        self.cfg = cfg

    # ── Closed-form (Avellaneda-Stoikov) ─────────────────────────────────────

    def closed_form_spreads(
        self,
        t: float,
        q: int,
    ) -> tuple[float, float]:
        """
        Avellaneda-Stoikov closed-form optimal spreads (large-k limit).

        Avellaneda & Stoikov (2008), eq. 17.

        Returns
        -------
        (δ*_bid, δ*_ask)
        """
        c   = self.cfg
        tau = c.T - t
        base    = 1.0 / c.k
        risk    = c.gamma * c.sigma ** 2 * tau
        inv_adj = risk / c.k

        bid = base + 0.5 * risk + (q - 0.5) * inv_adj
        ask = base + 0.5 * risk - (q + 0.5) * inv_adj

        return max(0.0, bid), max(0.0, ask)

    # ── Full PDE: backward Euler ─────────────────────────────────────────────

    def solve_value_function(self) -> np.ndarray:
        """
        Solve h(t, q) via backward Euler.

        Returns
        -------
        np.ndarray shape (Nt+1, 2·Q_max+1): value function grid.
        """
        c   = self.cfg
        nQ  = 2 * c.Q_max + 1
        dt  = c.T / c.Nt
        q_arr = np.arange(-c.Q_max, c.Q_max + 1)

        h = np.zeros((c.Nt + 1, nQ))

        # Terminal condition: h(T, q) = −κ·q²  (CJP eq. 4.18)
        h[c.Nt] = -c.kappa * q_arr ** 2

        A_over_ke = c.A / (c.k * np.e)

        for t_idx in range(c.Nt - 1, -1, -1):
            h_next = h[t_idx + 1]

            # Bid contribution: exp(k·(h(q+1) − h(q)))  for q < Q_max
            delta_bid = np.full(nQ, 0.0)
            delta_bid[:-1] = h_next[1:] - h_next[:-1]
            A_bid = np.where(
                q_arr < c.Q_max,
                A_over_ke * np.exp(c.k * delta_bid),
                0.0,
            )

            # Ask contribution: exp(k·(h(q−1) − h(q)))  for q > −Q_max
            delta_ask = np.full(nQ, 0.0)
            delta_ask[1:] = h_next[:-1] - h_next[1:]
            A_ask = np.where(
                q_arr > -c.Q_max,
                A_over_ke * np.exp(c.k * delta_ask),
                0.0,
            )

            phi_term = c.phi * q_arr ** 2 * c.sigma ** 2

            h[t_idx] = h_next + dt * (phi_term - A_bid - A_ask)

        return h

    def solve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the HJB and return (bid_spread, ask_spread, value_function).

        Returns
        -------
        bid_spread : shape (Nt, 2·Q_max+1)
        ask_spread : shape (Nt, 2·Q_max+1)
        h          : shape (Nt+1, 2·Q_max+1)  value function
        """
        c   = self.cfg
        nQ  = 2 * c.Q_max + 1
        q_arr = np.arange(-c.Q_max, c.Q_max + 1)

        h = self.solve_value_function()

        bid_spread = np.full((c.Nt, nQ), np.inf)
        ask_spread = np.full((c.Nt, nQ), np.inf)

        for t_idx in range(c.Nt):
            # δ*_bid = max(0, 1/k − (h(q+1) − h(q)))
            dh_bid = np.zeros(nQ)
            dh_bid[:-1] = h[t_idx, 1:] - h[t_idx, :-1]
            bid_spread[t_idx] = np.where(
                q_arr < c.Q_max,
                np.maximum(0.0, 1.0 / c.k - dh_bid),
                np.inf,
            )

            # δ*_ask = max(0, 1/k − (h(q−1) − h(q)))
            dh_ask = np.zeros(nQ)
            dh_ask[1:] = h[t_idx, :-1] - h[t_idx, 1:]
            ask_spread[t_idx] = np.where(
                q_arr > -c.Q_max,
                np.maximum(0.0, 1.0 / c.k - dh_ask),
                np.inf,
            )

        return bid_spread, ask_spread, h

    def get_quotes(
        self,
        t: float,
        q: int,
        bid_spread: np.ndarray,
        ask_spread: np.ndarray,
    ) -> tuple[float, float]:
        """
        Look up optimal quotes from the pre-computed policy arrays.

        Parameters
        ----------
        t          : Current time ∈ [0, T].
        q          : Current inventory ∈ [−Q_max, Q_max].
        bid_spread : From solve().
        ask_spread : From solve().
        """
        c = self.cfg
        qi    = q + c.Q_max
        t_idx = min(int(t / c.T * c.Nt), c.Nt - 1)
        return float(bid_spread[t_idx, qi]), float(ask_spread[t_idx, qi])

    # ── Comparison to closed form ─────────────────────────────────────────────

    def compare_to_closed_form(self) -> float:
        """
        Return max |δ*_numerical − δ*_closed_form| over all (t, q).
        """
        c  = self.cfg
        dt = c.T / c.Nt
        bid, ask, _ = self.solve()
        q_arr = np.arange(-c.Q_max, c.Q_max + 1)

        max_diff = 0.0
        for t_idx in range(c.Nt):
            t = t_idx * dt
            for qi, q in enumerate(q_arr):
                b_num = bid[t_idx, qi]
                a_num = ask[t_idx, qi]
                if b_num > 1e5 or a_num > 1e5:
                    continue
                b_cf, a_cf = self.closed_form_spreads(t, int(q))
                max_diff = max(max_diff, abs(b_num - b_cf), abs(a_num - a_cf))

        return max_diff

    # ── Visualization ─────────────────────────────────────────────────────────

    def plot_policy(
        self,
        output_path: str = "results/hjb_optimal_spreads.png",
    ) -> None:
        """
        Plot optimal bid and ask spreads vs inventory at three time slices.
        """
        c     = self.cfg
        bid, ask, _ = self.solve()
        q_arr = np.arange(-c.Q_max, c.Q_max + 1)

        fig, ax = plt.subplots(figsize=(9, 5))

        t_slices = [0, c.Nt // 2, c.Nt - 1]
        labels   = [f"t = 0", f"t = T/2", f"t ≈ T"]
        colors   = ["steelblue", "darkorange", "green"]

        for t_idx, label, color in zip(t_slices, labels, colors):
            b = bid[t_idx]
            a = ask[t_idx]
            finite = (b < 1e5) & (a < 1e5)
            ax.plot(q_arr[finite], b[finite],
                    color=color, linestyle="-",
                    label=f"Bid ({label})")
            ax.plot(q_arr[finite], a[finite],
                    color=color, linestyle="--",
                    label=f"Ask ({label})")

        ax.set_xlabel("Inventory q (shares)")
        ax.set_ylabel("Optimal half-spread δ* (price units)")
        ax.set_title("HJB Optimal Bid/Ask Spreads vs Inventory")
        ax.legend(loc="upper center", ncol=3, fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved HJB policy plot to: {output_path}")


def solve_and_plot(
    cfg: HJBConfig = HJBConfig(),
    output_path: str = "results/hjb_optimal_spreads.png",
) -> None:
    """Convenience wrapper: solve HJB and save the policy chart."""
    solver = HJBSolverPython(cfg)
    solver.plot_policy(output_path)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg    = HJBConfig(Nt=100, Q_max=10, sigma=0.001, phi=0.01, k=1.5)
    solver = HJBSolverPython(cfg)

    bid, ask, h = solver.solve()
    q_vals = np.arange(-cfg.Q_max, cfg.Q_max + 1)
    t_half = cfg.T * 0.5
    t_idx  = cfg.Nt // 2

    print("HJB Optimal Quotes — t = T/2")
    print(f"{'q':>5}  {'δ*_bid':>10}  {'δ*_ask':>10}  {'half-spread':>12}  {'skew':>8}")
    print("─" * 55)
    for q in [-5, -3, 0, 3, 5]:
        qi = q + cfg.Q_max
        b  = bid[t_idx, qi]
        a  = ask[t_idx, qi]
        print(f"{q:>5}  {b:>10.6f}  {a:>10.6f}  "
              f"{(b+a)/2:>12.6f}  {a-b:>8.6f}")

    diff = solver.compare_to_closed_form()
    print(f"\nMax |numerical − closed-form| = {diff:.6f}")

    solver.plot_policy("results/hjb_optimal_spreads.png")
