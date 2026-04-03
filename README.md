# High-Frequency Market Microstructure Signal Platform

[![CI](https://github.com/placeholder/hf-microstructure/actions/workflows/ci.yml/badge.svg)](https://github.com/placeholder/hf-microstructure/actions)

## Overview

Market microstructure studies how the trading process—order book events, trade
execution, and quoting strategies—determines short-term price formation and
liquidity. HFT firms profit by decoding the informational content of limit order
book data faster than competitors: which orders signal informed traders, how
quickly does a price signal decay, and how should a market maker adjust quotes
to balance profit against inventory risk?

This platform provides a full research-grade implementation of the core
microstructure toolkit: four alpha signals derived from Level-2 LOB data
(Order Flow Imbalance, VPIN, Kyle's lambda, Hawkes arrival clustering),
rigorous econometric evaluation (IC curves, alpha decay, ICIR), a signal
combiner, and a Hamilton-Jacobi-Bellman PDE optimal quoting engine implementing
the Cartea-Jaimungal-Penalva framework. The C++ layer achieves sub-microsecond
feature computation suitable for live deployment; the Python research layer
provides interactive analysis and visualization.

## The Signal Research Pipeline

```
[LOB Data: FI-2010 / LOBSTER / Synthetic]
            │
            ▼
   [Feature Engine (C++, <1 μs/tick)]
            │
   ┌────────┼────────┬──────────────┐
   ▼        ▼        ▼              ▼
 [OFI]   [VPIN]  [Kyle λ]      [Hawkes]
   │        │        │              │
   └────────┴────────┴──────────────┘
            │
   [IC Analysis & Alpha Decay (horizons 1..100 ticks)]
            │
   [Signal Combination: IC-weighted / PCA / Ridge]
            │
   [HJB Optimal Quoting Engine (C++ + Python)]
            │
   [Optimal Bid/Ask Quotes: δ*_bid(t,q), δ*_ask(t,q)]
            │
   [Plotly Dash Live Dashboard]
```

---

## The Mathematics

### Order Flow Imbalance — Cont, Kukanov & Stoikov (2014)

OFI measures the net directional pressure from order book events at each tick:

```
e_n(bid) = V_n^b · 1{P_n^b ≥ P_{n-1}^b}
         − V_n^b · 1{P_n^b < P_{n-1}^b}
         − V_{n-1}^b · 1{P_n^b ≤ P_{n-1}^b}

e_n(ask) = −V_n^a · 1{P_n^a ≤ P_{n-1}^a}
         + V_n^a · 1{P_n^a > P_{n-1}^a}
         + V_{n-1}^a · 1{P_n^a ≥ P_{n-1}^a}

OFI_n = e_n(bid) + e_n(ask)
```

**Intuition:** When the best bid price rises (or more volume arrives at the
existing best bid), buy-side pressure increases and price tends to move up.
OFI aggregates these micro-events into a single directional signal.

Normalized rolling OFI:
```
ŌFIP(W) = Σ_{n=t-W+1}^{t} OFI_n / (Σ |OFI_n| + ε)   ∈ [−1, +1]
```

Cont et al. (2014) find R² of 5–15% in the regression ΔP_mid = α + β·ŌFIP.

Multi-level OFI aggregates across all 5 LOB levels, reducing noise via
diversification. Level weights are estimated by PCA (first principal component).

---

### VPIN — Easley, López de Prado & O'Hara (2012)

VPIN measures the probability of trading against an informed counterparty.
Instead of trade-by-trade Lee-Ready signing (error-prone at high frequency),
it uses bulk volume classification:

```
V_buy(τ) = V_τ · Φ(ΔP_τ / σ_{ΔP})
V_sell(τ) = V_τ − V_buy(τ)

VPIN = (1/n) · Σ_{τ=t-n+1}^{t} |V_buy(τ) − V_sell(τ)| / V_τ
```

where τ indexes equal-volume buckets (V_τ = constant) and n = 50.

**Intuition:** Informed traders tend to buy (sell) when they have positive
(negative) private information, creating order flow imbalance within a
volume bucket. High VPIN warns that the next fill is likely against an
informed trader — adverse selection risk is elevated.

---

### Kyle's Lambda — Kyle (1985)

Linear price impact:
```
ΔP = λ · q + ε
```

Rolling OLS estimator (closed-form, O(1) per tick with running sums):
```
λ̂ = Σ_{t=t-W+1}^{t} q_s · ΔP_s  /  Σ q_s²
```

**Intuition:** λ measures market depth — how many cents the mid-price moves
per share of net order flow. Higher λ = thinner book, more sensitive to
each order. Exhibits a U-shaped intraday pattern (high at open/close,
low at midday) consistent with information asymmetry theory.

---

### Hawkes Processes — Bacry, Mastromatteo & Muzy (2015)

Order arrivals are self-exciting: each order temporarily increases the
probability of further orders (clustering, feedback loops).

Intensity:
```
λ(t) = μ + α · Σ_{t_i < t} exp(−β(t − t_i))
```

Log-likelihood (O(N) via the recursive Rᵢ formula):
```
L(μ,α,β) = −μT − (α/β)·Σᵢ[1 − exp(−β(T−tᵢ))] + Σᵢ log(μ + α·Rᵢ)
```

**O(N) recursive Rᵢ** (the key implementation detail for large tick data):
```
R_1 = 0
R_i = (R_{i-1} + 1) · exp(−β · (t_i − t_{i-1}))
```

Fitted parameters reveal market structure:
- **Branching ratio α/β:** fraction of orders that are self-excited (endogenous)
- **Excitation half-life log(2)/β:** how long the excitement from one order persists
- **Baseline μ:** exogenous (fundamental news-driven) arrival rate

---

### HJB Optimal Quoting — Cartea, Jaimungal & Penalva (2015)

A CARA-utility market maker solves the Hamilton-Jacobi-Bellman PDE to
determine the bid/ask spread that maximises risk-adjusted wealth.

Value function decomposition (CJP §4.2):
```
V(t, x, q, s) = x + qs + h(t, q)
```

HJB for h (backward-in-time PDE):
```
∂h/∂t = φq²σ²
       − (A/ke) · exp(k·(h(t,q+1) − h(t,q)))   [optimal bid contribution]
       − (A/ke) · exp(k·(h(t,q−1) − h(t,q)))   [optimal ask contribution]

h(T, q) = −κq²   [terminal condition]
```

Fill rate: Λ(δ) = A · exp(−k·δ)  (exponential decay with depth δ).

Optimal spreads from the value function:
```
δ*_bid(t,q) = max(0, 1/k − (h(t,q+1) − h(t,q)))
δ*_ask(t,q) = max(0, 1/k − (h(t,q−1) − h(t,q)))
```

Closed-form Avellaneda-Stoikov approximation (large-k limit):
```
δ*_bid(t,q) = 1/k + (γσ²/2)·(T−t) + γσ²·(q − ½)·(T−t)/k
δ*_ask(t,q) = 1/k + (γσ²/2)·(T−t) − γσ²·(q + ½)·(T−t)/k
```

**Intuition:** The spread has two components:
1. **Base spread 1/k**: minimum spread to earn positive expected revenue from fills.
2. **Risk premium (γσ²/2)·(T−t)**: widens as more time remains and variance
   accumulates. Narrows to 1/k at end-of-day (no remaining inventory risk).

When q > 0 (long), the MM widens the bid (discourages buying) and narrows
the ask (encourages selling to reduce inventory) — a natural hedging behaviour.

---

## Signal Research Results

Run `python python/research/signal_report.py` on synthetic or real data
to populate this table.

| Signal    | IC(1 tick) | IC(10 ticks) | IC(100 ticks) | Half-life (ticks) | ICIR |
|-----------|------------|--------------|---------------|-------------------|------|
| OFI       |            |              |               |                   |      |
| VPIN      |            |              |               |                   |      |
| Kyle λ    |            |              |               |                   |      |
| Combined  |            |              |               |                   |      |

![Alpha Decay](results/alpha_decay.png)
*Alpha decay curves: IC(h) vs holding period. The steeper the slope,
the faster the signal loses predictive power.*

![IC Comparison](results/ic_comparison.png)

---

## Hawkes Process Fit Results

Run `./build/cpp/hf_demo` or `python python/hjb/optimal_quotes.py` to populate:

```
μ = X.XXX events/sec   (baseline arrival rate — exogenous flow)
α = X.XXX              (excitation amplitude)
β = X.XXX              (decay rate)
Branching ratio α/β = X.XXX  (X.X% of orders are self-excited / endogenous)
Half-life of excitation = X.X seconds
KS test statistic = 0.0XX   (well-calibrated if < 0.05 at 95% confidence)
```

---

## HJB Optimal Quotes

![HJB Policy](results/hjb_optimal_spreads.png)

*Optimal bid (solid) and ask (dashed) spreads as a function of inventory q
at t=0, T/2, T. Spreads widen with |q| and narrow toward session end.
At q=0 the policy is symmetric; positive q skews quotes toward selling.*

---

## Key Design Decisions

**Why OFI instead of simple bid/ask imbalance?**
Simple OBI = (bid_vol − ask_vol)/(bid_vol + ask_vol) conflates volume at the
best levels with the directional signal from order flow *events*. OFI tracks
incremental changes (arrivals, cancellations, price moves) and directly links
order book dynamics to price formation per Cont et al. (2014). Empirically
OFI has higher R² in mid-price regressions.

**Why VPIN uses bulk volume classification instead of Lee-Ready trade signing?**
Lee-Ready requires sub-millisecond timestamp precision to correctly attribute
trades to initiating orders. At HFT speeds this introduces systematic errors.
Bulk classification (Φ(ΔP/σ_ΔP)) uses only price changes within a bucket —
available from any LOB feed — and is robust to latency noise per Easley et al.
(2012).

**Why O(N) recursive Hawkes MLE instead of O(N²)?**
The naive double-sum for the log-likelihood runs in O(N²) per likelihood
evaluation. For a 10-hour trading day at 100 events/sec that is 3.6 million
events — the O(N²) cost is 1.3 × 10¹³ operations per optimisation step,
completely infeasible. The recursive formula Rᵢ = (Rᵢ₋₁ + 1)·exp(−βΔt)
reduces this to O(N) per evaluation, making real-time fitting practical.

**Why HJB PDE instead of pure RL for optimal quoting?**
The market maker's problem has a well-defined analytical structure: known
(or calibrated) fill-rate function Λ(δ), Gaussian price diffusion, and a
finite-horizon CARA utility objective. These conditions make the PDE solvable
in minutes, with a provably optimal policy and interpretable parameters. RL
would require millions of simulated trades to converge and would produce a
black-box policy. The PDE solution also provides analytical insight into *why*
spreads widen with inventory and time.

**Why IC-weighting instead of equal-weighting signal combination?**
Equal weighting ignores the empirical predictive content of each signal. A
signal with IC = 0.001 gets the same weight as one with IC = 0.08. IC-weighted
combination is the optimal linear combination under the Fundamental Law of
Active Management (Grinold & Kahn 2000): it maximises the combined ICIR.

---

## Interview Q&A

**Q: What does the branching ratio of a Hawkes process tell you about market structure?**

The branching ratio n = α/β ∈ [0, 1) is the expected number of secondary
("child") events triggered by each primary ("parent") event. If n = 0.7, then
70% of order arrivals are endogenous — they were triggered by the arrival of a
previous order (momentum, herding, algorithmic feedback) rather than by new
fundamental information. The remaining 30% (μ-driven) are exogenous. A market
approaching n → 1 is near the self-excitation instability, which produces
flash-crash-like bursts. Typical liquid equities have n ≈ 0.5–0.8 intraday.

**Q: Why is OFI a better predictor of short-term price moves than simple order imbalance?**

Simple order imbalance (OBI = (V_bid − V_ask)/(V_bid + V_ask)) is a snapshot
of queue sizes. It doesn't distinguish between a large resting order that has
been sitting for minutes (stale, low information) and a fresh large order that
just arrived (high information). OFI tracks *changes*: the arrival of new
volume at the best bid, the cancellation of ask volume, and price-level
movements. These events directly generate trading pressure. Cont et al. (2014)
show that OFI explains 5–15% of mid-price variance at 1-minute horizons, vs
< 1% for simple OBI. At sub-second horizons OFI is even more dominant.

**Q: Derive the optimal bid and ask spreads from the Avellaneda-Stoikov model.**

In A-S (2008), the MM maximises E[−exp(−γ·W(T))]. The mid-price follows GBM
with volatility σ. The MM posts bid at s − δ_b and ask at s + δ_a. Fill rates
are Λ(δ) = A·exp(−k·δ).

The HJB for the value function separates into an ODE for the indifference
price r(t,q) = s − qγσ²(T−t) (inventory-adjusted mid), and the optimal
spread δ* = γσ²(T−t)/2 + (2/γ)·ln(1 + γ/k). For large k (deep book),
ln(1 + γ/k) ≈ γ/k and δ* ≈ γσ²(T−t)/2 + 1/k. The inventory-adjusted
version subtracts q·γσ²(T−t)/k from the bid and adds it to the ask.

**Q: What is adverse selection and how does VPIN measure it?**

Adverse selection occurs when a market maker fills an order from an informed
trader who knows the asset's true value is far from the quoted price. The MM
ends up holding inventory at a loss. VPIN estimates the fraction of order flow
that is "toxic" (information-driven vs liquidity-driven) by measuring the
directional imbalance within equal-volume buckets. High VPIN (> 0.7) means
that most recent volume flowed predominantly in one direction — a signature of
informed trading. The MM should widen spreads proportionally to VPIN to
recover the expected adverse selection cost.

**Q: Kyle's lambda is estimated at 0.0002 $/share. What does that mean practically?**

λ = 0.0002 means a net order flow of 10,000 shares moves the price by
0.0002 × 10,000 = $2.00. For a $50 stock that is 4% — very illiquid.
A typical large-cap on NASDAQ might have λ ≈ 0.000005 (i.e., a 10,000-share
order moves price by $0.05 = 0.1%). A trading algorithm would use this
estimate to size orders (avoid exceeding the depth where price impact
dominates transaction cost) and to assess when its own order flow has been
noticed by other participants.

**Q: How would you combine these signals into a live trading strategy?**

1. **IC-weighted combination**: compute IC(h) at each horizon for each signal
   and form a weighted composite. Weight by IC to maximise ICIR.
2. **Position sizing**: use Kyle's lambda to determine the maximum position
   size where expected alpha exceeds expected price impact.
3. **Entry/exit**: enter when combined signal crosses ±1σ; use VPIN as a
   toxicity filter (reduce position size when VPIN > 0.7).
4. **Quoting**: use the HJB engine to compute optimal bid/ask spread at the
   current inventory q and time t. When the combined signal is strongly
   positive, skew quotes toward the ask (sell aggressively) and widen the
   bid.
5. **Risk management**: monitor branching ratio of the Hawkes fit in real time;
   widen spreads further when the market is near the excitation instability.

---

## References

1. Cont, R., Kukanov, A., Stoikov, S. (2014). "The Price Impact of Order Book
   Events." *Journal of Financial Econometrics* 12(1):47-88.

2. Easley, D., López de Prado, M. M., O'Hara, M. (2012). "Flow Toxicity and
   Liquidity in a High-Frequency World." *Review of Financial Studies*
   25(5):1457-1493.

3. Kyle, A. S. (1985). "Continuous Auctions and Insider Trading."
   *Econometrica* 53(6):1315-1335.

4. Bacry, E., Mastromatteo, I., Muzy, J.-F. (2015). "Hawkes Processes in
   Finance." *Market Microstructure and Liquidity* 1(1):1550005.

5. Cartea, Á., Jaimungal, S., Penalva, J. (2015). *Algorithmic and
   High-Frequency Trading.* Cambridge University Press.

6. Avellaneda, M., Stoikov, S. (2008). "High-frequency trading in a limit
   order book." *Quantitative Finance* 8(3):217-224.

7. Grinold, R. C., Kahn, R. N. (2000). *Active Portfolio Management.*
   McGraw-Hill, 2nd edition.

---

## Build & Run

### C++ components

```bash
# Configure and build (Release mode, O3 + march=native)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run the demo (streams synthetic data, prints IC/Hawkes/HJB tables)
./build/hf_demo

# Run C++ tests
ctest --test-dir build --output-on-failure
# or directly:
./build/hf_tests
```

### Python research pipeline

```bash
pip install -r requirements.txt

# Generate synthetic data and preview
python python/data/loader.py

# Run full IC analysis (prints IC table + alpha decay)
python python/research/ic_analysis.py

# Generate alpha decay curves (saves results/alpha_decay.png)
python python/research/alpha_decay.py

# Solve HJB and print optimal spread table
python python/hjb/optimal_quotes.py

# Launch live dashboard at http://localhost:8050
python python/dashboard/app.py
```

### Python tests

```bash
pytest tests/ -v
```

### All tests

```bash
# C++
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure

# Python
pytest tests/ -v --tb=short
```
