# HF Market Microstructure Signal Platform — Deep Dive

**Live deployment:** https://high-frequency-microstructure-sig.vercel.app  
**Backend:** https://highfrequencymicrostructuresig-production.up.railway.app  
**Repository:** https://github.com/rcodeborg2311/high_frequency_microstructure_SIG

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Signal Library](#3-signal-library)
   - 3.1 [Order Flow Imbalance (OFI)](#31-order-flow-imbalance-ofi)
   - 3.2 [VPIN — Volume-Synchronized PIN](#32-vpin--volume-synchronized-pin)
   - 3.3 [Kyle's Lambda](#33-kyles-lambda)
   - 3.4 [Hawkes Process](#34-hawkes-process)
4. [HJB Optimal Market-Making](#4-hjb-optimal-market-making)
5. [C++ High-Performance Engine](#5-c-high-performance-engine)
6. [Synthetic Data Generator](#6-synthetic-data-generator)
7. [Live Data Feed — Coinbase L2](#7-live-data-feed--coinbase-l2)
8. [FastAPI Backend](#8-fastapi-backend)
9. [React/TypeScript Frontend](#9-reacttypescript-frontend)
10. [Deployment Infrastructure](#10-deployment-infrastructure)
11. [Test Suite](#11-test-suite)
12. [Mathematical Reference](#12-mathematical-reference)

---

## 1. Project Overview

This platform is a research-grade implementation of the core market microstructure toolkit used in quantitative finance. It connects directly to Coinbase's BTC-USD Level 2 order book feed and computes four real-time microstructure signals derived from academic literature, while simultaneously solving and displaying the optimal market-making quoting policy from a Hamilton-Jacobi-Bellman PDE.

The goal is to demonstrate in a single running system:

- **Toxicity detection** — identifying when informed traders are active
- **Price impact modeling** — quantifying how order flow moves prices
- **Optimal execution** — computing theoretically optimal bid/ask spreads given inventory and risk aversion
- **Arrival intensity** — modeling the self-exciting nature of order flow

Everything from the signal math to the WebSocket streaming to the React dashboard was built from scratch with direct reference to the original academic papers.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Sources                            │
│                                                                 │
│   Coinbase Advanced Trade WSS ──┐                               │
│   wss://advanced-trade-ws.      │                               │
│   coinbase.com  (BTC-USD L2)    │                               │
│                                 ▼                               │
│   SyntheticLOB (fallback) ──► SharedState (thread-safe)        │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────┐
│                      FastAPI Backend  (Railway)                 │
│                                                                 │
│   Signal computation per tick:                                  │
│     _tick_signals() → OFI, VPIN, Kyle λ (incremental O(1))     │
│                                                                 │
│   HJB solver: runs once at startup                              │
│     _BID_SP, _ASK_SP = HJBSolverPython.solve()                 │
│                                                                 │
│   Endpoints:                                                    │
│     GET  /api/state   → full market snapshot (REST)             │
│     GET  /api/hjb     → recompute HJB for ?q=&gamma=           │
│     WS   /ws          → streams state every 250ms              │
└──────────────────────────────────┬──────────────────────────────┘
                                   │  WebSocket (wss://)
                                   │  4 frames/second
┌──────────────────────────────────▼──────────────────────────────┐
│                    React Frontend  (Vercel)                     │
│                                                                 │
│   useMarketWebSocket()  → live state every 250ms               │
│   useHJB(q, γ)          → debounced REST call on slider drag   │
│                                                                 │
│   4 chart panels:                                               │
│     LOBDepth   — custom SVG bid/ask depth chart                │
│     SignalChart — Z-scored OFI/VPIN/Kyle λ + mid price         │
│     HJBSpread  — optimal δ*(q) curves                         │
│     PnLPanel   — cumulative P&L + Hawkes intensity λ(t)        │
└─────────────────────────────────────────────────────────────────┘
```

**Threading model:** The Coinbase WebSocket and synthetic replay each run in daemon threads. Signal computation (`_tick_signals`) mutates shared deques under a `threading.Lock`. The FastAPI async event loop reads from shared state without holding the lock long — it copies lists out while locked, then serializes to JSON outside the lock.

---

## 3. Signal Library

All four signals are implemented in `python/signals/`. Each inherits from `BaseSignal` (abstract base class in `ofi.py`) which enforces a `compute(snapshot: pd.DataFrame) → pd.Series` interface. The server uses faster incremental versions of all three computationally-heavy signals.

### 3.1 Order Flow Imbalance (OFI)

**Reference:** Cont, Kukanov & Stoikov (2014), *Journal of Financial Econometrics* 12(1):47-88

**Core idea:** At each tick, the LOB either deepens or shallows at each level. OFI measures the net signed volume flow across bid and ask sides — it is the best single predictor of short-term mid-price movement discovered in the empirical microstructure literature, with reported R² of 5–15% at 1-minute horizons.

**Math — per-level OFI event (eqs. 2–3, p. 51):**

```
e_n(bid) = +V_n^b  if P_n^b ≥ P_{n-1}^b   (bid strengthened)
           −V_n^b  if P_n^b < P_{n-1}^b    (bid weakened)
           −V_{n-1}^b  if P_n^b ≤ P_{n-1}^b  (prev volume consumed)

e_n(ask) = −V_n^a  if P_n^a ≤ P_{n-1}^a   (ask weakened)
           +V_n^a  if P_n^a > P_{n-1}^a    (ask strengthened)
           +V_{n-1}^a  if P_n^a ≥ P_{n-1}^a  (prev volume still there)

OFI_n = Σ_{l=1}^{L} [e_n^l(bid) + e_n^l(ask)]
```

**Normalization:** The raw OFI can have arbitrary scale. We normalize over a rolling window W:

```
ŌFIP(W) = Σ_{n=t-W+1}^{t} OFI_n  /  (Σ |OFI_n| + ε)  ∈ [−1, +1]
```

This makes OFI interpretable: +1 means all recent flow is buy-side, −1 means all sell-side.

**Implementation detail — incremental server version:**  
`python/api/server.py:_tick_signals()` maintains running sums `_ofi_sum` and `_ofi_abs_sum`. Each tick it adds the new value and subtracts the oldest value from the deque — O(1) per tick with no array allocation.

**Price impact regression:**  
`OFISignal.price_impact_beta()` implements OLS: ΔP_mid = α + β·ŌFIP + ε, returning β (price impact coefficient), α, R², and n_obs.

---

### 3.2 VPIN — Volume-Synchronized PIN

**Reference:** Easley, López de Prado & O'Hara (2012), *Review of Financial Studies* 25(5):1457-1493

**Core idea:** The classical PIN (Probability of Informed Trading) model requires latent variable estimation. VPIN replaces calendar-time sampling with volume-time sampling, making it computable in real time. It measures the fraction of volume that is "toxic" — coming from informed traders who systematically trade against market makers.

**Math — bulk volume classification (eq. 11, p. 1467):**

Each bucket contains exactly `V` units of volume. The price change within the bucket determines how much was buy-initiated vs. sell-initiated:

```
V_buy(τ) = V · Φ(ΔP_τ / σ_ΔP)
V_sell(τ) = V − V_buy(τ)

where Φ is the standard normal CDF and σ_ΔP is the rolling std of ΔP.
```

**VPIN estimate (eq. 12, p. 1468):**

```
VPIN = (1/n) · Σ_{τ=t-n+1}^{t} |V_buy(τ) − V_sell(τ)| / V  ∈ [0, 1]
```

where `n = 50` trailing buckets by default. High VPIN (> 0.7) indicates the order flow is heavily one-sided — a signal that informed traders are present and adverse selection risk is elevated.

**Interpretation:**
- VPIN ≈ 0.5 → balanced buy/sell flow, safe to quote tight spreads
- VPIN > 0.7 → toxic flow, widen spreads or pull quotes
- VPIN spike precedes crashes — Easley et al. showed VPIN hit 0.93 before the May 6, 2010 Flash Crash

**Platform config:** `bucket_size=500`, `n_buckets=20`

---

### 3.3 Kyle's Lambda

**Reference:** Kyle (1985), *Econometrica* 53(6):1315-1335

**Core idea:** In Kyle's sequential trade model, prices adjust linearly to order flow. The slope λ (lambda) is the market's price impact coefficient — how much prices move per unit of net order flow. A high λ means the market is thin and orders move prices a lot.

**Model (eq. 5, p. 1319):**

```
ΔP = λ · q + ε
```

where q is signed order flow (positive = net buying) and ΔP is the price change.

**Rolling OLS estimator:**

```
λ̂(t) = Σ_{s=t-W+1}^{t} q_s · ΔP_s  /  Σ_{s=t-W+1}^{t} q_s²
```

This closed-form estimator avoids matrix inversion and runs in O(1) per tick using the cumulative sum trick implemented in `KyleLambdaSignal.compute_from_arrays()`.

**Intraday pattern:** λ exhibits a U-shape — high at open (wide spreads, thin book), falling at midday, rising again at close. The `intraday_lambda()` method bins λ by time-of-day to visualize this pattern.

**Units:** $/share (price impact per unit order flow)

---

### 3.4 Hawkes Process

**Reference:** Bacry, Mastromatteo & Muzy (2015), *Market Microstructure and Liquidity* 1(1):1550005

**Core idea:** Order arrivals are not Poisson — they self-excite. Each trade makes subsequent trades more likely for a period that decays exponentially. The Hawkes process captures this clustering.

**Intensity function (eq. 1, p. 4):**

```
λ(t) = μ + α · Σ_{t_i < t} exp(−β(t − t_i))
```

- μ: baseline intensity (exogenous arrivals, events/sec)
- α: excitation amplitude (how much each event triggers more)
- β: decay rate (how quickly excitation fades)
- α/β: branching ratio — fraction of events that are endogenous

**Log-likelihood for MLE (eq. 5, p. 8):**

```
L(μ,α,β) = −μT − (α/β)·Σᵢ[1−exp(−β(T−tᵢ))] + Σᵢ log(μ + α·Rᵢ)
```

**O(N) recursive Rᵢ (eq. 6, p. 8):**

```
R₁ = 0
Rᵢ = (Rᵢ₋₁ + 1) · exp(−β · (tᵢ − tᵢ₋₁))
```

This recursion reduces the naïve O(N²) log-likelihood to O(N).

**Fitting:** L-BFGS-B on log-transformed parameters (to enforce μ,α,β > 0) with a soft stationarity penalty when α/β ≥ 1.

**Simulation:** Ogata's thinning algorithm — generates exact sample paths.

**Goodness-of-fit:** KS test on the compensated residual process. Under the true model, inter-event compensated intervals Λ(t_{i+1}) − Λ(t_i) are Exp(1). The KS statistic is returned in `HawkesResult`.

**Dashboard:** The bottom half of the P&L panel shows λ(t) computed with fixed params μ=0.5, α=0.3, β=1.0. Orange dots mark individual trade arrivals. The blue curve shows how intensity spikes after clusters of trades and decays back toward μ.

---

## 4. HJB Optimal Market-Making

**References:**  
- Avellaneda & Stoikov (2008), *Quantitative Finance* 8(3):217-224  
- Cartea, Jaimungal & Penalva (2015), *Algorithmic and High-Frequency Trading*, CUP, Chapter 4

### Problem Setup

A market maker quotes bid and ask prices continuously. Their wealth evolves as:

```
dX = (S + δ_ask) · dN_ask − (S − δ_bid) · dN_bid
dq = dN_bid − dN_ask
dS = σ dW
```

where X is cash, q is inventory, S is mid-price, and δ_bid, δ_ask are the half-spreads. Fill rates (Poisson) depend on the quoted spread:

```
λ_ask(δ) = A · exp(−k · δ)
λ_bid(δ) = A · exp(−k · δ)
```

The market maker maximizes expected CARA utility of terminal wealth:

```
V(t, x, q, S) = max_{δ_bid, δ_ask} E[−exp(−γ · (X_T + q_T · S_T)) | X_t=x, q_t=q, S_t=S]
```

### Value Function Decomposition

Under the ansatz V(t,x,q,S) = −exp(−γ(x + qS + h(t,q))) the problem reduces to a finite-dimensional PDE for h(t,q) — the "excess value" over the trivial mark-to-market (CJP eq. 4.13):

```
∂h/∂t = φq²σ²
        − (A/ke) · exp(k·(h(t,q+1) − h(t,q)))    [bid fill contribution]
        − (A/ke) · exp(k·(h(t,q−1) − h(t,q)))    [ask fill contribution]

Terminal condition:  h(T, q) = −κ·q²
```

The φq²σ² term penalizes inventory risk. The exponential terms represent the Poisson fill rate intensity weighted by the value of an additional trade.

### Backward Euler Solution

`HJBSolverPython.solve_value_function()` solves this on a grid (Nt × (2·Q_max+1)) via backward Euler:

```
h[t][q] = h[t+1][q] + dt · (φq²σ² − A_bid[q] − A_ask[q])
```

The grid has Nt=80 time steps and inventory states q ∈ [−10, +10].

### Optimal Spreads

Once h is known, the optimal spreads follow by first-order conditions (CJP eq. 4.17):

```
δ*_bid(t,q) = max(0,  1/k − (h(t,q+1) − h(t,q)))
δ*_ask(t,q) = max(0,  1/k − (h(t,q−1) − h(t,q)))
```

**Intuition:** The 1/k term is the base spread (minimum viable spread for the fill rate). The h-difference terms are inventory adjustments — if you're long (q > 0), you widen the bid and narrow the ask to encourage selling.

### Closed-Form Approximation

For large k (thin book), Avellaneda-Stoikov derive an analytic formula (eq. 17):

```
δ*_bid = 1/k + (γσ²/2)(T−t) + γσ²(q − ½)(T−t)/k
δ*_ask = 1/k + (γσ²/2)(T−t) − γσ²(q + ½)(T−t)/k
```

The platform validates the numerical PDE solution against this — max deviation is typically < 0.0001 for the default config.

### Interactive Parameters

| Parameter | Symbol | Effect |
|-----------|--------|--------|
| Inventory slider | q | Shifts the δ*(q) curve left/right; current position highlighted with star |
| Risk aversion γ | γ | Higher γ → wider spreads, more aggressive inventory skew |

When γ is changed via slider, the platform calls `GET /api/hjb?q=&gamma=` which recomputes the full PDE on the server and returns the new spread curves. This is debounced 80ms to avoid thrashing.

---

## 5. C++ High-Performance Engine

The C++ layer (`cpp/`) is a production-speed implementation of the same feature computation. It's designed for latency-critical deployments where the Python signal engine would be too slow.

### `FeatureEngine`

**File:** `cpp/include/feature_engine.hpp`, `cpp/src/feature_engine.cpp`

Implements the same OFI algorithm as the Python layer, but:
- Processes 1M snapshots/second (benchmarked via nanobench)
- Uses `std::deque<double>` for O(1) rolling window updates
- Returns a feature vector of dimension `2·L + 4`:

```
[0..L-1]       per-level OFI (raw)
[L..2L-1]      per-level volume imbalance: (bid_vol − ask_vol)/(bid_vol + ask_vol)
[2L]           normalized rolling OFI  ∈ [−1, +1]
[2L+1]         quoted spread
[2L+2]         mid-price return
[2L+3]         best-bid Order Book Imbalance (OBI)
```

### `HJBSolver`

**File:** `cpp/include/hjb_solver.hpp`, `cpp/src/hjb_solver.cpp`

Direct C++ port of the Python HJB solver. The `QuotingPolicy::get_quotes(t, q)` method is the hot path — it does two array lookups (no math at quote time) after the solve is precomputed offline.

### `LOBParser`

**File:** `cpp/include/lob_parser.hpp`, `cpp/src/lob_parser.cpp`

Parses raw LOB feed data into `LOBSnapshot` structs. Handles fixed-width binary and CSV formats. `LOBSnapshot` stores up to 10 levels:

```cpp
struct PriceLevel { double price; long volume; };
struct LOBSnapshot {
    double     time;
    int        n_levels;
    PriceLevel bids[10];
    PriceLevel asks[10];
};
```

### `HawkesMLE`

**File:** `cpp/include/hawkes_mle.hpp`, `cpp/src/hawkes_mle.cpp`

C++ implementation of the O(N) recursive Hawkes MLE with L-BFGS-B via a custom gradient descent.

### Build

```bash
cd hf-microstructure
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/hf_tests          # run Catch2 test suite
./build/hf_microstructure # main binary
```

---

## 6. Synthetic Data Generator

**File:** `python/data/synthetic.py`

The `SyntheticLOB` class generates realistic LOB data where OFI has **known** predictive power — used for ground-truth validation of IC and signal correctness.

**Data-generating process:**

| Variable | Distribution |
|----------|-------------|
| Mid price | GBM: σ = 0.0001/tick |
| Spread | LogNormal(log(0.0002), 0.25) |
| Bid/ask volumes | LogNormal(log(500), 0.4) × exp(−0.5·(l−1)) per level |
| Trade arrivals | Hawkes process: μ=0.5, α=0.3, β=1.0 |
| Trade direction | P(buy\|OBI) = 0.5 + 0.3·OBI |

The OBI-dependent trade direction is the key: it hardwires a causal path OFI → price movement, giving expected IC ≈ 0.10–0.15 at 5-tick horizons. This means test assertions like "OFI IC > 0.01, p < 0.01" pass against synthetic data as a mathematical guarantee.

When Coinbase is unavailable (offline, SSL failure, market closed), the server falls back to synthetic data seamlessly. The synthetic thread backs off to 0.5s per tick (from 0.025s) when the live feed is active.

---

## 7. Live Data Feed — Coinbase L2

**File:** `python/api/server.py:_coinbase_thread()`

Connects to `wss://advanced-trade-ws.coinbase.com` using the Coinbase Advanced Trade WebSocket API.

**Subscription message:**
```json
{
  "type": "subscribe",
  "product_ids": ["BTC-USD"],
  "channel": "level2"
}
```

**Update protocol:** Each `l2_data` message contains a list of price level updates with `side` (bid/offer), `price_level`, and `new_quantity`. Quantity of 0 means the level was removed. The thread maintains `_S.cb_bids` and `_S.cb_asks` as sorted dicts and rebuilds the top-5 snapshot on each update.

**Failover logic:**

```
Coinbase connects  →  _S.source = "live: Coinbase BTC-USD"
Coinbase drops     →  on_close sets _S.source = "synthetic"
run_forever(reconnect=5)  →  auto-reconnects every 5s
Synthetic thread   →  runs in parallel, pauses when live feed is active
```

---

## 8. FastAPI Backend

**File:** `python/api/server.py`

### State Machine

`SharedState` is the central data store. All 600-tick rolling buffers live here:

```python
mid_prices, spreads           # price series
bid_pxs[5], bid_vols[5]       # LOB depth (5 levels)
ask_pxs[5], ask_vols[5]
ofi_vals, vpin_vals, kyle_vals  # signals
pnl, pnl_total, n_fills       # simulated P&L
alerts                        # ["VPIN_HIGH", "OFI_BUY", ...]
event_times                   # for Hawkes display
```

### Signal Computation

`_tick_signals(row, prev_row)` runs on every incoming tick (from either Coinbase or synthetic). It is designed for O(1) amortized cost:

- **OFI:** maintains `_ofi_sum` and `_ofi_abs_sum` as running totals, pops oldest value from `_ofi_history` deque each tick
- **VPIN:** accumulates volume into a bucket, closes bucket when full, appends to `_vpin_buckets` deque
- **Kyle λ:** maintains `_kyle_qp` (Σ q·ΔP) and `_kyle_q2` (Σ q²) deques, divides

### Simulated P&L

The server simulates P&L from HJB market-making at each tick:

```python
fill_prob = exp(−k · min(δ*_bid, δ*_ask))
pnl_tick  = (δ*_bid + δ*_ask)/2 − 2·|ΔP|   if fill occurs
```

This is a simplified model: earn half the spread on fills, lose twice the price move on adverse selection. It's not meant to be realistic — it demonstrates the spread/adverse-selection tradeoff visually.

### WebSocket Streaming

`/ws` streams `_build_state_dict()` at 250ms intervals. The state dict serializes ~8KB of JSON per frame (400 ticks of history × 4 signal arrays). Each connected client receives an independent stream via `ConnectionManager`.

---

## 9. React/TypeScript Frontend

**Stack:** React 18, TypeScript, Vite, Tailwind CSS, Recharts, Lucide icons

### Component Tree

```
App
├── Header          — tick counter, WS connection badge, source badge
├── StatsBar        — 9 live metrics with color thresholds
├── AlertBanner     — VPIN_HIGH / OFI_BUY / OFI_SELL badges
├── Controls        — custom range sliders for q and γ
└── main grid (2×2)
    ├── LOBDepth    — custom SVG, no charting library
    ├── SignalChart — Recharts ComposedChart, dual-axis
    ├── HJBSpread   — Recharts LineChart with ReferenceLine
    └── PnLPanel    — Area chart (P&L) + Line chart (Hawkes λ)
```

### Data Flow

```
useMarketWebSocket()
  → WebSocket to Railway backend
  → parses MarketState every 250ms
  → setState(data) triggers re-render

useHJB(q, gamma)
  → debounced fetch /api/hjb?q=&gamma=
  → returns HJBData (spread curves for all inventory levels)
  → triggers HJBSpread and LOBDepth re-render only
```

### LOBDepth — Custom SVG

The LOB depth chart is written in raw SVG (not Recharts) for precise control. Bid bars extend left from the mid-price axis, ask bars extend right. Bar width is proportional to volume, bar height is proportional to price-level spacing. HJB quote lines are rendered as dashed horizontal lines with annotations.

### Signal Z-Scoring

Before plotting, all three signals are Z-scored in `SignalChart`:

```typescript
function zscore(arr: number[]) {
  const mu  = mean(arr)
  const std = stddev(arr)
  return std > 1e-10 ? arr.map(v => (v - mu) / std) : arr.map(v => v - mu)
}
```

This puts OFI (unitless, ∈ [−1,+1]), VPIN (∈ [0,1]), and Kyle λ ($/share) on the same axis. Reference bands at ±0.7σ mark the alert thresholds.

### Alert System

Alerts are generated server-side and sent in the WebSocket payload. The frontend renders them as dismissible badges:

| Alert | Condition | Color |
|-------|-----------|-------|
| VPIN_HIGH | vpin > 0.75 | Red |
| OFI_BUY | ofi > 0.7 | Blue |
| OFI_SELL | ofi < −0.7 | Red |

---

## 10. Deployment Infrastructure

### Split Architecture

| Service | Host | Role |
|---------|------|------|
| Backend | Railway | FastAPI, Coinbase WS thread, signal computation, HJB solver |
| Frontend | Vercel | React SPA, static assets, CDN delivery |

The two are decoupled: Vercel rebuilds the frontend on every push; Railway rebuilds the backend. The only coupling is `VITE_API_URL` — a Vercel environment variable baked into the frontend build pointing to the Railway domain.

### Dockerfile

Two-stage approach avoided (frontend is on Vercel). The backend Dockerfile is minimal:

```dockerfile
FROM python:3.12-slim
RUN apt-get install -y build-essential
COPY requirements.txt . && pip install -r requirements.txt
COPY python/ ./python/
CMD uvicorn python.api.server:app --host 0.0.0.0 --port ${PORT:-8000}
```

Shell-form `CMD` (not JSON array) is required so Railway's injected `$PORT` environment variable expands at runtime.

### Dependencies (Production)

```
numpy==1.26.4          numerical arrays, OFI/VPIN math
pandas==2.2.0          DataFrame interface for batch signal computation
scipy==1.12.0          norm.cdf (VPIN), minimize (Hawkes MLE)
fastapi==0.111.0       async web framework
uvicorn[standard]      ASGI server with WebSocket support
websockets==12.0       WebSocket client/server protocol
websocket-client==1.8.0  Coinbase WS connection (websocket-client library)
```

---

## 11. Test Suite

**File:** `tests/`

### Python Tests

| File | Tests | What it validates |
|------|-------|-------------------|
| `test_ofi.py` | 5 | OFI bounds, positive on bid increase, negative on bid drop, multi-level variance reduction, IC > 0.01 on synthetic data, zero on flat LOB |
| `test_vpin.py` | 5 | VPIN ∈ [0,1], ≈0.5 for balanced flow, > 0.5 for informed flow, bucket accumulation accuracy, tick-level series correctness |
| `test_kyle_lambda.py` | — | Rolling OLS correctness, U-shape intraday pattern |
| `test_hjb_python.py` | — | Closed-form vs. numerical convergence, spread monotonicity |

**Key test — IC on synthetic data (`test_ofi.py:test_ofi_ic_exceeds_threshold_on_synthetic`):**

```python
ic, pval = pearsonr(sig.loc[common], fwd.loc[common])
assert abs(ic) > 0.01 and pval < 0.01
```

This test is a mathematical guarantee: because `SyntheticLOB` hardwires OBI → price, any correct OFI implementation must pass it.

### C++ Tests (Catch2)

| File | Tests | What it validates |
|------|-------|-------------------|
| `test_feature_engine.cpp` | 6 | OFI=0 on unchanged LOB, OFI>0 on large bid order, OFI<0 on bid drop, feature vector dimension, normalized OFI ∈ [−1,1], 1M snapshots/sec performance |
| `test_hjb_solver.cpp` | — | Spread non-negativity, monotonicity in inventory, closed-form convergence |
| `test_hawkes_mle.cpp` | — | Log-likelihood correctness, MLE recovery of known parameters |

---

## 12. Mathematical Reference

### OFI Price Impact (Cont et al. 2014)

```
ΔP_mid = α + β · ŌFIP(W) + ε

where β ≈ bid-ask spread × 0.3  (empirical, US equities)
      R² ≈ 5–15% at 1-minute horizon
```

### VPIN Toxicity Classification

```
VPIN = E[|2V_buy/V − 1|]

where V_buy = V · Φ(ΔP_bucket / σ_ΔP)

Threshold:  VPIN > 0.7 → elevated adverse selection risk
```

### Kyle Lambda

```
λ̂ = Cov(ΔP, q) / Var(q)  =  Σ(q·ΔP) / Σ(q²)

Typical BTC-USD: λ ≈ 10⁻⁶ to 10⁻⁵ $/share
Tick-by-tick values will be much smaller for liquid markets.
```

### Hawkes Branching Ratio

```
n = α/β  ∈ [0, 1)  for stationarity

n → 0:  nearly Poisson, arrivals are independent
n → 1:  near-critical, strong clustering / momentum
```

### HJB Optimal Spread Summary

```
δ*_total = δ*_bid + δ*_ask ≈ 2/k + γσ²(T−t)

The γσ²(T−t) term is the "inventory risk premium" — additional spread
demanded to compensate for holding inventory risk to session end.

At t → T:  spreads collapse to 2/k  (pure fill-rate minimum)
At t = 0:  maximum spreads, maximum risk aversion impact
```

### Configuration Parameters (Default)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Q_max | 10 | Max ±10 units inventory |
| Nt | 80 | 80 time steps in [0, T] |
| T | 1/6.5 years | One 6.5-hour trading session |
| σ | 0.001 | Mid-price diffusion per step |
| φ | 0.01 | Running inventory penalty coefficient |
| κ | 0.01 | Terminal inventory liquidation cost |
| A | 1.0 | Fill-rate constant |
| k | 1.5 | Fill-rate depth (higher k = thinner book) |
| γ | 0.01 | CARA risk aversion (interactive slider: 0.001–0.1) |
