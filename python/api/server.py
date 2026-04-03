"""
FastAPI WebSocket server for HF Market Microstructure Signal Platform.

Exposes:
  WS  /ws              — streams MarketState every 250ms
  GET /api/hjb         — compute HJB optimal quotes for ?q=0&gamma=0.01
  GET /api/state       — latest snapshot (REST)
  Static /             — serves React frontend build
"""

from __future__ import annotations

import asyncio
import collections
import json
import sys
import threading
import time
from pathlib import Path
from typing import List

import numpy as np
from scipy.stats import norm as _norm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


from python.data.synthetic import SyntheticLOB
from python.hjb.optimal_quotes import HJBSolverPython, HJBConfig

# ── HJB (pre-solved once at startup) ──────────────────────────────────────────
_HJB_CFG  = HJBConfig(Nt=80, Q_max=10, sigma=0.001, phi=0.01, k=1.5, gamma=0.01)
_HJB      = HJBSolverPython(_HJB_CFG)
_BID_SP, _ASK_SP, _H = _HJB.solve()

# ── Shared state ───────────────────────────────────────────────────────────────
MAXLEN = 600

class SharedState:
    def __init__(self):
        self.lock         = threading.Lock()
        self.mid_prices   = collections.deque(maxlen=MAXLEN)
        self.spreads      = collections.deque(maxlen=MAXLEN)
        self.bid_vols     = [collections.deque(maxlen=MAXLEN) for _ in range(5)]
        self.ask_vols     = [collections.deque(maxlen=MAXLEN) for _ in range(5)]
        self.bid_pxs      = [collections.deque(maxlen=MAXLEN) for _ in range(5)]
        self.ask_pxs      = [collections.deque(maxlen=MAXLEN) for _ in range(5)]
        self.ofi_vals     = collections.deque(maxlen=MAXLEN)
        self.vpin_vals    = collections.deque(maxlen=MAXLEN)
        self.kyle_vals    = collections.deque(maxlen=MAXLEN)
        self.event_times  = collections.deque(maxlen=200)
        self.pnl          = collections.deque(maxlen=MAXLEN)
        self.pnl_total    = 0.0
        self.n_fills      = 0
        self.tick         = 0
        self.source       = "synthetic"
        self.alerts: collections.deque[str] = collections.deque(maxlen=10)
        self.cb_bids: dict[float, float] = {}
        self.cb_asks: dict[float, float] = {}

_S = SharedState()

# ── Signal bookkeeping ─────────────────────────────────────────────────────────
_ofi_history     = collections.deque(maxlen=80)
_ofi_abs_sum     = 0.0
_ofi_sum         = 0.0
_kyle_qp         = collections.deque(maxlen=60)
_kyle_q2         = collections.deque(maxlen=60)
_vpin_buckets    = collections.deque(maxlen=20)
_vpin_bucket_vol = 0.0
_vpin_bucket_dp  = 0.0
_vpin_sigma_dp   = 0.001


def _tick_signals(row: dict, prev_row: dict | None) -> tuple[float, float, float]:
    global _ofi_abs_sum, _ofi_sum, _vpin_bucket_vol, _vpin_bucket_dp, _vpin_sigma_dp
    mid      = row["mid"]
    prev_mid = prev_row["mid"] if prev_row else mid
    dp       = mid - prev_mid

    # OFI
    raw_ofi = 0.0
    if prev_row:
        for l in range(5):
            pb, vb = prev_row[f"bp{l}"], prev_row[f"bv{l}"]
            pa, va = prev_row[f"ap{l}"], prev_row[f"av{l}"]
            cb, cvb = row[f"bp{l}"], row[f"bv{l}"]
            ca, cva = row[f"ap{l}"], row[f"av{l}"]
            e_bid = (cvb if cb >= pb else -cvb) - (vb if cb <= pb else 0)
            e_ask = (-cva if ca <= pa else cva) + (va if ca >= pa else 0)
            raw_ofi += e_bid + e_ask
    _ofi_history.append(raw_ofi)
    _ofi_sum     += raw_ofi
    _ofi_abs_sum += abs(raw_ofi)
    norm_ofi = float(np.clip(_ofi_sum / (_ofi_abs_sum + 1e-9), -1, 1))

    # VPIN
    vol = row.get("vol", 200.0)
    _vpin_sigma_dp = float(0.99 * _vpin_sigma_dp + 0.01 * abs(dp))
    remaining = vol
    vpin_val  = float(np.mean(_vpin_buckets)) if _vpin_buckets else 0.5
    while remaining > 1e-6:
        space = 500 - _vpin_bucket_vol
        fill  = min(remaining, space)
        _vpin_bucket_dp  += dp * fill / 500
        _vpin_bucket_vol += fill
        remaining        -= fill
        if _vpin_bucket_vol >= 499.9:
            z    = _vpin_bucket_dp / max(_vpin_sigma_dp, 1e-9)
            vb   = 500 * float(_norm.cdf(z))
            _vpin_buckets.append(abs(vb - (500 - vb)) / 500)
            vpin_val = float(np.mean(_vpin_buckets))
            _vpin_bucket_vol = 0.0
            _vpin_bucket_dp  = 0.0

    # Kyle λ
    q = row.get("order_flow", np.sign(dp) * vol)
    _kyle_qp.append(q * dp)
    _kyle_q2.append(q * q)
    sum_q2   = sum(_kyle_q2)
    kyle_val = float(sum(_kyle_qp) / sum_q2) if sum_q2 > 1e-12 else 0.0

    return norm_ofi, vpin_val, kyle_val


# ── Synthetic data thread ──────────────────────────────────────────────────────

def _synthetic_thread():
    gen = SyntheticLOB(n_levels=5, seed=42)
    df  = gen.generate(n_steps=100_000)
    prev_row: dict | None = None
    prev_mid = None
    tick_time = 0.0

    for idx in range(len(df)):
        row_s = df.iloc[idx]
        mid   = float(row_s["mid_price"])
        vol   = float(abs(row_s.get("order_flow", 200)))
        dp    = mid - prev_mid if prev_mid is not None else 0.0

        row = {"mid": mid, "vol": vol, "order_flow": float(row_s.get("order_flow", 0))}
        for l in range(5):
            row[f"bp{l}"] = float(row_s[f"bid_price_{l+1}"])
            row[f"bv{l}"] = float(row_s[f"bid_vol_{l+1}"])
            row[f"ap{l}"] = float(row_s[f"ask_price_{l+1}"])
            row[f"av{l}"] = float(row_s[f"ask_vol_{l+1}"])

        ofi, vpin, kyle = _tick_signals(row, prev_row)

        if row_s.get("event", 0):
            tick_time += 0.1
            with _S.lock:
                _S.event_times.append(tick_time)

        with _S.lock:
            t_f   = float(np.clip(idx / len(df), 0, 0.999))
            t_idx = int(t_f * _HJB_CFG.Nt)
            b_sp  = float(_BID_SP[t_idx, 10])
            a_sp  = float(_ASK_SP[t_idx, 10])

            fill_prob = float(np.exp(-_HJB_CFG.k * min(b_sp, a_sp)))
            pnl_tick  = 0.0
            if np.random.random() < fill_prob * 0.1:
                pnl_tick = (b_sp + a_sp) * 0.5 - abs(dp) * 2
                _S.n_fills += 1
            _S.pnl_total += pnl_tick
            _S.pnl.append(_S.pnl_total)

            _S.mid_prices.append(mid)
            _S.spreads.append(float(row_s["spread"]))
            _S.ofi_vals.append(ofi)
            _S.vpin_vals.append(vpin)
            _S.kyle_vals.append(kyle)
            for l in range(5):
                _S.bid_pxs[l].append(row[f"bp{l}"])
                _S.bid_vols[l].append(row[f"bv{l}"])
                _S.ask_pxs[l].append(row[f"ap{l}"])
                _S.ask_vols[l].append(row[f"av{l}"])
            _S.tick += 1
            if not _S.source.startswith("live"):
                _S.source = "synthetic"

            if vpin > 0.75 and (not _S.alerts or _S.alerts[-1] != "VPIN_HIGH"):
                _S.alerts.append("VPIN_HIGH")
            if abs(ofi) > 0.7 and (not _S.alerts or _S.alerts[-1] not in ("OFI_BUY", "OFI_SELL")):
                _S.alerts.append("OFI_BUY" if ofi > 0 else "OFI_SELL")

            is_live = _S.source.startswith("live")

        prev_row = row
        prev_mid = mid
        # Back off if Coinbase is live — keep thread alive as warm standby
        time.sleep(0.5 if is_live else 0.025)


# ── Coinbase Advanced Trade WebSocket thread ──────────────────────────────────

def _coinbase_thread():
    try:
        import websocket as _websocket
    except ImportError:
        return  # websocket-client not installed

    CB_URL = "wss://advanced-trade-ws.coinbase.com"

    def on_open(ws):
        ws.send(json.dumps({
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channel": "level2",
        }))

    def on_message(_ws, message):
        try:
            msg = json.loads(message)
            if msg.get("channel") != "l2_data":
                return

            for event in msg.get("events", []):
                for u in event.get("updates", []):
                    side = u.get("side")
                    px   = float(u.get("price_level", 0))
                    qty  = float(u.get("new_quantity", 0))
                    if side == "bid":
                        if qty == 0:
                            _S.cb_bids.pop(px, None)
                        else:
                            _S.cb_bids[px] = qty
                    elif side == "offer":
                        if qty == 0:
                            _S.cb_asks.pop(px, None)
                        else:
                            _S.cb_asks[px] = qty

            bids = sorted(_S.cb_bids.items(), reverse=True)[:5]
            asks = sorted(_S.cb_asks.items())[:5]
            if len(bids) < 1 or len(asks) < 1:
                return

            mid = (bids[0][0] + asks[0][0]) / 2.0

            with _S.lock:
                prev_mid_val = list(_S.mid_prices)[-1] if _S.mid_prices else mid
                prev_row = None
                if _S.tick > 0:
                    prev_row = {
                        "mid": prev_mid_val,
                        **{f"bp{l}": list(_S.bid_pxs[l])[-1] if _S.bid_pxs[l] else bids[min(l, len(bids)-1)][0] for l in range(5)},
                        **{f"bv{l}": list(_S.bid_vols[l])[-1] if _S.bid_vols[l] else bids[min(l, len(bids)-1)][1] for l in range(5)},
                        **{f"ap{l}": list(_S.ask_pxs[l])[-1] if _S.ask_pxs[l] else asks[min(l, len(asks)-1)][0] for l in range(5)},
                        **{f"av{l}": list(_S.ask_vols[l])[-1] if _S.ask_vols[l] else asks[min(l, len(asks)-1)][1] for l in range(5)},
                    }

            row = {
                "mid": mid,
                "vol": 200.0,
                "order_flow": np.sign(mid - (prev_row["mid"] if prev_row else mid)) * 200.0,
            }
            for l in range(5):
                row[f"bp{l}"] = bids[l][0] if l < len(bids) else bids[-1][0] - l * 10
                row[f"bv{l}"] = bids[l][1] if l < len(bids) else 0.01
                row[f"ap{l}"] = asks[l][0] if l < len(asks) else asks[-1][0] + l * 10
                row[f"av{l}"] = asks[l][1] if l < len(asks) else 0.01

            ofi, vpin, kyle = _tick_signals(row, prev_row)

            with _S.lock:
                _S.mid_prices.append(mid)
                _S.spreads.append(asks[0][0] - bids[0][0])
                _S.ofi_vals.append(ofi)
                _S.vpin_vals.append(vpin)
                _S.kyle_vals.append(kyle)
                for l in range(5):
                    _S.bid_pxs[l].append(row[f"bp{l}"])
                    _S.bid_vols[l].append(row[f"bv{l}"])
                    _S.ask_pxs[l].append(row[f"ap{l}"])
                    _S.ask_vols[l].append(row[f"av{l}"])
                # P&L: reuse HJB spread at current tick
                t_f   = float(np.clip(_S.tick / max(_S.tick + 1, 1), 0, 0.999))
                t_idx = int(t_f * _HJB_CFG.Nt)
                b_sp  = float(_BID_SP[t_idx, 10])
                a_sp  = float(_ASK_SP[t_idx, 10])
                dp    = mid - (prev_row["mid"] if prev_row else mid)
                fill_prob = float(np.exp(-_HJB_CFG.k * min(b_sp, a_sp)))
                pnl_tick  = 0.0
                if np.random.random() < fill_prob * 0.1:
                    pnl_tick = (b_sp + a_sp) * 0.5 - abs(dp) * 2
                    _S.n_fills += 1
                _S.pnl_total += pnl_tick
                _S.pnl.append(_S.pnl_total)
                _S.tick += 1
                _S.source = "live: Coinbase BTC-USD"

                if vpin > 0.75 and (not _S.alerts or _S.alerts[-1] != "VPIN_HIGH"):
                    _S.alerts.append("VPIN_HIGH")
                if abs(ofi) > 0.7 and (not _S.alerts or _S.alerts[-1] not in ("OFI_BUY", "OFI_SELL")):
                    _S.alerts.append("OFI_BUY" if ofi > 0 else "OFI_SELL")

        except Exception:
            pass

    def on_error(_ws, _error):
        pass  # retry handled by run_forever reconnect logic

    def on_close(_ws, _code, _msg):
        with _S.lock:
            if _S.source.startswith("live"):
                _S.source = "synthetic"

    ws_app = _websocket.WebSocketApp(
        CB_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    import ssl
    ws_app.run_forever(
        ping_interval=20,
        reconnect=5,
        sslopt={"cert_reqs": ssl.CERT_NONE},
    )


# ── Start data threads ─────────────────────────────────────────────────────────
_syn_thread = threading.Thread(target=_synthetic_thread, daemon=True)
_syn_thread.start()

_cb_thread = threading.Thread(target=_coinbase_thread, daemon=True)
_cb_thread.start()

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="HF Microstructure API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_state_dict(window: int = 400) -> dict:
    """Build JSON-serializable state snapshot."""
    with _S.lock:
        mids     = list(_S.mid_prices)
        ofis     = list(_S.ofi_vals)
        vpins    = list(_S.vpin_vals)
        kyles    = list(_S.kyle_vals)
        pnls     = list(_S.pnl)
        source   = _S.source
        tick     = _S.tick
        alerts   = list(_S.alerts)
        n_fills  = _S.n_fills
        bid_vols = [list(d) for d in _S.bid_vols]
        ask_vols = [list(d) for d in _S.ask_vols]
        bid_pxs  = [list(d) for d in _S.bid_pxs]
        ask_pxs  = [list(d) for d in _S.ask_pxs]
        ev_times = list(_S.event_times)

    if not mids:
        return {"tick": 0, "source": source, "ready": False}

    lo = max(0, len(mids) - window)

    # Rolling Sharpe
    pnl_arr = np.array(pnls, dtype=float)
    sharpe  = 0.0
    if len(pnl_arr) > 20:
        inc = np.diff(pnl_arr[-200:])
        if np.std(inc) > 0:
            sharpe = float(np.mean(inc) / np.std(inc) * np.sqrt(252 * 390 * 4))

    lob_bids = []
    lob_asks = []
    for l in range(5):
        if bid_pxs[l] and bid_vols[l]:
            lob_bids.append({"px": bid_pxs[l][-1], "vol": bid_vols[l][-1]})
        if ask_pxs[l] and ask_vols[l]:
            lob_asks.append({"px": ask_pxs[l][-1], "vol": ask_vols[l][-1]})

    def _safe(arr, default=0.0):
        return float(arr[-1]) if arr else default

    def _clip_nan(v):
        import math
        return 0.0 if (math.isnan(v) or math.isinf(v)) else v

    return {
        "tick":    tick,
        "source":  source,
        "ready":   True,
        "mid":     _safe(mids),
        "spread":  _safe(list(_S.spreads)),
        "ofi":     _clip_nan(_safe(ofis)),
        "vpin":    _safe(vpins, 0.5),
        "kyle":    _clip_nan(_safe(kyles)),
        "pnl":     _safe(pnls),
        "n_fills": n_fills,
        "sharpe":  _clip_nan(sharpe),
        "fill_rate": n_fills / max(tick, 1) * 100,
        "mids":    [float(x) for x in mids[lo:]],
        "ofis":    [_clip_nan(float(x)) for x in ofis[lo:]],
        "vpins":   [float(x) for x in vpins[lo:]],
        "kyles":   [_clip_nan(float(x)) for x in kyles[lo:]],
        "pnls":    [float(x) for x in pnls[lo:]],
        "lob":     {"bids": lob_bids, "asks": lob_asks},
        "alerts":  alerts,
        "event_times": [float(t) for t in ev_times],
    }


def _compute_hjb(q: int, gamma: float) -> dict:
    cfg = HJBConfig(
        Nt=80, Q_max=10, sigma=0.001, phi=0.01,
        k=1.5, gamma=float(gamma)
    )
    hjb = HJBSolverPython(cfg)
    bid_sp, ask_sp, _ = hjb.solve()

    with _S.lock:
        tick = _S.tick

    Q_max  = cfg.Q_max
    t_frac = min(tick / max(tick + 1, 1), 0.999)
    t_idx  = int(t_frac * cfg.Nt)
    qi     = int(np.clip(q + Q_max, 0, 2 * Q_max))

    q_arr  = list(range(-Q_max, Q_max + 1))
    b_row  = bid_sp[t_idx].tolist()
    a_row  = ask_sp[t_idx].tolist()

    # Mask boundary infinities
    b_masked = [v if v < 1e5 else None for v in b_row]
    a_masked = [v if v < 1e5 else None for v in a_row]

    cur_bid = float(bid_sp[t_idx, qi]) if bid_sp[t_idx, qi] < 1e5 else None
    cur_ask = float(ask_sp[t_idx, qi]) if ask_sp[t_idx, qi] < 1e5 else None

    return {
        "q_arr":        q_arr,
        "bid_spreads":  b_masked,
        "ask_spreads":  a_masked,
        "cur_bid_sp":   cur_bid,
        "cur_ask_sp":   cur_ask,
        "t_frac":       float(t_frac),
        "gamma":        float(gamma),
        "q":            int(q),
    }


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/api/state")
def get_state():
    return JSONResponse(_build_state_dict())


@app.get("/api/hjb")
def get_hjb(q: int = 0, gamma: float = 0.01):
    return JSONResponse(_compute_hjb(q, gamma))


# ── WebSocket ──────────────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: str):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


_mgr = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await _mgr.connect(ws)
    try:
        while True:
            payload = json.dumps(_build_state_dict())
            await ws.send_text(payload)
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        _mgr.disconnect(ws)
    except Exception:
        _mgr.disconnect(ws)


@app.get("/")
def health():
    return {"status": "ok", "tick": _S.tick}
