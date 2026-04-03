"""
Data loaders for FI-2010 and LOBSTER LOB datasets.

FI-2010 format (10-level LOB, 5 Finnish stocks, 9 trading days):
    File: NOKIA_2010-06-14_34200000_57600000_orderbook_10.csv
    Columns: Time, Ask1, AskVol1, ..., Ask10, AskVol10,
             Bid1, BidVol1, ..., Bid10, BidVol10
    Prices: integer units (divide by 10000 for actual price)
    Time:   seconds from midnight

LOBSTER format:
    Orderbook file: Ask1, AskVol1, Bid1, BidVol1, Ask2, ...
    Message file:   Time, EventType, OrderID, Size, Price, Direction
    EventTypes: 1=submission, 2=cancellation, 3=deletion,
                4=visible execution, 5=hidden execution
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd


class DataLoader:
    """
    Load and normalize LOB data from FI-2010 and LOBSTER CSV files.

    The returned DataFrames use a standardized schema:
        - Columns: bid_price_{l}, bid_vol_{l}, ask_price_{l}, ask_vol_{l}
          for l ∈ {1, ..., n_levels}.
        - Additional: mid_price, spread.
        - Index: time (seconds from midnight, or DatetimeIndex if available).
    """

    N_LEVELS    = 10
    PRICE_SCALE = 1.0 / 10_000.0  # FI-2010 integer price → float

    # ── FI-2010 ──────────────────────────────────────────────────────────────

    def load_fi2010(
        self,
        filepath: str,
        n_levels: int = 10,
        normalize_prices: bool = True,
    ) -> pd.DataFrame:
        """
        Load a FI-2010 orderbook CSV file.

        Parameters
        ----------
        filepath         : Path to the CSV file.
        n_levels         : Number of LOB levels to load (1..10).
        normalize_prices : Divide integer prices by 10_000.

        Returns
        -------
        DataFrame indexed by ``time`` (seconds from midnight).
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"FI-2010 file not found: {filepath}")

        # FI-2010 has no header
        raw = pd.read_csv(filepath, header=None)

        scale = self.PRICE_SCALE if normalize_prices else 1.0
        records = []

        for _, row in raw.iterrows():
            rec: dict = {"time": row.iloc[0]}
            for l in range(1, n_levels + 1):
                base = 1 + (l - 1) * 4
                rec[f"ask_price_{l}"] = row.iloc[base + 0] * scale
                rec[f"ask_vol_{l}"]   = row.iloc[base + 1]
                rec[f"bid_price_{l}"] = row.iloc[base + 2] * scale
                rec[f"bid_vol_{l}"]   = row.iloc[base + 3]
            records.append(rec)

        df = pd.DataFrame(records).set_index("time")
        df["mid_price"] = (df["bid_price_1"] + df["ask_price_1"]) * 0.5
        df["spread"]    = df["ask_price_1"] - df["bid_price_1"]
        return df

    # ── LOBSTER ──────────────────────────────────────────────────────────────

    def load_lobster(
        self,
        ob_file:  str,
        msg_file: str,
        n_levels: int = 10,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load a LOBSTER orderbook + message pair.

        Parameters
        ----------
        ob_file  : Path to the *_orderbook_N.csv file.
        msg_file : Path to the *_message_N.csv file.
        n_levels : Number of LOB levels to return.

        Returns
        -------
        (orderbook_df, messages_df)
        """
        # Messages: Time, EventType, OrderID, Size, Price, Direction
        msg_cols = ["time", "event_type", "order_id", "size", "price", "direction"]
        msg_df   = pd.read_csv(msg_file, header=None, names=msg_cols)
        msg_df["price"] /= 10_000.0

        # Orderbook: Ask1, AskVol1, Bid1, BidVol1, Ask2, ...
        ob_raw = pd.read_csv(ob_file, header=None)

        ob_records = []
        for row_idx, row in ob_raw.iterrows():
            rec: dict = {}
            for l in range(1, n_levels + 1):
                base = (l - 1) * 4
                rec[f"ask_price_{l}"] = row.iloc[base + 0] / 10_000.0
                rec[f"ask_vol_{l}"]   = row.iloc[base + 1]
                rec[f"bid_price_{l}"] = row.iloc[base + 2] / 10_000.0
                rec[f"bid_vol_{l}"]   = row.iloc[base + 3]
            ob_records.append(rec)

        ob_df = pd.DataFrame(ob_records)

        # Assign timestamps from message file
        n_rows  = min(len(ob_df), len(msg_df))
        ob_df   = ob_df.iloc[:n_rows].copy()
        ob_df["time"] = msg_df["time"].values[:n_rows]
        ob_df = ob_df.set_index("time")

        ob_df["mid_price"] = (ob_df["bid_price_1"] + ob_df["ask_price_1"]) * 0.5
        ob_df["spread"]    = ob_df["ask_price_1"] - ob_df["bid_price_1"]

        msg_df = msg_df.set_index("time")

        return ob_df, msg_df

    # ── Synthetic ─────────────────────────────────────────────────────────────

    def generate_synthetic(
        self,
        n_steps: int = 500_000,
        seed:    int = 42,
    ) -> pd.DataFrame:
        """
        Generate synthetic LOB data with known OFI → price relationship.

        Mid price: geometric Brownian motion, σ = 0.0001 per step.
        Spread:    drawn from LogNormal(μ=log(0.0002), σ=0.00005·scale).
        Quantities: log-normal, level l gets exp(−0.5·(l−1)) × V₁.
        Trades:     Hawkes process with μ=0.5/s, α=0.3, β=1.0.
        Trade direction: P(buy) = 0.5 + 0.3·OBI (OBI = order book imbalance).

        This dataset has a ground-truth OFI predictive relationship so that
        IC tests with known threshold can pass.

        Returns
        -------
        DataFrame with standardized LOB schema plus 'order_flow', 'volume',
        'trade_direction', 'event'.
        """
        from .synthetic import SyntheticLOB
        gen = SyntheticLOB(n_levels=5, seed=seed)
        return gen.generate(n_steps)


# ── Command-line: generate and preview synthetic data ───────────────────────

if __name__ == "__main__":
    loader = DataLoader()
    df     = loader.generate_synthetic(n_steps=10_000)
    print(f"Generated {len(df)} synthetic LOB ticks")
    print(df.describe().to_string())
