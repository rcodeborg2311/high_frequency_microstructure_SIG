"""
hf-microstructure signal library.

Signals:
  OFISignal        – Order Flow Imbalance (Cont, Kukanov, Stoikov 2014)
  VPINSignal       – Volume-Synchronized PIN (Easley, López de Prado, O'Hara 2012)
  KyleLambdaSignal – Kyle's lambda price impact (Kyle 1985)
  HawkesSignal     – Hawkes arrival-rate / branching ratio (Bacry et al. 2015)
  SignalCombiner   – IC-weighted / PCA / Ridge combination
"""

from .ofi import OFISignal
from .vpin import VPINSignal
from .kyle_lambda import KyleLambdaSignal
from .hawkes import HawkesSignal
from .signal_combiner import SignalCombiner

__all__ = [
    "OFISignal",
    "VPINSignal",
    "KyleLambdaSignal",
    "HawkesSignal",
    "SignalCombiner",
]
