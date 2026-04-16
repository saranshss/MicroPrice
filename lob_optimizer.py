"""
lob_optimizer.py
================
Optimal stopping framework for a single sell order in an LOB simulation.

The agent sells at:
    τ = min(T, first t such that g(state_t) > f(t))

where f(t) = c0 + c1·t is a linear urgency threshold and g is a signal
function of LOB state. The selling price is pb(τ). Objective: maximize
E[pb(τ)] over (c0, c1) using mini-batch SGD with finite-difference gradients.

Classes
-------
SellPolicy      : stopping rule container (c0, c1, g, T)
PathCache       : pre-simulates N paths and caches g values for fast evaluation
PolicyOptimizer : grid scan + SGD to find optimal (c0, c1)

Module-level g functions (each takes LOBState, returns float)
-------------------------------------------------------------
g_imbalance    : qb / (qa + qb)   — bid-heavy → upward pressure → sell
g_bid_price    : pb                — sell when best bid is high
g_microprice   : I·pa + (1-I)·pb  — Stoikov micro-price signal
g_ask_pressure : qa / (qa + qb)   — ask-heavy → liquidity thin → sell now
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from order_book_simulator import LOBSimulator, LOBState


# ── g functions ────────────────────────────────────────────────────────────────

def g_imbalance(state: LOBState) -> float:
    """qb / (qa + qb) — high → bid-heavy → upward price pressure."""
    return state.imbalance()


def g_bid_price(state: LOBState) -> float:
    """pb — sell when best bid price is high."""
    return state.pb


def g_microprice(state: LOBState) -> float:
    """Stoikov micro-price: I·pa + (1-I)·pb."""
    return state.micro_price()


def g_ask_pressure(state: LOBState) -> float:
    """qa / (qa + qb) — high → ask-heavy → sell immediately."""
    total = state.qa + state.qb
    return state.qa / total if total > 0 else 0.5


# ── SellPolicy ─────────────────────────────────────────────────────────────────

@dataclass
class SellPolicy:
    """
    Stopping rule: sell at first t where g(state_t) > f(t), or at deadline T.

    Attributes
    ----------
    g     : callable (LOBState) -> float  — LOB state signal
    c0    : float — intercept of linear threshold f(t) = c0 + c1*t
    c1    : float — slope of linear threshold
    T     : float — hard deadline
    """
    g  : Callable[[LOBState], float]
    c0 : float
    c1 : float
    T  : float

    def f(self, t: float) -> float:
        """Urgency threshold at time t."""
        return self.c0 + self.c1 * t

    def should_stop(self, state: LOBState, t: float) -> bool:
        """Return True if we should sell now."""
        return t >= self.T or self.g(state) > self.f(t)

    def selling_price(self, row: pd.Series) -> float:
        """Market sell at best bid."""
        return float(row["pb"])


# ── PathCache ──────────────────────────────────────────────────────────────────

class PathCache:
    """
    Pre-simulated LOB paths with cached g values for fast policy evaluation.

    Parameters are stored as pre-extracted numpy arrays so that ``evaluate``
    is purely vectorised (no DataFrame access during optimisation).

    Attributes
    ----------
    paths   : list[pd.DataFrame] — raw simulation outputs (for plotting)
    T       : float
    n_paths : int
    """

    def __init__(
        self,
        paths: list[pd.DataFrame],
        T: float,
        times:   list[np.ndarray],
        g_vals:  list[np.ndarray],
        pb_vals: list[np.ndarray],
    ) -> None:
        self.paths   = paths
        self.T       = T
        self.n_paths = len(paths)
        self._times   = times
        self._g_vals  = g_vals
        self._pb_vals = pb_vals

    @classmethod
    def build(
        cls,
        sim_kwargs: dict,
        T: float,
        n_paths: int,
        g: Callable[[LOBState], float],
        seed: Optional[int] = None,
    ) -> "PathCache":
        """
        Simulate *n_paths* independent LOB paths up to time *T*.

        Each path uses seed+i for reproducibility. g is evaluated once per
        row and stored in a 'g_val' column and as a pre-extracted numpy array.

        Parameters
        ----------
        sim_kwargs : dict  — passed directly to LOBSimulator(**sim_kwargs)
        T          : float — time horizon
        n_paths    : int   — number of independent Monte Carlo paths
        g          : callable (LOBState) -> float
        seed       : int or None
        """
        base_seed = seed if seed is not None else 0
        paths:   list[pd.DataFrame] = []
        times:   list[np.ndarray]   = []
        g_vals:  list[np.ndarray]   = []
        pb_vals: list[np.ndarray]   = []

        for i in range(n_paths):
            sim = LOBSimulator(**sim_kwargs, seed=base_seed + i)
            df  = sim.run(T=T)

            # Compute g values row-by-row (done once at build time)
            gv = np.empty(len(df))
            for k, row in enumerate(df.itertuples(index=False)):
                state = LOBState(
                    pa=row.pa, qa=int(row.qa),
                    pb=row.pb, qb=int(row.qb),
                    time=row.time,
                )
                gv[k] = g(state)
            df = df.copy()
            df["g_val"] = gv

            paths.append(df)
            times.append(df["time"].values.copy())
            g_vals.append(gv)
            pb_vals.append(df["pb"].values.copy())

        return cls(paths, T, times, g_vals, pb_vals)

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate(self, c0: float, c1: float) -> np.ndarray:
        """
        For each cached path, find the first event where g_val > c0 + c1*time
        (or use the last row if none triggers) and return pb at that event.

        Returns
        -------
        np.ndarray of shape (n_paths,) — selling prices.
        """
        prices = np.empty(self.n_paths)
        for i in range(self.n_paths):
            t  = self._times[i]
            gv = self._g_vals[i]
            pb = self._pb_vals[i]
            triggered = gv > (c0 + c1 * t)
            idx = int(np.argmax(triggered)) if triggered.any() else len(t) - 1
            prices[i] = pb[idx]
        return prices

    def _evaluate_batch(
        self,
        c0: float,
        c1: float,
        indices: np.ndarray,
    ) -> float:
        """Mean selling price over a subset of paths (used in SGD mini-batches)."""
        total = 0.0
        for i in indices:
            t  = self._times[i]
            gv = self._g_vals[i]
            pb = self._pb_vals[i]
            triggered = gv > (c0 + c1 * t)
            idx = int(np.argmax(triggered)) if triggered.any() else len(t) - 1
            total += pb[idx]
        return total / len(indices)

    def expected_price(self, c0: float, c1: float) -> float:
        """E[pb(τ)] over all cached paths."""
        return float(self.evaluate(c0, c1).mean())


# ── PolicyOptimizer ────────────────────────────────────────────────────────────

class PolicyOptimizer:
    """
    Grid scan and SGD optimisation over the threshold parameters (c0, c1).

    Parameters
    ----------
    cache : PathCache — pre-simulated path library
    """

    def __init__(self, cache: PathCache) -> None:
        self.cache = cache

    # ── Grid scan ──────────────────────────────────────────────────────────────

    def scan(
        self,
        c0_grid: np.ndarray,
        c1_grid: np.ndarray,
    ) -> pd.DataFrame:
        """
        Evaluate E[price] at every (c0, c1) grid point.

        Parameters
        ----------
        c0_grid : 1-D array of c0 values
        c1_grid : 1-D array of c1 values

        Returns
        -------
        pd.DataFrame with columns [c0, c1, expected_price, std_price].
        """
        rows = []
        for c0 in c0_grid:
            for c1 in c1_grid:
                prices = self.cache.evaluate(float(c0), float(c1))
                rows.append({
                    "c0"            : float(c0),
                    "c1"            : float(c1),
                    "expected_price": float(prices.mean()),
                    "std_price"     : float(prices.std()),
                })
        return pd.DataFrame(rows)

    # ── SGD ────────────────────────────────────────────────────────────────────

    def sgd(
        self,
        c0_init:    float = 0.5,
        c1_init:    float = 0.0,
        lr:         float = 0.01,
        n_steps:    int   = 300,
        batch_size: int   = 64,
        epsilon:    float = 1e-3,
        lr_decay:   float = 1.0,
    ) -> dict:
        """
        Mini-batch gradient *ascent* to maximise E[pb(τ)] over (c0, c1).

        Gradients are estimated via central finite differences on mini-batches:

            ∂J/∂c0 ≈ [J_B(c0+ε, c1) − J_B(c0−ε, c1)] / 2ε

        where J_B is the mini-batch mean over *batch_size* randomly sampled paths.

        Parameters
        ----------
        c0_init    : initial c0
        c1_init    : initial c1
        lr         : initial learning rate
        n_steps    : number of SGD steps
        batch_size : paths per mini-batch
        epsilon    : finite-difference step size
        lr_decay   : multiplicative lr decay per step (1.0 = constant)

        Returns
        -------
        dict with keys:
            "c0"             : float — c0 at best observed full-data price
            "c1"             : float — c1 at best observed full-data price
            "expected_price" : float — E[price] at best (c0, c1)
            "history"        : pd.DataFrame [step, c0, c1, batch_price, full_price]
        """
        cache = self.cache
        rng   = np.random.default_rng(42)
        n     = cache.n_paths
        bs    = min(batch_size, n)

        c0, c1     = float(c0_init), float(c1_init)
        current_lr = float(lr)

        best_price = -np.inf
        best_c0, best_c1 = c0, c1

        history: list[dict] = []

        for step in range(n_steps):
            batch_idx = rng.choice(n, size=bs, replace=False)

            # Central finite differences for gradient estimation
            j_c0p = cache._evaluate_batch(c0 + epsilon, c1,            batch_idx)
            j_c0m = cache._evaluate_batch(c0 - epsilon, c1,            batch_idx)
            j_c1p = cache._evaluate_batch(c0,           c1 + epsilon,  batch_idx)
            j_c1m = cache._evaluate_batch(c0,           c1 - epsilon,  batch_idx)

            grad_c0 = (j_c0p - j_c0m) / (2.0 * epsilon)
            grad_c1 = (j_c1p - j_c1m) / (2.0 * epsilon)

            # Gradient ascent step
            c0 += current_lr * grad_c0
            c1 += current_lr * grad_c1
            current_lr *= lr_decay

            # Track progress
            batch_price = cache._evaluate_batch(c0, c1, batch_idx)
            full_price  = cache.expected_price(c0, c1)

            history.append({
                "step"        : step,
                "c0"          : c0,
                "c1"          : c1,
                "batch_price" : batch_price,
                "full_price"  : full_price,
            })

            if full_price > best_price:
                best_price    = full_price
                best_c0, best_c1 = c0, c1

        return {
            "c0"            : best_c0,
            "c1"            : best_c1,
            "expected_price": best_price,
            "history"       : pd.DataFrame(history),
        }

    # ── Compare ────────────────────────────────────────────────────────────────

    def compare(
        self,
        policies: dict[str, tuple[float, float]],
    ) -> pd.DataFrame:
        """
        Evaluate multiple named policies and return summary statistics.

        Parameters
        ----------
        policies : dict mapping name -> (c0, c1)

        Returns
        -------
        pd.DataFrame with columns:
            mean_price | std_price | mean_tau | sell_at_T_frac
        indexed by policy name.
        """
        cache = self.cache
        rows: list[dict] = []

        for name, (c0, c1) in policies.items():
            prices    = cache.evaluate(float(c0), float(c1))
            taus      = np.empty(cache.n_paths)
            at_T_mask = np.zeros(cache.n_paths, dtype=bool)

            for i in range(cache.n_paths):
                t         = cache._times[i]
                gv        = cache._g_vals[i]
                triggered = gv > (c0 + c1 * t)
                if triggered.any():
                    idx        = int(np.argmax(triggered))
                    taus[i]    = t[idx]
                    at_T_mask[i] = False
                else:
                    taus[i]      = t[-1]
                    at_T_mask[i] = True

            rows.append({
                "name"          : name,
                "mean_price"    : float(prices.mean()),
                "std_price"     : float(prices.std()),
                "mean_tau"      : float(taus.mean()),
                "sell_at_T_frac": float(at_T_mask.mean()),
            })

        return pd.DataFrame(rows).set_index("name")
