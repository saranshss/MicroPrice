"""
Level 1 Limit Order Book Simulator
===================================
Models the best bid and best ask of an order book as four competing
Poisson processes (Gillespie / next-event algorithm):

    λ_a   — limit sell orders arriving at the ask (add to ask qty)
    λ_b   — limit buy  orders arriving at the bid (add to bid qty)
    λ_ma  — market buy  orders (consume ask qty; price ticks up when depleted)
    λ_mb  — market sell orders (consume bid qty; price ticks down when depleted)

Optional cancellation processes (state-dependent rates):
    θ_a * q_a  — cancellations on the ask side
    θ_b * q_b  — cancellations on the bid side

References
----------
Stoikov, S. (2018). The micro-price: a high-frequency estimator of future prices.
Cont, Stoikov, Talreja (2010). A stochastic model for order book dynamics.
"""

from __future__ import annotations

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Event types ────────────────────────────────────────────────────────────────

class EventType(IntEnum):
    LIMIT_ASK   = 0   # limit sell order at ask  → qa += 1
    LIMIT_BID   = 1   # limit buy  order at bid  → qb += 1
    MARKET_BUY  = 2   # market buy  hits ask     → qa -= 1 (price up if qa→0)
    MARKET_SELL = 3   # market sell hits bid     → qb -= 1 (price down if qb→0)
    CANCEL_ASK  = 4   # cancellation at ask      → qa -= 1 (price up if qa→0)
    CANCEL_BID  = 5   # cancellation at bid      → qb -= 1 (price down if qb→0)

    def __str__(self) -> str:          # noqa: D401
        return self.name


EVENT_COLORS = {
    EventType.LIMIT_ASK:   "salmon",
    EventType.LIMIT_BID:   "skyblue",
    EventType.MARKET_BUY:  "red",
    EventType.MARKET_SELL: "blue",
    EventType.CANCEL_ASK:  "orange",
    EventType.CANCEL_BID:  "steelblue",
}


# ── State dataclass ────────────────────────────────────────────────────────────

@dataclass
class LOBState:
    pa: float   # best ask price
    qa: int     # best ask quantity
    pb: float   # best bid price
    qb: int     # best bid quantity
    time: float = 0.0

    def imbalance(self) -> float:
        """Order-book imbalance: Q_b / (Q_a + Q_b). Ranges in (0, 1)."""
        total = self.qa + self.qb
        return self.qb / total if total > 0 else 0.5

    def mid_price(self) -> float:
        return (self.pa + self.pb) / 2.0

    def spread(self) -> float:
        return self.pa - self.pb

    def micro_price(self) -> float:
        """Stoikov micro-price: imbalance-weighted interpolation between bid and ask."""
        i = self.imbalance()
        return i * self.pa + (1.0 - i) * self.pb


# ── Simulator ──────────────────────────────────────────────────────────────────

class LOBSimulator:
    """
    Level 1 Limit Order Book Simulator (Gillespie algorithm).

    Parameters
    ----------
    pa, qa : float, int
        Initial ask price and quantity.
    pb, qb : float, int
        Initial bid price and quantity.
    lambda_a : float
        Limit sell arrival rate (adds to ask side).
    lambda_b : float
        Limit buy arrival rate (adds to bid side).
    lambda_ma : float
        Market buy arrival rate (consumes ask; triggers upward tick on depletion).
    lambda_mb : float
        Market sell arrival rate (consumes bid; triggers downward tick on depletion).
    theta_a : float
        Per-unit cancellation rate on the ask side (default 0).
    theta_b : float
        Per-unit cancellation rate on the bid side (default 0).
    tick_size : float
        Minimum price increment (default 0.005).
    replenish_qty : int or None
        Fixed quantity at the new price level after depletion.
        If None, drawn from max(1, Poisson(mean_qty)) where mean_qty = (qa+qb)/2.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        pa: float,
        qa: int,
        pb: float,
        qb: int,
        lambda_a: float,
        lambda_b: float,
        lambda_ma: float,
        lambda_mb: float,
        theta_a: float = 0.0,
        theta_b: float = 0.0,
        tick_size: float = 0.005,
        replenish_qty: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if pb >= pa:
            raise ValueError(
                f"Bid price ({pb}) must be strictly less than ask price ({pa})."
            )
        if qa <= 0 or qb <= 0:
            raise ValueError("Initial quantities must be positive integers.")
        if any(r < 0 for r in [lambda_a, lambda_b, lambda_ma, lambda_mb, theta_a, theta_b]):
            raise ValueError("All rate parameters must be non-negative.")

        # Store constructor args for reset()
        self._init_pa = float(pa)
        self._init_qa = int(qa)
        self._init_pb = float(pb)
        self._init_qb = int(qb)
        self._mean_qty = (qa + qb) / 2.0

        self.lambda_a  = float(lambda_a)
        self.lambda_b  = float(lambda_b)
        self.lambda_ma = float(lambda_ma)
        self.lambda_mb = float(lambda_mb)
        self.theta_a   = float(theta_a)
        self.theta_b   = float(theta_b)
        self.tick_size = float(tick_size)
        self.replenish_qty = replenish_qty

        self.rng   = np.random.default_rng(seed)
        self.state = LOBState(
            pa=float(pa), qa=int(qa),
            pb=float(pb), qb=int(qb),
            time=0.0,
        )
        self._history: list[dict] = []
        self._record(None)   # initial snapshot

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _new_qty(self) -> int:
        """Sample a replenishment quantity for a freshly exposed price level."""
        if self.replenish_qty is not None:
            return self.replenish_qty
        return max(1, int(self.rng.poisson(self._mean_qty)))

    def _compute_rates(self) -> tuple[np.ndarray, list[EventType]]:
        """Return the rate vector and matching event-type list for current state."""
        s = self.state
        rates = np.array([
            self.lambda_a,             # LIMIT_ASK
            self.lambda_b,             # LIMIT_BID
            self.lambda_ma,            # MARKET_BUY
            self.lambda_mb,            # MARKET_SELL
            self.theta_a * s.qa,       # CANCEL_ASK  (state-dependent)
            self.theta_b * s.qb,       # CANCEL_BID  (state-dependent)
        ], dtype=float)
        events = list(EventType)
        return rates, events

    def _apply_event(self, event: EventType) -> None:
        """Mutate self.state according to event type."""
        s = self.state

        if event == EventType.LIMIT_ASK:
            s.qa += 1

        elif event == EventType.LIMIT_BID:
            s.qb += 1

        elif event == EventType.MARKET_BUY:
            s.qa -= 1
            if s.qa == 0:
                s.pa += self.tick_size
                s.qa  = self._new_qty()

        elif event == EventType.MARKET_SELL:
            s.qb -= 1
            if s.qb == 0:
                s.pb -= self.tick_size
                s.qb  = self._new_qty()

        elif event == EventType.CANCEL_ASK:
            s.qa -= 1
            if s.qa == 0:
                s.pa += self.tick_size
                s.qa  = self._new_qty()

        elif event == EventType.CANCEL_BID:
            s.qb -= 1
            if s.qb == 0:
                s.pb -= self.tick_size
                s.qb  = self._new_qty()

    def _record(self, event: Optional[EventType]) -> None:
        """Append current state snapshot to internal history list."""
        s = self.state
        self._history.append({
            "time"        : s.time,
            "pa"          : s.pa,
            "qa"          : s.qa,
            "pb"          : s.pb,
            "qb"          : s.qb,
            "mid"         : s.mid_price(),
            "spread"      : s.spread(),
            "imbalance"   : s.imbalance(),
            "micro_price" : s.micro_price(),
            "event"       : event,
            "event_name"  : event.name if event is not None else "INIT",
        })

    # ── Public simulation interface ────────────────────────────────────────────

    def step(self) -> EventType:
        """
        Advance the simulation by exactly one event (Gillespie direct method).

        Returns
        -------
        EventType
            The event that occurred.
        """
        rates, events = self._compute_rates()
        total_rate = rates.sum()

        if total_rate == 0.0:
            raise RuntimeError("Total event rate is zero; simulation is stuck.")

        # Time to next event
        dt = self.rng.exponential(1.0 / total_rate)
        self.state.time += dt

        # Which event fires?
        probs = rates / total_rate
        event = events[int(self.rng.choice(len(events), p=probs))]

        self._apply_event(event)
        self._record(event)
        return event

    def run(
        self,
        T: Optional[float] = None,
        n_events: Optional[int] = None,
        max_events: int = 10_000_000,
    ) -> pd.DataFrame:
        """
        Run the simulation.

        Provide exactly one stopping criterion:

        Parameters
        ----------
        T : float, optional
            Simulate until simulated time reaches T.
        n_events : int, optional
            Simulate exactly this many events.
        max_events : int
            Hard safety cap (default 10 M).

        Returns
        -------
        pd.DataFrame
            Full event history (one row per event, including initial state).
        """
        if (T is None) == (n_events is None):
            raise ValueError("Provide exactly one of T (time horizon) or n_events.")

        count = 0
        if T is not None:
            while self.state.time < T and count < max_events:
                self.step()
                count += 1
        else:
            while count < n_events and count < max_events:
                self.step()
                count += 1

        return self.get_history()

    def get_history(self) -> pd.DataFrame:
        """Return event history as a DataFrame."""
        return pd.DataFrame(self._history)

    def reset(self, seed: Optional[int] = None) -> "LOBSimulator":
        """
        Reset state and history to initial conditions.

        Parameters
        ----------
        seed : int, optional
            New random seed. If None, the existing RNG continues.

        Returns
        -------
        self  (for chaining)
        """
        self.state = LOBState(
            pa=self._init_pa, qa=self._init_qa,
            pb=self._init_pb, qb=self._init_qb,
            time=0.0,
        )
        self._history.clear()
        self._record(None)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return self


# ── Analyzer ───────────────────────────────────────────────────────────────────

class LOBAnalyzer:
    """
    Computes summary statistics and derived series from a simulation history.

    Parameters
    ----------
    history : pd.DataFrame
        DataFrame returned by ``LOBSimulator.run()`` or ``get_history()``.
    """

    def __init__(self, history: pd.DataFrame) -> None:
        self.h = history.copy()

    # ── Derived series ─────────────────────────────────────────────────────────

    def microprice_series(self) -> pd.Series:
        """Stoikov micro-price at each snapshot."""
        return self.h["micro_price"]

    def time_average_imbalance(self) -> float:
        """
        Time-weighted average imbalance (TWAP):
        ∫ I(t) dt / T  via the trapezoid rule.
        """
        h = self.h
        if len(h) < 2:
            return float(h["imbalance"].iloc[0])
        T = h["time"].iloc[-1] - h["time"].iloc[0]
        if T == 0:
            return float(h["imbalance"].mean())
        trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        return float(trapz(h["imbalance"], h["time"]) / T)

    def price_move_events(self) -> pd.DataFrame:
        """
        Return only the rows where a price tick occurred
        (ask moved up or bid moved down).
        """
        h = self.h
        ask_moved = h["pa"].diff().fillna(0) != 0
        bid_moved = h["pb"].diff().fillna(0) != 0
        return h[ask_moved | bid_moved].copy()

    # ── Summary statistics ─────────────────────────────────────────────────────

    def event_frequency(self) -> pd.Series:
        """Count of each event type, sorted descending."""
        return self.h["event_name"].value_counts()

    def summary_stats(self) -> pd.DataFrame:
        """One-row DataFrame with key simulation statistics."""
        h = self.h
        pm = self.price_move_events()
        ask_ups   = int((pm["pa"].diff() > 0).sum())
        bid_downs = int((pm["pb"].diff() < 0).sum())

        row = {
            "n_events"       : len(h) - 1,
            "total_time"     : round(float(h["time"].iloc[-1]), 6),
            "mean_spread"    : round(float(h["spread"].mean()), 6),
            "std_spread"     : round(float(h["spread"].std()), 6),
            "mean_mid"       : round(float(h["mid"].mean()), 6),
            "std_mid"        : round(float(h["mid"].std()), 6),
            "mean_imbalance" : round(float(h["imbalance"].mean()), 4),
            "twap_imbalance" : round(self.time_average_imbalance(), 4),
            "mean_qa"        : round(float(h["qa"].mean()), 2),
            "mean_qb"        : round(float(h["qb"].mean()), 2),
            "ask_price_ups"  : ask_ups,
            "bid_price_downs": bid_downs,
            "final_mid"      : round(float(h["mid"].iloc[-1]), 6),
        }
        return pd.DataFrame([row])


# ── Plotter ────────────────────────────────────────────────────────────────────

class LOBPlotter:
    """
    Visualization utilities for LOB simulation history.

    Parameters
    ----------
    history : pd.DataFrame
        DataFrame from ``LOBSimulator.run()``.
    tick_size : float
        Tick size used in the simulation (for reference lines).
    """

    def __init__(self, history: pd.DataFrame, tick_size: float = 0.005) -> None:
        self.h = history
        self.tick_size = tick_size

    # ── Individual panels ──────────────────────────────────────────────────────

    def plot_price_dynamics(self, ax: Optional[plt.Axes] = None):
        """Ask, bid, mid, and micro-price over simulated time."""
        h = self.h
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(12, 4))
        else:
            fig = ax.get_figure()

        ax.step(h["time"], h["pa"],          where="post",
                color="red",   alpha=0.7, lw=0.9, label="Ask")
        ax.step(h["time"], h["pb"],          where="post",
                color="green", alpha=0.7, lw=0.9, label="Bid")
        ax.step(h["time"], h["mid"],         where="post",
                color="black", alpha=0.9, lw=1.3, ls="--", label="Mid")
        ax.step(h["time"], h["micro_price"], where="post",
                color="blue",  alpha=0.8, lw=1.0, ls=":", label="Micro-price")

        # Mark price-move events as thin vertical lines
        pm = h[h["pa"].diff().fillna(0) != 0]
        for t in pm["time"]:
            ax.axvline(t, color="grey", lw=0.4, alpha=0.25)

        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.set_title("Price Dynamics")
        ax.legend(fontsize=8, ncol=4)
        ax.grid(True, alpha=0.25)
        return (fig, ax) if created else (fig, ax)

    def plot_spread(self, ax: Optional[plt.Axes] = None):
        """Bid–ask spread over time with tick-size reference line."""
        h = self.h
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(12, 3))
        else:
            fig = ax.get_figure()

        ax.step(h["time"], h["spread"], where="post",
                color="purple", alpha=0.8, lw=0.9, label="Spread")
        ax.axhline(self.tick_size, color="red", ls="--", lw=0.8, label="1 tick")
        ax.set_xlabel("Time")
        ax.set_ylabel("Spread")
        ax.set_title("Bid–Ask Spread")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        return (fig, ax) if created else (fig, ax)

    def plot_quantities(self, ax: Optional[plt.Axes] = None):
        """Ask and bid queue depths over time."""
        h = self.h
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(12, 3))
        else:
            fig = ax.get_figure()

        ax.step(h["time"], h["qa"], where="post",
                color="red",   alpha=0.7, lw=0.9, label="Ask qty")
        ax.step(h["time"], h["qb"], where="post",
                color="green", alpha=0.7, lw=0.9, label="Bid qty")
        ax.set_xlabel("Time")
        ax.set_ylabel("Quantity")
        ax.set_title("Best Bid / Ask Queue Depth")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        return (fig, ax) if created else (fig, ax)

    def plot_imbalance(self, ax: Optional[plt.Axes] = None):
        """Order-book imbalance I = Q_b / (Q_a + Q_b) over time."""
        h = self.h
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(12, 3))
        else:
            fig = ax.get_figure()

        ax.step(h["time"], h["imbalance"], where="post",
                color="darkorange", alpha=0.85, lw=0.9, label="Imbalance")
        ax.axhline(0.5, color="black", ls="--", lw=0.8, label="Balanced (0.5)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Time")
        ax.set_ylabel("Imbalance")
        ax.set_title("Order-book Imbalance  I = Q_b / (Q_a + Q_b)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        return (fig, ax) if created else (fig, ax)

    def plot_event_timeline(self, max_events: int = 500, ax: Optional[plt.Axes] = None):
        """
        Scatter timeline colored by event type (first *max_events* rows).
        Useful for visually verifying the relative intensities of each process.
        """
        h = self.h.iloc[:max_events]
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(14, 2.5))
        else:
            fig = ax.get_figure()

        for event, color in EVENT_COLORS.items():
            mask = h["event"] == event
            ax.scatter(
                h.loc[mask, "time"],
                [event.value] * int(mask.sum()),
                c=color, s=6, alpha=0.7, label=str(event),
            )

        ax.set_yticks(list(range(len(EventType))))
        ax.set_yticklabels([str(e) for e in EventType], fontsize=8)
        ax.set_xlabel("Time")
        ax.set_title(f"Event Timeline (first {max_events} events)")
        ax.legend(fontsize=7, ncol=3, loc="upper right")
        ax.grid(True, alpha=0.2, axis="x")
        return (fig, ax) if created else (fig, ax)

    # ── Dashboard ──────────────────────────────────────────────────────────────

    def plot_dashboard(self, figsize=(14, 12)):
        """
        2 × 2 dashboard: price dynamics, spread, quantities, imbalance.

        Returns
        -------
        (fig, axes) : (Figure, dict of str→Axes)
        """
        fig = plt.figure(figsize=figsize)
        gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.3)

        axes = {
            "prices"    : fig.add_subplot(gs[0, 0]),
            "spread"    : fig.add_subplot(gs[0, 1]),
            "quantities": fig.add_subplot(gs[1, 0]),
            "imbalance" : fig.add_subplot(gs[1, 1]),
        }

        self.plot_price_dynamics(ax=axes["prices"])
        self.plot_spread        (ax=axes["spread"])
        self.plot_quantities    (ax=axes["quantities"])
        self.plot_imbalance     (ax=axes["imbalance"])

        return fig, axes
