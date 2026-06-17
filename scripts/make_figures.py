"""Generate the publication figures into ``results/figures/``.

Figures
-------
1. ``signature_plot.png`` — the volatility signature plot: average realized
   variance against sampling interval under microstructure noise. RV explodes at
   high frequency while the realized kernel and the coarse-sampled RV sit on the
   true quadratic variation. The canonical motivation for noise-robust
   estimators.
2. ``leaderboard_h1.png`` — average out-of-sample QLIKE by model at h = 1.
3. ``qlike_by_horizon.png`` — average QLIKE rank by model across horizons.
4. ``spx_realized_vol.png`` — real .SPX annualised realized volatility with the
   GFC and COVID crisis windows shaded.

Usage
-----
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volbench.data import TRADING_DAYS, load_oxford_rv  # noqa: E402
from volbench.realized import realized_kernel_parzen, realized_variance  # noqa: E402
from volbench.simulate import simulate_many_days  # noqa: E402

FIG_DIR = ROOT / "results" / "figures"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = ROOT / "results" / "tables"


def _aggregate_returns(returns: np.ndarray, m: int) -> np.ndarray:
    """Aggregate consecutive returns into non-overlapping blocks of size ``m``."""
    n = (returns.size // m) * m
    return returns[:n].reshape(-1, m).sum(axis=1)


def figure_signature_plot(days: int = 1500, seed: int = 7) -> None:
    """Volatility signature plot under microstructure noise."""
    n_steps = 780
    sim = simulate_many_days(
        days, seed=seed, n_steps=n_steps, ann_vol=0.20, kappa=5.0, vol_of_vol=0.8,
        jump_intensity=0.0, jump_size_vol=0.0, noise_ratio=1.0,
    )
    qv = float(np.mean(sim["qv"]))
    intervals = [1, 2, 3, 5, 8, 13, 20, 30, 39, 52, 78, 130]
    mean_rv = []
    for m in intervals:
        rvs = [realized_variance(_aggregate_returns(r, m)) for r in sim["returns"]]
        mean_rv.append(float(np.mean(rvs)))
    rk = float(np.mean([realized_kernel_parzen(r) for r in sim["returns"]]))

    minutes = [m * (390 / n_steps) for m in intervals]  # map steps -> minutes
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(minutes, np.array(mean_rv) * 1e4, "o-", color="#c0392b", label="Realized variance")
    ax.axhline(qv * 1e4, ls="--", color="#2c3e50", label="True quadratic variation")
    ax.axhline(rk * 1e4, ls=":", color="#27ae60", lw=2, label="Realized kernel (Parzen)")
    ax.set_xlabel("Sampling interval (minutes)")
    ax.set_ylabel(r"Mean realized variance ($\times 10^{-4}$)")
    ax.set_title("Volatility signature plot under microstructure noise")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "signature_plot.png", dpi=140)
    plt.close(fig)
    print("  wrote signature_plot.png")


def figure_crypto_signature(days: int = 20) -> None:
    """Real-data volatility signature plot from live BTC 1-minute bars.

    The simulated signature plot shows realized variance exploding at high
    frequency under noise; this is the same effect on *real* data. RV is computed
    at increasing sampling intervals (averaged over recent days) and compared to a
    realized kernel. Network-dependent — skips quietly when offline.
    """
    import json
    import urllib.request

    try:
        url = ("https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m"
               "&limit=1000")
        # Page back from the latest bars to assemble ~`days` of 1-minute closes.
        pages, last = [], None
        for _ in range(max(1, days * 1440 // 1000 + 1)):
            u = url + (f"&endTime={last}" if last else "")
            with urllib.request.urlopen(u, timeout=20) as r:  # noqa: S310
                page = json.load(r)
            if not page:
                break
            pages = page + pages
            last = page[0][0] - 1
        closes = np.array([float(row[4]) for row in pages])
        times = np.array([row[0] for row in pages], dtype="int64")
    except Exception as exc:  # noqa: BLE001 - figure is a bonus; skip if offline
        print(f"  (skip crypto signature plot: {exc})")
        return

    day_idx = times // (1000 * 60 * 60 * 24)
    intervals = [1, 2, 3, 5, 10, 15, 20, 30, 60]
    mean_rv = []
    for m in intervals:
        rvs = []
        for d in np.unique(day_idx):
            p = closes[day_idx == d]
            if p.size < 200:
                continue
            r = np.diff(np.log(p))[:: 1]
            agg = _aggregate_returns(r, m)
            rvs.append(realized_variance(agg))
        mean_rv.append(float(np.mean(rvs)) if rvs else np.nan)
    rk_vals = [realized_kernel_parzen(np.diff(np.log(closes[day_idx == d])))
               for d in np.unique(day_idx) if (day_idx == d).sum() >= 200]
    rk = float(np.mean(rk_vals)) if rk_vals else np.nan

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(intervals, np.array(mean_rv) * 1e4, "o-", color="#8e44ad", label="Realized variance (BTC)")
    if np.isfinite(rk):
        ax.axhline(rk * 1e4, ls=":", color="#27ae60", lw=2, label="Realized kernel (1-min)")
    ax.set_xlabel("Sampling interval (minutes)")
    ax.set_ylabel(r"Mean realized variance ($\times 10^{-4}$)")
    ax.set_title(f"Real-data signature plot: BTC, last {days} days (1-min bars)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "crypto_signature.png", dpi=140)
    plt.close(fig)
    print("  wrote crypto_signature.png")


def figure_leaderboard() -> None:
    """Average QLIKE by model at h = 1 (from the benchmark leaderboard)."""
    path = TABLES_DIR / "leaderboard_qlike_h1.csv"
    if not path.exists():
        print("  (skip leaderboard: run scripts/run_benchmark.py first)")
        return
    df = pd.read_csv(path, index_col=0).sort_values("avg_QLIKE")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    colors = ["#27ae60" if n == "LogHAR" else "#34495e" for n in df.index]
    ax.barh(df.index[::-1], df["avg_QLIKE"][::-1], color=colors[::-1])
    ax.set_xlabel("Average out-of-sample QLIKE (lower is better)")
    ax.set_title("Model leaderboard, h = 1 (8 indices, real data)")
    for i, v in enumerate(df["avg_QLIKE"][::-1]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "leaderboard_h1.png", dpi=140)
    plt.close(fig)
    print("  wrote leaderboard_h1.png")


def figure_qlike_by_horizon() -> None:
    """Average QLIKE rank by model across horizons (from summary.json)."""
    path = RESULTS_DIR / "summary.json"
    if not path.exists():
        print("  (skip horizon plot: run scripts/run_benchmark.py first)")
        return
    summary = json.loads(path.read_text())
    horizons = summary["horizons"]
    by_h = summary["by_horizon"]
    models = list(by_h[str(horizons[0])]["avg_rank"].keys())
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m in models:
        ranks = [by_h[str(h)]["avg_rank"][m] for h in horizons]
        lw = 2.6 if m in ("LogHAR", "GBRT", "HAR") else 1.2
        ax.plot(horizons, ranks, "o-", lw=lw, label=m)
    ax.set_xticks(horizons)
    ax.set_xlabel("Forecast horizon (days)")
    ax.set_ylabel("Average QLIKE rank (1 = best)")
    ax.set_title("Model rank across horizons")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qlike_by_horizon.png", dpi=140)
    plt.close(fig)
    print("  wrote qlike_by_horizon.png")


def figure_spx_realized_vol() -> None:
    """Real .SPX annualised realized volatility with crisis windows shaded."""
    ds = load_oxford_rv(tickers=[".SPX"])
    frame = ds.frame(".SPX")
    ann_vol = np.sqrt(frame["rv5"].to_numpy() * TRADING_DAYS)
    dates = frame.index
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.plot(dates, ann_vol, lw=0.7, color="#2c3e50")
    for (lo, hi, label) in [
        ("2008-09-01", "2009-06-30", "GFC"),
        ("2020-02-15", "2020-06-30", "COVID"),
    ]:
        ax.axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color="#e74c3c", alpha=0.18)
        ax.text(pd.Timestamp(lo), ann_vol.max() * 0.92, " " + label, fontsize=9, color="#c0392b")
    ax.set_ylabel("Annualised realized volatility")
    ax.set_title(".SPX 5-minute realized volatility (Oxford-Man, 2000–2022)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "spx_realized_vol.png", dpi=140)
    plt.close(fig)
    print("  wrote spx_realized_vol.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating figures ...")
    figure_signature_plot()
    figure_crypto_signature()
    figure_spx_realized_vol()
    figure_leaderboard()
    figure_qlike_by_horizon()
    print(f"Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
