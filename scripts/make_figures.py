"""Generate the publication figures into ``results/figures/``.

Figures
-------
1. ``signature_plot.png``: the volatility signature plot: average realized
   variance against sampling interval under microstructure noise. RV explodes at
   high frequency while the realized kernel and the coarse-sampled RV sit on the
   true quadratic variation. The canonical motivation for noise-robust
   estimators.
2. ``leaderboard_h1.png``: average out-of-sample QLIKE by model at h = 1.
3. ``qlike_by_horizon.png``: average QLIKE rank by model across horizons.
4. ``spx_realized_vol.png``: real .SPX annualised realized volatility with the
   GFC and COVID crisis windows shaded.
5. ``transfer_matrix.png``: the Q5 cross-asset transfer matrix: where the HAR
   family stays in / leaves the 90% MCS across asset classes and horizons (the
   project's primary pre-registered deliverable).

Usage
-----
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
})

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
    fig.savefig(FIG_DIR / "signature_plot.png", dpi=150)
    plt.close(fig)
    print("  wrote signature_plot.png")


def figure_crypto_signature(days: int = 20, end_date: str = "2026-06-23") -> None:
    """Real-data volatility signature plot from BTC 1-minute bars.

    The simulated signature plot shows realized variance exploding at high
    frequency under noise; this is the same effect on *real* data. RV is computed
    at increasing sampling intervals (averaged over the ``days`` ending
    ``end_date``) and compared to a realized kernel.

    The window is anchored to a fixed research date, and the computed series is
    cached to ``results/crypto_signature.json``, so the figure is reproducible
    offline and does not drift with a live "last N days" fetch. Delete the cache to
    re-pull.
    """
    import json
    import urllib.request
    from datetime import datetime, timezone

    cache = FIG_DIR.parent / "crypto_signature.json"
    intervals = [1, 2, 3, 5, 10, 15, 20, 30, 60]

    if cache.exists():
        rec = json.loads(cache.read_text())
        mean_rv, rk, end_date = rec["mean_rv"], rec["rk"], rec.get("end_date", end_date)
    else:
        try:
            end_ms = int(
                datetime.strptime(end_date, "%Y-%m-%d")
                .replace(tzinfo=timezone.utc).timestamp() * 1000
            )
            url = ("https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m"
                   "&limit=1000")
            # Page back from the fixed research date to assemble ~`days` of closes.
            pages, last = [], end_ms
            for _ in range(max(1, days * 1440 // 1000 + 1)):
                u = url + f"&endTime={last}"
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
        mean_rv = []
        for m in intervals:
            rvs = []
            for d in np.unique(day_idx):
                p = closes[day_idx == d]
                if p.size < 200:
                    continue
                agg = _aggregate_returns(np.diff(np.log(p)), m)
                rvs.append(realized_variance(agg))
            mean_rv.append(float(np.mean(rvs)) if rvs else float("nan"))
        rk_vals = [realized_kernel_parzen(np.diff(np.log(closes[day_idx == d])))
                   for d in np.unique(day_idx) if (day_idx == d).sum() >= 200]
        rk = float(np.mean(rk_vals)) if rk_vals else float("nan")
        cache.write_text(json.dumps(
            {"end_date": end_date, "days": days, "intervals": intervals,
             "mean_rv": mean_rv, "rk": rk}, indent=2))

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(intervals, np.array(mean_rv) * 1e4, "o-", color="#8e44ad", label="Realized variance (BTC)")
    if np.isfinite(rk):
        ax.axhline(rk * 1e4, ls=":", color="#27ae60", lw=2, label="Realized kernel (1-min)")
    ax.set_xlabel("Sampling interval (minutes)")
    ax.set_ylabel(r"Mean realized variance ($\times 10^{-4}$)")
    ax.set_title(f"Real-data signature plot: BTC, {days} days to {end_date} (1-min bars)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "crypto_signature.png", dpi=150)
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
    fig.savefig(FIG_DIR / "leaderboard_h1.png", dpi=150)
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
    fig.savefig(FIG_DIR / "qlike_by_horizon.png", dpi=150)
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
    ax.set_ylabel("Annualised realized volatility (decimal, 0.20 = 20%)")
    ax.set_title(".SPX 5-minute realized volatility (Oxford-Man, 2000–2022)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "spx_realized_vol.png", dpi=150)
    plt.close(fig)
    print("  wrote spx_realized_vol.png")


def figure_transfer_matrix() -> None:
    """Q5 cross-asset transfer matrix (the primary pre-registered deliverable).

    For each asset class and horizon, colour the cell by its *worst* per-instrument
    state: green if a HAR-family model is in the 90% MCS and single-best for
    every instrument, amber if some instruments are only competitive, red if at
    least one instrument shows a genuine HAR degradation. The annotation gives the
    dominate fraction (``k/n``) and names the actual crack. Reads the committed
    ``results/transfer_matrix.json`` (built by ``scripts/build_transfer_matrix.py``).
    """
    from matplotlib.colors import ListedColormap  # noqa: PLC0415
    from matplotlib.patches import Patch  # noqa: PLC0415

    path = RESULTS_DIR / "transfer_matrix.json"
    if not path.exists():
        print("  (skip transfer matrix: run scripts/build_transfer_matrix.py first)")
        return
    tm = json.loads(path.read_text())
    rows = [
        ("Equities (8)", "Equities (8)"),
        ("Crypto: 4 coins", "Crypto (4)"),
        ("Crypto: 22 coins", "Crypto (22, +dead)"),
        ("Futures: rates (FV/TY)", "Rate futures (FV/TY)"),
        ("Futures: commodity (8)", "Commodity futures (8)"),
        ("Futures: equity-index (ES/NQ)", "Equity-idx futures (ES/NQ)"),
        ("Futures: fx (EU)", "FX future (EU)"),
        ("FX: major (7)", "FX major (7)"),
        ("FX: secondary/EM (6)", "FX secondary/EM (6)"),
    ]
    horizons = ["1", "5", "22"]
    state = np.zeros((len(rows), len(horizons)), dtype=int)
    cell_text = [["" for _ in horizons] for _ in rows]
    for i, (key, _) in enumerate(rows):
        by_h = tm[key]["by_horizon"]
        for j, h in enumerate(horizons):
            cell = by_h[h]
            counts = cell["verdict_counts"]
            n = sum(counts.values())
            dom = counts.get("dominates", 0)
            n_degr = counts.get("degrades", 0)
            n_comp = counts.get("competitive", 0)
            if n_degr > 0:
                state[i, j] = 2
            elif n_comp > 0:
                state[i, j] = 1
            else:
                state[i, j] = 0
            txt = f"{dom}/{n}"
            degr = [f"{c['name']}→{c['best']}" for c in cell["cracks"]
                    if c["verdict"] == "degrades"]
            if degr:
                txt += "\n↓ " + ", ".join(degr)
            elif n_comp > 0:
                txt += f"\n~{n_comp} comp."
            cell_text[i][j] = txt

    cmap = ListedColormap(["#1a9850", "#fdae61", "#d73027"])
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    ax.imshow(state, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(horizons)), [f"h = {h}" for h in horizons])
    ax.set_yticks(range(len(rows)), [r[1] for r in rows])
    ax.set_xlabel("Forecast horizon (days)")
    for i in range(len(rows)):
        for j in range(len(horizons)):
            ax.text(j, i, cell_text[i][j], ha="center", va="center", fontsize=8.5,
                    color="#5a3d00" if state[i, j] == 1 else "white", fontweight="bold")
    ax.set_xticks(np.arange(-0.5, len(horizons), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title(
        "Q5 cross-asset transfer matrix: where the HAR family stays in the 90% MCS\n"
        "cell = instruments where a HAR model is in-MCS & single-best (QLIKE)",
        fontsize=10,
    )
    handles = [
        Patch(color="#1a9850", label="HAR dominates (all)"),
        Patch(color="#fdae61", label="competitive (some instr.)"),
        Patch(color="#d73027", label="HAR degrades (≥1 instr.)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07),
              ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "transfer_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote transfer_matrix.png")


def figure_var_dq() -> None:
    """DQ pass count per VaR engine (out of 8 indices) from caviar.json.

    A grouped bar chart: one bar per engine coloured by the number of
    indices on which the Engle-Manganelli DQ test does not reject at 5%.
    The dashed line shows the 5% nominal violation rate for reference.
    Reads results/caviar.json; skips quietly if the file is absent.
    """
    path = RESULTS_DIR / "caviar.json"
    if not path.exists():
        print("  (skip var_dq: run scripts/run_caviar.py first)")
        return

    data = json.loads(path.read_text())
    summary = data.get("summary", {})

    # Extract DQ pass counts and average violation rates per engine.
    engines = list(summary.keys())
    dq_pass = [summary[e].get("dq_pass_count", 0) for e in engines]
    avg_viol = [summary[e].get("avg_violation_rate", float("nan")) * 100 for e in engines]
    n_indices = summary[engines[0]].get("n_indices", 8) if engines else 8

    # Colour bars by pass count: green for the best, orange for mid, red for zero.
    best = max(dq_pass) if dq_pass else 0
    colors = []
    for v in dq_pass:
        if v == best and v > 0:
            colors.append("#1a9850")
        elif v > 0:
            colors.append("#fdae61")
        else:
            colors.append("#d73027")

    x = np.arange(len(engines))
    fig, ax1 = plt.subplots(figsize=(9, 4.8))

    bars = ax1.bar(x, dq_pass, color=colors, width=0.5, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(engines, rotation=20, ha="right")
    ax1.set_ylabel(f"DQ pass count (out of {int(n_indices)} indices)")
    ax1.set_ylim(0, n_indices + 0.5)
    ax1.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # Overlay mean violation rate on a secondary y-axis.
    ax2 = ax1.twinx()
    ax2.plot(x, avg_viol, "o--", color="#2c3e50", linewidth=1.2,
             markersize=5, label="Mean violation rate (%)", zorder=4)
    ax2.axhline(5.0, color="#555555", linewidth=0.8, linestyle=":",
                label="Nominal 5%")
    ax2.set_ylabel("Mean violation rate (%)")
    ax2.set_ylim(0, max(avg_viol) * 1.4 if any(np.isfinite(avg_viol)) else 15)
    ax2.legend(loc="upper right", fontsize=8, frameon=False)

    # Label DQ pass counts above each bar.
    for bar, v in zip(bars, dq_pass):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 str(v), ha="center", va="bottom", fontsize=9)

    ax1.set_title(
        "VaR engine comparison: DQ pass count and mean violation rate\n"
        "(alpha=5%, 8 equity indices; DQ pass = fail-to-reject at 5%)"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "var_dq.png", dpi=150)
    plt.close(fig)
    print("  wrote var_dq.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating figures ...")
    figure_signature_plot()
    figure_crypto_signature()
    figure_spx_realized_vol()
    figure_leaderboard()
    figure_qlike_by_horizon()
    figure_transfer_matrix()
    figure_var_dq()
    print(f"Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
