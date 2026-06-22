"""Adversarial audit of GC@h22 MCS={EWMA} finding.

Runs:
1. Seed sweep: seeds [0,1,2,3,7,11,42], mcs_reps=10000
2. Reps sensitivity: seed=0, mcs_reps in [2000, 20000]
3. Contamination check: isolates HARQ's outsized loss and reruns without it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_crypto_expanded import BENCHMARK, HAR_FAMILY, _verdict, build_suite  # noqa: E402
from run_volare_futures import VOLARE_FUTURES_CONFIG  # noqa: E402

from volbench.backtest import run_backtest  # noqa: E402
from volbench.data import load_realized_panel  # noqa: E402
from volbench.models import ARFIMALog  # noqa: E402

DATA = ROOT / "data" / "volare_futures_realized.csv"

# ---- load GC data once ----
ds = load_realized_panel(DATA, VOLARE_FUTURES_CONFIG, symbols=["GC"])
fr = ds.frame("GC")
rv = fr["rv5"].to_numpy()
suite = build_suite(fr) + [ARFIMALog()]
print(f"GC: {len(rv)} obs, suite: {[m.name for m in suite]}")

# ---- seed sweep ----
SEEDS = [0, 1, 2, 3, 7, 11, 42]
seed_results = []

print("\n=== SEED SWEEP (mcs_reps=10000, h=22) ===")
for seed in SEEDS:
    res = run_backtest(rv, horizon=22, models=suite, mcs_reps=10000, seed=seed, benchmark=BENCHMARK)
    mcs_inc = set(res.mcs["QLIKE"].included)
    mean_q = res.mean_losses["QLIKE"]
    har_in = any(n in mcs_inc for n in HAR_FAMILY)
    best = min(mean_q, key=lambda k: mean_q[k])
    mcs_sorted = sorted(mcs_inc)
    v, _ = _verdict(mcs_inc, mean_q, res.dm_vs_har["QLIKE"])
    seed_results.append({
        "seed": seed,
        "har_in_mcs": har_in,
        "mcs": mcs_sorted,
        "best": best,
        "verdict": v,
    })
    print(f"  seed={seed:>2}  har_in_mcs={har_in}  mcs={mcs_sorted}  best={best}  verdict={v}")

# ---- reps sensitivity ----
print("\n=== REPS SENSITIVITY (seed=0, h=22) ===")
reps_results = {}
for reps in [2000, 20000]:
    res = run_backtest(rv, horizon=22, models=suite, mcs_reps=reps, seed=0, benchmark=BENCHMARK)
    mcs_inc = set(res.mcs["QLIKE"].included)
    mean_q = res.mean_losses["QLIKE"]
    har_in = any(n in mcs_inc for n in HAR_FAMILY)
    best = min(mean_q, key=lambda k: mean_q[k])
    v, _ = _verdict(mcs_inc, mean_q, res.dm_vs_har["QLIKE"])
    reps_results[reps] = {"har_in_mcs": har_in, "mcs": sorted(mcs_inc), "best": best, "verdict": v}
    print(f"  reps={reps:>6}  har_in_mcs={har_in}  mcs={sorted(mcs_inc)}  best={best}")

# ---- contamination check: run without HARQ ----
print("\n=== CONTAMINATION CHECK (no HARQ, seed=0, mcs_reps=10000, h=22) ===")
suite_no_harq = [m for m in suite if m.name != "HARQ"]
res_no_harq = run_backtest(rv, horizon=22, models=suite_no_harq, mcs_reps=10000, seed=0, benchmark=BENCHMARK)
mcs_inc_nh = set(res_no_harq.mcs["QLIKE"].included)
mean_q_nh = res_no_harq.mean_losses["QLIKE"]
har_in_nh = any(n in mcs_inc_nh for n in HAR_FAMILY)
best_nh = min(mean_q_nh, key=lambda k: mean_q_nh[k])
v_nh, _ = _verdict(mcs_inc_nh, mean_q_nh, res_no_harq.dm_vs_har["QLIKE"])
print(f"  no-HARQ: har_in_mcs={har_in_nh}  mcs={sorted(mcs_inc_nh)}  best={best_nh}  verdict={v_nh}")

# ---- show mean losses ranked ----
print("\n=== MEAN QLIKE LOSSES (seed=0, original) ===")
res0 = run_backtest(rv, horizon=22, models=suite, mcs_reps=10000, seed=0, benchmark=BENCHMARK)
mean_q0 = res0.mean_losses["QLIKE"]
mcs_inc0 = set(res0.mcs["QLIKE"].included)
elim = res0.mcs["QLIKE"].elimination_order
pvals = res0.mcs["QLIKE"].p_values
print("  Elimination order and p-values:")
for nm in elim:
    tag = " [IN MCS]" if nm in mcs_inc0 else ""
    tag2 = " [HAR]" if nm in HAR_FAMILY else ""
    print(f"    {nm:<18} mean_QLIKE={mean_q0[nm]:.6f}  mcs_pval={pvals[nm]:.4f}{tag}{tag2}")

# ---- summarize ----
print("\n=== SUMMARY ===")
print(json.dumps(seed_results, indent=2))
print("reps_results:", json.dumps(reps_results, indent=2))
print(f"contamination(no_HARQ): har_in_mcs={har_in_nh}, mcs={sorted(mcs_inc_nh)}, best={best_nh}")
