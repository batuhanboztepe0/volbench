"""Validate realized estimators against simulated ground truth.

The bundled real data ships daily realized *measures* but not the underlying
intraday returns, so the estimators in :mod:`volbench.realized` cannot be
checked against a known integrated variance on real data. Instead we simulate
intraday paths with **known** integrated variance (IV) and jump variation (JV)
and confirm each estimator recovers what it targets:

* ``RV / QV`` -> 1 (realized variance estimates total quadratic variation),
* ``BV / IV`` and ``medRV / IV`` -> 1 (jump-robust, target the continuous part),
* ``(RV - BV) / JV`` -> 1 (the jump-variation decomposition),
* ``RK / QV`` -> 1 on clean data, and crucially ``RK / QV`` stays ~1 **under
  microstructure noise** while ``RV / QV`` explodes (the signature-plot effect),
* the BNS jump test holds its size (false-positive rate ~ the nominal level) and
  has power against injected jumps.

Writes ``results/validation.json``.

Usage
-----
    python scripts/validate_estimators.py [--days N] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volbench.realized import (  # noqa: E402
    bipower_variation,
    bns_jump_test,
    median_rv,
    realized_kernel_parzen,
    realized_variance,
)
from volbench.simulate import simulate_many_days  # noqa: E402

RESULTS_DIR = ROOT / "results"
JUMP_ALPHA: float = 0.05


def _aggregate_ratio(numer: np.ndarray, denom: np.ndarray) -> float:
    """Ratio of summed quantities (robust to small per-day denominators)."""
    d = float(np.sum(denom))
    return float(np.sum(numer) / d) if d > 0 else float("nan")


def validate(days: int = 4000, seed: int = 0) -> dict:
    """Run all estimator-validation experiments.

    Parameters
    ----------
    days : int, default 4000
        Number of simulated days per experiment.
    seed : int, default 0
        Base seed for reproducibility.

    Returns
    -------
    dict
        Nested results suitable for JSON serialisation.
    """
    n_steps = 390

    # --- Experiment 1a: pure continuous price (no jumps, no noise). The
    #     jump-robust estimators (BV, medRV) are only *asymptotically* jump-robust,
    #     so their unbiasedness target of 1.00 is only meaningful on jump-free
    #     paths (here QV == IV). Validating "BV/IV = 1.00" on jump-contaminated
    #     data conflated the estimator with the scenario. -----------------------
    sim_clean = simulate_many_days(
        days, seed=seed, n_steps=n_steps, ann_vol=0.20, kappa=5.0, vol_of_vol=0.8,
        jump_intensity=0.0, jump_size_vol=0.0, noise_ratio=0.0,
    )
    rv0 = np.array([realized_variance(r) for r in sim_clean["returns"]])
    bv0 = np.array([bipower_variation(r) for r in sim_clean["returns"]])
    mrv0 = np.array([median_rv(r) for r in sim_clean["returns"]])
    clean = {
        "rv_over_qv": _aggregate_ratio(rv0, sim_clean["qv"]),
        "bv_over_iv": _aggregate_ratio(bv0, sim_clean["iv"]),
        "medrv_over_iv": _aggregate_ratio(mrv0, sim_clean["iv"]),
    }

    # --- Experiment 1b: continuous + jumps. Validates the jump-variation
    #     decomposition (RV-BV)/JV and the jump count, and reports BV/IV *under
    #     jumps*, which carries a known finite-sample upward bias at M=390 (BV is
    #     jump-robust only as M -> inf), not a 1.00 target. ---------------------
    sim = simulate_many_days(
        days, seed=seed + 5, n_steps=n_steps, ann_vol=0.20, kappa=5.0, vol_of_vol=0.8,
        jump_intensity=0.5, jump_size_vol=0.012, noise_ratio=0.0,
    )
    rv = np.array([realized_variance(r) for r in sim["returns"]])
    bv = np.array([bipower_variation(r) for r in sim["returns"]])
    iv, jv, qv = sim["iv"], sim["jv"], sim["qv"]
    jumps = {
        "rv_over_qv": _aggregate_ratio(rv, qv),
        "bv_over_iv_with_jumps": _aggregate_ratio(bv, iv),
        "rvminusbv_over_jv": _aggregate_ratio(np.maximum(rv - bv, 0.0), jv),
        "mean_jumps_per_day": float(np.mean(sim["n_jumps"])),
    }

    # --- Experiment 2: realized kernel on clean (no-noise, no-jump) data. -----
    sim_c = simulate_many_days(
        days, seed=seed + 1, n_steps=n_steps, ann_vol=0.20, kappa=5.0,
        vol_of_vol=0.8, jump_intensity=0.0, jump_size_vol=0.0, noise_ratio=0.0,
    )
    rv_c = np.array([realized_variance(r) for r in sim_c["returns"]])
    rk_c = np.array([realized_kernel_parzen(r) for r in sim_c["returns"]])
    kernel_clean = {
        "rv_over_qv": _aggregate_ratio(rv_c, sim_c["qv"]),
        "rk_over_qv": _aggregate_ratio(rk_c, sim_c["qv"]),
    }

    # --- Experiment 3: microstructure noise (signature-plot effect). ---------
    sim_n = simulate_many_days(
        days, seed=seed + 2, n_steps=n_steps, ann_vol=0.20, kappa=5.0,
        vol_of_vol=0.8, jump_intensity=0.0, jump_size_vol=0.0, noise_ratio=1.0,
    )
    rv_n = np.array([realized_variance(r) for r in sim_n["returns"]])
    rk_n = np.array([realized_kernel_parzen(r) for r in sim_n["returns"]])
    noisy = {
        "rv_over_qv": _aggregate_ratio(rv_n, sim_n["qv"]),   # inflated
        "rk_over_qv": _aggregate_ratio(rk_n, sim_n["qv"]),   # robust ~1
    }

    # --- Experiment 4: jump-test size (no jumps -> false-positive rate). ------
    sim_size = simulate_many_days(
        days, seed=seed + 3, n_steps=n_steps, ann_vol=0.20, kappa=5.0,
        vol_of_vol=0.8, jump_intensity=0.0, jump_size_vol=0.0, noise_ratio=0.0,
    )
    p_size = np.array([bns_jump_test(r)["p_value"] for r in sim_size["returns"]])
    fpr = float(np.mean(p_size[np.isfinite(p_size)] < JUMP_ALPHA))

    # --- Experiment 5: jump-test power (large jumps -> detection rate). -------
    sim_pow = simulate_many_days(
        days, seed=seed + 4, n_steps=n_steps, ann_vol=0.20, kappa=5.0,
        vol_of_vol=0.8, jump_intensity=1.0, jump_size_vol=0.03, noise_ratio=0.0,
    )
    detect = []
    for r, njump in zip(sim_pow["returns"], sim_pow["n_jumps"]):
        if njump > 0:
            detect.append(bns_jump_test(r)["p_value"] < JUMP_ALPHA)
    detection_rate = float(np.mean(detect)) if detect else float("nan")

    jump_test = {
        "false_positive_rate": fpr,
        "nominal_level": JUMP_ALPHA,
        "detection_rate": detection_rate,
        "n_jump_days_tested": int(len(detect)),
    }

    return {
        "days": days,
        "n_steps": n_steps,
        "seed": seed,
        "clean": clean,
        "jumps": jumps,
        "kernel_clean": kernel_clean,
        "noisy": noisy,
        "jump_test": jump_test,
    }


def _print_table(res: dict) -> None:
    """Pretty-print the validation table to stdout."""
    print(f"\nEstimator validation ({res['days']} simulated days, {res['n_steps']} steps/day)\n")
    rows = [
        ("RV / QV (clean)", res["clean"]["rv_over_qv"], 1.0),
        ("BV / IV (clean)", res["clean"]["bv_over_iv"], 1.0),
        ("medRV / IV (clean)", res["clean"]["medrv_over_iv"], 1.0),
        ("BV / IV (with jumps)", res["jumps"]["bv_over_iv_with_jumps"], None),
        ("(RV - BV) / JV", res["jumps"]["rvminusbv_over_jv"], 1.0),
        ("RK / QV (clean)", res["kernel_clean"]["rk_over_qv"], 1.0),
        ("RV / QV (noise)", res["noisy"]["rv_over_qv"], None),
        ("RK / QV (noise)", res["noisy"]["rk_over_qv"], 1.0),
        ("Jump-test FPR @5%", res["jump_test"]["false_positive_rate"], 0.05),
        ("Jump-test power", res["jump_test"]["detection_rate"], None),
    ]
    print(f"{'Check':<26}{'Result':>10}{'Target':>10}")
    for name, val, target in rows:
        tgt = "n/a" if target is None else f"{target:.2f}"
        print(f"{name:<26}{val:>10.3f}{tgt:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    res = validate(days=args.days, seed=args.seed)
    _print_table(res)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "validation.json"
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
