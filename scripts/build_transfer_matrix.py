#!/usr/bin/env python3
"""Build the Q5 cross-asset transfer matrix (model x class) from per-class MCS results.

Assembles the committed per-class results JSONs into the unified
"Liu-Patton-Sheppard-for-models" transfer matrix: where the HAR family stays in /
leaves the 90% MCS across asset classes and horizons, plus the refinement
(HARQ / LogSHAR vs LogHAR) transfer that explains *why*. Read-only on results/.

Sources (committed):
  summary.json         8 equity indices (headline model set)
  har_family.json      equity HAR-family refinements (DM vs LogHAR)
  crypto.json          4 coins (exploratory)
  crypto_expanded.json 22 coins (confirmatory, survivorship-corrected)
  volare_futures.json  13 futures (by sub-class)

Outputs (results/ + results/tables/):
  transfer_matrix.json, transfer_matrix_verdict.csv,
  transfer_matrix_mcs_h1.csv, transfer_matrix_mcs_h22.csv,
  transfer_matrix_refinement.csv, transfer_matrix.md
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TAB = RES / "tables"
HORIZONS = ["1", "5", "22"]
HAR_FAMILY = {"HAR", "LogHAR", "HAR-J", "LogHAR-J", "HAR-CJ", "LogHAR-CJ", "SHAR", "LogSHAR"}
# model display order for the MCS-survival matrix
MODEL_ORDER = [
    "RW", "HistMean", "MA22", "EWMA", "AR1Log", "ARFIMA",
    "HAR", "LogHAR", "HAR-J", "LogHAR-J", "HAR-CJ", "LogHAR-CJ", "SHAR", "LogSHAR",
    "HARQ", "GBRT",
]


def load(name):
    return json.loads((RES / f"{name}.json").read_text())


summary = load("summary")
harfam = load("har_family")
crypto4 = load("crypto")
crypto22 = load("crypto_expanded")
futures = load("volare_futures")


def modal(counter: Counter):
    if not counter:
        return None
    m, n = counter.most_common(1)[0]
    return f"{m} ({n}/{sum(counter.values())})"


def harfam_in_count(inst: dict) -> int:
    c = 0
    for d in inst.values():
        if "har_family_in_mcs" in d:
            c += 1 if d["har_family_in_mcs"] else 0
        elif any(m in HAR_FAMILY for m in d.get("mcs", [])):
            c += 1
    return c


def summarize_instruments(inst: dict, verdict_counts: dict | None = None) -> dict:
    """For classes stored as per-instrument dicts with 'mcs' lists + 'best'."""
    mcs = Counter()
    for d in inst.values():
        for m in d.get("mcs", []):
            mcs[m] += 1
    if verdict_counts is None:
        verdict_counts = dict(Counter(d["verdict"] for d in inst.values()))
    verdict = max(("dominates", "competitive", "degrades"),
                  key=lambda v: verdict_counts.get(v, 0))
    cracks = [{"name": k, "verdict": d["verdict"], "best": d["best"]}
              for k, d in sorted(inst.items())
              if d["verdict"] in ("competitive", "degrades")]
    return dict(
        n=len(inst),
        mcs=dict(mcs),
        ran=sorted(mcs),  # models that appeared in any MCS for this class
        harfam_in=harfam_in_count(inst),
        best=modal(Counter(d["best"] for d in inst.values())),
        verdict=verdict,
        verdict_counts=verdict_counts,
        cracks=cracks,
        harq_bl=sum(1 for d in inst.values()
                    if d.get("q1_harq_vs_loghar", {}).get("beats_loghar")),
        logshar_bl=sum(1 for d in inst.values()
                       if d.get("q2_logshar_vs_loghar", {}).get("beats_loghar")),
    )


def equities(h):
    hh = summary["by_horizon"][h]
    mcs = hh["mcs_count"]
    harfam_in = max((mcs.get(m, 0) for m in HAR_FAMILY), default=0)  # LogHAR is in all
    return dict(
        n=hh["n_indices"], mcs=dict(mcs), ran=sorted(mcs),
        harfam_in=harfam_in,
        best=modal(Counter(v["model"] for v in hh["mz_best"].values())) + " [MZ-best]",
        verdict="dominates",  # LogHAR in MCS for all 8 at every horizon (derived)
        verdict_counts={"dominates": hh["n_indices"]},
        cracks=[],
        harq_bl=None,  # equity HARQ needs intraday bars -> simulation track only
        logshar_bl=harfam["by_horizon"][h]["beats_loghar"].get("LogSHAR"),
    )


def crypto_small(h):
    hh = crypto4["by_horizon"][h]
    mcs = hh["mcs_count"]
    ranks = hh["avg_rank"]
    return dict(
        n=hh["n_coins"], mcs=dict(mcs), ran=sorted(mcs),
        harfam_in=max((mcs.get(m, 0) for m in HAR_FAMILY), default=0),
        best=f"{min(ranks, key=ranks.get)} [min avg-rank]",
        verdict="dominates",
        verdict_counts={"dominates": hh["n_coins"]},
        cracks=[],
        harq_bl=f"{hh.get('harq_beats_har')} (vs HAR)",  # crypto.json benchmarks HARQ vs HAR
        logshar_bl=None,  # LogSHAR not in the 4-coin headline set
    )


def fut_subset(h, subclass=None):
    per = futures["by_horizon"][h]["per_contract"]
    inst = {k: v for k, v in per.items() if subclass is None or v["subclass"] == subclass}
    vc = None
    if subclass is not None:
        vc = futures["by_horizon"][h]["by_subclass"].get(subclass)
    return summarize_instruments(inst, vc)


# ---- assemble classes (label -> per-horizon record) ----
CLASSES = [
    ("Equities (8)", "exploratory", equities),
    ("Crypto — 4 coins", "exploratory", crypto_small),
    ("Crypto — 22 coins", "confirmatory",
     lambda h: summarize_instruments(crypto22["by_horizon"][h]["per_coin"],
                                     crypto22["by_horizon"][h]["verdict_counts"])),
    ("Futures — rates (FV/TY)", "confirmatory", lambda h: fut_subset(h, "rates")),
    ("Futures — commodity (8)", "confirmatory", lambda h: fut_subset(h, "commodity")),
    ("Futures — equity-index (ES/NQ)", "confirmatory", lambda h: fut_subset(h, "equity_index")),
    ("Futures — fx (EU)", "confirmatory", lambda h: fut_subset(h, "fx")),
    ("Futures — all (13)", "confirmatory", lambda h: fut_subset(h, None)),
]

matrix = {label: {"track": track, "by_horizon": {h: fn(h) for h in HORIZONS}}
          for label, track, fn in CLASSES}

# ---------------------------------------------------------------- writers
TAB.mkdir(exist_ok=True)
(RES / "transfer_matrix.json").write_text(json.dumps(matrix, indent=2))


def verdict_cell(rec):
    n = rec["n"]
    dom = rec["verdict_counts"].get("dominates", 0)
    parts = [f"DOMINATES ({n}/{n})" if dom == n else f"DOMINATES {dom}/{n}"]
    for cr in rec.get("cracks", []):
        tag = "DEGRADE" if cr["verdict"] == "degrades" else "competitive"
        parts.append(f"{tag}: {cr['name']}→{cr['best']}")
    parts.append(f"best {rec['best']}")
    if rec["harfam_in"] < n:
        parts.append(f"HAR-fam MCS {rec['harfam_in']}/{n}")
    return " · ".join(parts)


# Table A — verdict matrix
with (TAB / "transfer_matrix_verdict.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["asset_class", "track"] + [f"h={h}" for h in HORIZONS])
    for label in matrix:
        rec = matrix[label]
        w.writerow([label, rec["track"]] + [verdict_cell(rec["by_horizon"][h]) for h in HORIZONS])


# Table C — model x class MCS survival at a given horizon
def mcs_cell(label, rec, model):
    h_rec = rec
    n = h_rec["n"]
    if model in h_rec["mcs"]:
        return f"{h_rec['mcs'][model]}/{n}"
    # equity HAR-family refinements ran only in a within-family MCS (not comparable)
    if label == "Equities (8)" and model in (HAR_FAMILY - {"HAR", "LogHAR"}):
        return "n/c"
    if label == "Equities (8)" and model == "HARQ":
        return "sim-only"
    if model in h_rec.get("ran", []):
        return f"0/{n}"
    return "–"


for h in ("1", "22"):
    with (TAB / f"transfer_matrix_mcs_h{h}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + list(matrix.keys()))
        for model in MODEL_ORDER:
            row = [model]
            for label in matrix:
                row.append(mcs_cell(label, matrix[label]["by_horizon"][h], model))
            w.writerow(row)

# Table B — refinement transfer (HARQ / LogSHAR beats LogHAR, by class x horizon)
with (TAB / "transfer_matrix_refinement.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["asset_class", "refinement"] + [f"h={h}" for h in HORIZONS])
    for label in matrix:
        recs = matrix[label]["by_horizon"]
        for ref, key in (("HARQ beats LogHAR", "harq_bl"), ("LogSHAR beats LogHAR", "logshar_bl")):
            cells = []
            for h in HORIZONS:
                v = recs[h][key]
                n = recs[h]["n"]
                if v is None:
                    cells.append("n/a")
                elif isinstance(v, str):
                    cells.append(v)
                else:
                    cells.append(f"{v}/{n}")
            w.writerow([label, ref] + cells)

# ---------------------------------------------------------------- markdown
lines = []
lines.append("# Q5 — Cross-asset transfer matrix\n")
lines.append("*The Liu–Patton–Sheppard analog for models: where the HAR family stays in / "
             "leaves the 90% MCS (QLIKE) across asset classes. Generated by "
             "`scripts/build_transfer_matrix.py` from the committed per-class results.*\n")
lines.append("\n## A. Verdict by class × horizon\n")
lines.append("Verdict per the pre-registration: **DOMINATES** = HAR family in the MCS and "
             "single-best; **competitive** = HAR in MCS but a non-HAR ties/leads for some "
             "instruments; **DEGRADE** = HAR family displaced as single-best (DM-significant) "
             "or excluded from the MCS.\n")
lines.append("| Asset class | Track | h = 1 | h = 5 | h = 22 |")
lines.append("|---|---|---|---|---|")
for label in matrix:
    rec = matrix[label]
    cells = [verdict_cell(rec["by_horizon"][h]) for h in HORIZONS]
    lines.append(f"| {label} | {rec['track']} | {cells[0]} | {cells[1]} | {cells[2]} |")

lines.append("\n## B. Refinement transfer — does the equity-tuned refinement carry over?\n")
lines.append("Count of instruments where the refinement **DM-beats LogHAR** (α = 0.05), per class.\n")
lines.append("| Asset class | Refinement | h = 1 | h = 5 | h = 22 |")
lines.append("|---|---|---|---|---|")
for label in matrix:
    recs = matrix[label]["by_horizon"]
    for ref, key in (("HARQ vs LogHAR", "harq_bl"), ("LogSHAR vs LogHAR", "logshar_bl")):
        cells = []
        for h in HORIZONS:
            v = recs[h][key]
            n = recs[h]["n"]
            cells.append("n/a" if v is None else (v if isinstance(v, str) else f"{v}/{n}"))
        lines.append(f"| {label} | {ref} | {cells[0]} | {cells[1]} | {cells[2]} |")

for h in ("1", "22"):
    lines.append(f"\n## C. Model 90% MCS survival (k / n instruments) — h = {h}\n")
    lines.append("`n/c` = ran in a separate within-family MCS (not comparable); "
                 "`sim-only` = real-data run not available; `–` = model not in this class's set.\n")
    header = "| Model | " + " | ".join(matrix.keys()) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(matrix) + 1))
    for model in MODEL_ORDER:
        row = [model]
        for label in matrix:
            row.append(mcs_cell(label, matrix[label]["by_horizon"][h], model))
        lines.append("| " + " | ".join(row) + " |")

(TAB / "transfer_matrix.md").write_text("\n".join(lines) + "\n")

print("\n".join(lines))
print("\n[written] results/transfer_matrix.json + results/tables/transfer_matrix_*.csv|md")
