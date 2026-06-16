"""Command-line entry point: ``volbench run --horizon H --ticker .SPX``.

A thin wrapper over :func:`volbench.backtest.run_backtest` for quick one-off
runs without writing a script. The full reproduction pipeline lives in
``scripts/``.
"""

from __future__ import annotations

import argparse

from .backtest import run_backtest
from .data import load_oxford_rv


def _cmd_run(args: argparse.Namespace) -> int:
    ds = load_oxford_rv()
    if args.ticker not in ds.tickers:
        raise SystemExit(f"unknown ticker {args.ticker!r}; available: {ds.tickers}")
    rv = ds.series(args.ticker)
    res = run_backtest(rv, horizon=args.horizon, mcs_reps=args.mcs_reps, seed=args.seed)
    ranked = sorted(res.mean_losses["QLIKE"].items(), key=lambda kv: kv[1])
    print(f"\n{args.ticker}  h={args.horizon}  ({res.origins.size} test origins)")
    print(f"{'model':<10}{'QLIKE':>10}{'MSE-var':>14}{'in MCS':>9}")
    in_mcs = set(res.mcs["QLIKE"].included)
    for name, q in ranked:
        mse = res.mean_losses["MSE-var"][name]
        print(f"{name:<10}{q:>10.4f}{mse:>14.3e}{'  yes' if name in in_mcs else '   no':>9}")
    print(f"\n90% MCS (QLIKE): {sorted(in_mcs)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns a process exit code."""
    parser = argparse.ArgumentParser(prog="volbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the benchmark on one index/horizon")
    run.add_argument("--ticker", default=".SPX", help="index symbol (default: .SPX)")
    run.add_argument("--horizon", type=int, default=1, help="forecast horizon in days")
    run.add_argument("--mcs-reps", type=int, default=2000, dest="mcs_reps")
    run.add_argument("--seed", type=int, default=0)
    run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
