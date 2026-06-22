"""Run metadata for results reproducibility.

A small helper so every results JSON can record *how* it was produced — the seed,
the bootstrap replication count, the MCS block-length floor rule, the key package
versions, and the git SHA — making the committed artifacts self-documenting.
"""

from __future__ import annotations

import subprocess
from importlib import metadata

_PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "statsmodels", "arch")


def _git_sha() -> str | None:
    """Return the short git SHA of the working tree, or ``None`` outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _package_versions() -> dict[str, str | None]:
    """Installed versions of the packages whose numerics drive the results."""
    versions: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_meta(seed: int, mcs_reps: int, **extra: object) -> dict:
    """Build a ``meta`` block for a results JSON.

    Parameters
    ----------
    seed, mcs_reps
        The fixed seed and the MCS bootstrap replication count used for the run.
    **extra
        Any run-specific fields (e.g. ``refit_every``) to record alongside.
    """
    meta: dict = {
        "seed": seed,
        "mcs_reps": mcs_reps,
        "block_floor": "max(block_length, horizon + 2)",
        "git_sha": _git_sha(),
        "packages": _package_versions(),
    }
    meta.update(extra)
    return meta
