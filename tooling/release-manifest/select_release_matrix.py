#!/usr/bin/env python3
"""Filter the release-binaries matrix before any build job is instantiated."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


MATRIX_PATH = Path(__file__).resolve().parent / "release_binaries_matrix.json"


def load_matrix(path: Path = MATRIX_PATH) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"{path} must contain a JSON array")
    return rows


def known_targets(rows: list[dict[str, Any]]) -> list[str]:
    targets: list[str] = []
    for row in rows:
        target = row.get("target")
        if isinstance(target, str):
            targets.append(target)
    return targets


def is_windows_release_row(row: dict[str, Any]) -> bool:
    """Whether a matrix row builds a Windows leg.

    Single predicate for every skip-windows consumer (matrix selection,
    completeness accounting, finalization) so the row set cannot drift
    between them. Matches the `pc-windows-msvc` Rust-target infix shared by
    the neutral host, the GPU plugin legs, and the ARM64 leg.
    """
    target = row.get("target")
    return isinstance(target, str) and "pc-windows-msvc" in target


def without_windows_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the matrix without Windows legs.

    Fail-closed: refusing to produce an empty matrix rather than letting a
    caller release nothing and call it success.
    """
    kept = [row for row in rows if not is_windows_release_row(row)]
    if not kept:
        raise SystemExit("skip-windows filtered every matrix row; refusing an empty release matrix")
    return kept


def select_matrix(
    rows: list[dict[str, Any]], only_target: str, skip_windows: bool = False
) -> list[dict[str, Any]]:
    if skip_windows:
        rows = without_windows_rows(rows)
    if only_target == "":
        return rows
    selected = [row for row in rows if row.get("target") == only_target]
    if not selected:
        if skip_windows:
            raise SystemExit(
                f"unknown only_target {only_target!r} under skip-windows; expected one of: "
                + ", ".join(known_targets(rows))
            )
        expected = ", ".join(known_targets(rows))
        raise SystemExit(
            f"unknown only_target {only_target!r}; expected one of: {expected}"
        )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-target", default="")
    parser.add_argument(
        "--skip-windows",
        action="store_true",
        help="Drop every pc-windows-msvc leg (temporary: no Azure signing). "
        "Formal releases allow only this filter; diagnostics combine it with --only-target.",
    )
    args = parser.parse_args()
    selected = select_matrix(load_matrix(), args.only_target, args.skip_windows)
    sys.stdout.write(json.dumps(selected, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
