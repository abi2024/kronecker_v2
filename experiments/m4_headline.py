"""Milestone 4 — 124M / 2.5B tokens / 3 seeds, all arms, plus the robustness probe."""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/")
    args = ap.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
