"""Milestone 5 — tokens longer than 32 bytes: show one-hot emits identical vectors, wave does not."""

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
