"""Fingerprint verification — every code table on disk must match its record.

Run manifests store a table NAME (e.g. ``wave768_l2``), not a hash: the frozen
trainer trusts ``data/tables/<name>.pt`` to be the table that name had when the
run happened. This gate makes that trust checkable:

1. For every ``hashes.json`` under the tables root, recompute the SHA-256 of
   each recorded ``.pt`` and compare — **OK / STALE / MISSING**. Tables present
   but unrecorded are flagged (warning, not failure).
2. Bind every run to disk truth: for each ``results/**/manifest.json`` naming a
   table, record that table's *current* hash into
   ``results/fingerprints_audit.json`` — the sidecar manifests never had.
3. ``--fix`` rewrites each ``hashes.json`` from disk truth (after you have
   satisfied yourself the disk is right, not the record).

Exit 1 on any STALE or MISSING recorded table — a CI gate:

    python experiments/verify_fingerprints.py
    python experiments/verify_fingerprints.py --fix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import torch

from kronecker_v2.tables import table_hash

_CACHE: dict[Path, str] = {}


def current_hash(p: Path) -> str:
    if p not in _CACHE:
        _CACHE[p] = table_hash(torch.load(p, map_location="cpu",
                                          weights_only=True))
    return _CACHE[p]


def check_tables(tables_root: Path, fix: bool) -> dict:
    report = {"ok": [], "stale": [], "missing": [], "unrecorded": []}
    for hj in sorted(tables_root.rglob("hashes.json")):
        d = hj.parent
        recorded = json.loads(hj.read_text())
        for name, want in sorted(recorded.items()):
            p = d / f"{name}.pt"
            if not p.exists():
                report["missing"].append(str(p))
                print(f"  MISSING     {p}")
                continue
            got = current_hash(p)
            if got == want:
                report["ok"].append(str(p))
                print(f"  ok          {p}")
            else:
                report["stale"].append(str(p))
                print(f"  STALE       {p}\n"
                      f"              recorded {want[:16]}… disk {got[:16]}…")
        for p in sorted(d.glob("*.pt")):
            if p.stem not in recorded:
                report["unrecorded"].append(str(p))
                print(f"  UNRECORDED  {p}  (warning)")
        if fix:
            truth = {p.stem: current_hash(p) for p in sorted(d.glob("*.pt"))}
            hj.write_text(json.dumps(truth, indent=2))
            print(f"  fixed       {hj}  ({len(truth)} entries from disk)")
    return report


def bind_runs(results_root: Path, out: Path) -> dict:
    audit: dict = {}
    for mf in sorted(results_root.rglob("manifest.json")):
        cfg = json.loads(mf.read_text()).get("cfg", {})
        table = cfg.get("codec", {}).get("table")
        run = str(mf.parent.relative_to(results_root))
        if not table:
            audit[run] = {"table": None, "status": "no-table (learned arm)"}
            continue
        p = Path(cfg.get("data", {}).get("tables_dir", "data/tables")) / f"{table}.pt"
        if not p.is_absolute():
            p = _ROOT / p
        if not p.exists():
            audit[run] = {"table": table, "file": str(p),
                          "status": "TABLE MISSING"}
            print(f"  RUN {run}: table file missing -> {p}")
        else:
            audit[run] = {"table": table, "file": str(p),
                          "sha256": current_hash(p), "status": "bound"}
    out.write_text(json.dumps(audit, indent=2))
    n = sum(1 for v in audit.values() if v["status"] == "bound")
    print(f"  bound {n}/{len(audit)} runs -> {out}")
    return audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables-root", type=Path, default=_ROOT / "data" / "tables")
    ap.add_argument("--results-root", type=Path, default=_ROOT / "results")
    ap.add_argument("--fix", action="store_true")
    args = ap.parse_args()

    print(f"tables under {args.tables_root}:")
    rep = check_tables(args.tables_root, fix=args.fix)
    print(f"\nrun binding under {args.results_root}:")
    if args.results_root.exists():
        bind_runs(args.results_root, args.results_root / "fingerprints_audit.json")

    print(f"\n{len(rep['ok'])} ok · {len(rep['stale'])} stale · "
          f"{len(rep['missing'])} missing · {len(rep['unrecorded'])} unrecorded")
    if rep["stale"] or rep["missing"]:
        print("FINGERPRINT GATE: FAIL")
        raise SystemExit(1)
    print("FINGERPRINT GATE: PASS")


if __name__ == "__main__":
    main()
