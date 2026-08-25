"""The fingerprint gate as a contract test: the recorded table hashes must
match the tables on disk. Skips when tables have not been built (fresh clone)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_recorded_table_hashes_match_disk():
    tables = ROOT / "data" / "tables"
    if not any(tables.rglob("hashes.json")):
        pytest.skip("tables not built in this checkout")
    import experiments.verify_fingerprints as V
    rep = V.check_tables(tables, fix=False)
    assert not rep["stale"], f"stale fingerprints: {rep['stale']}"
    assert not rep["missing"], f"recorded tables missing: {rep['missing']}"
