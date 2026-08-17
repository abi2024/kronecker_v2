"""Run any analysis script with the full codec patch chain installed.

The frozen scorer (``m2_bucket_analysis.score_arm``) rebuilds each arm's
embedding from its manifest via ``T.build_wte``. That function is widened by
import-time patches — m5_train adds bag/rp, m3_train adds hash/albert — but the
patch exists only in a process that imported those modules. Analysis scripts
run standalone, so scoring an M3 results directory dies on the first learned
arm with ``KeyError: 'table'``.

This wrapper imports the full chain first, then forwards to whichever script
you name. It works for every current and future analysis without another
wrapper file:

    python experiments/patched.py m2_bucket_analysis --bucket-by prev-collision \
        --root results/m3 --data data/m3 --baseline m3_onehot
    python experiments/patched.py m3_script_analysis \
        --root results/m3 --data data/m3 --baseline m3_onehot

Rebuild-from-manifest stays correct for the learned arms because of the
numpy-PCG64 discipline: hash's bucket assignment is a non-persistent buffer
regenerated from its seed, and rp's Gaussian matrix likewise — bit-identical on
every platform, then the checkpoint restores the trained weights over them.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from experiments import m3_train                  # noqa: F401  — installs the chain


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    name = sys.argv[1].removesuffix(".py")
    mod = importlib.import_module(f"experiments.{name}")
    sys.argv = [name] + sys.argv[2:]
    mod.main()


if __name__ == "__main__":
    main()
