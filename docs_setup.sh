#!/usr/bin/env bash
# Wire the submission webapp into the repo. Run from the repo root.
set -euo pipefail

mkdir -p docs/figures

# index.html must already be saved to docs/index.html before running this.
if [ ! -f docs/index.html ]; then
  echo "ERROR: save the downloaded index.html to docs/index.html first" >&2
  exit 1
fi

# The page embeds three figures. Regenerate them, then copy into docs/ so the
# site is self-contained — GitHub Pages serves docs/ as its root, so a
# ../figures/ path would 404 there even though it works locally.
python experiments/m2_figures.py
cp figures/fig1_grid.png figures/fig3_which_side.png figures/fig4_attribution.png docs/figures/

echo
echo "done. open docs/index.html in a browser to check it, then:"
echo "  git add docs && git commit -m 'submission webapp'"
echo
echo "to publish: repo Settings -> Pages -> Source: main branch, /docs folder"
