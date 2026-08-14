#!/usr/bin/env bash
# Mirror small experiment artifacts outside the roll-backable working tree.
#
# The working directory sits under .agentmem.incoming-*/ , which an external
# process periodically rolls back to an older snapshot, deleting everything
# created since. Code survives via git; results do not, and they are not
# reproducible without re-running the experiment.
#
# dev/ scripts snapshot themselves through dev/artifacts.py::write_dev_json.
# This script covers what train.py and friends write directly, and doubles as a
# catch-up sweep. Only small files are copied -- checkpoints and per-transition
# arrays are deliberately included, big feature stores are not.
#
# Usage:  bash scripts_snapshot_results.sh [destination]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-/mnt/data/users/luzheng/workspace/ResidualMem_rescue_20260814/results}"
MAX_MB=200

case "$DEST" in
  *".agentmem.incoming"*)
    echo "refusing: destination $DEST is inside the roll-backable tree" >&2
    exit 1 ;;
esac

mkdir -p "$DEST"
copied=0
skipped=0

# Small, non-reproducible artifacts worth keeping.
while IFS= read -r -d '' src; do
  relative="${src#"$REPO"/}"
  size_mb=$(( $(stat -c %s "$src") / 1048576 ))
  if [ "$size_mb" -gt "$MAX_MB" ]; then
    skipped=$((skipped + 1))
    continue
  fi
  target="$DEST/$relative"
  mkdir -p "$(dirname "$target")"
  # Only copy when absent or newer, so repeated sweeps are cheap.
  if [ ! -e "$target" ] || [ "$src" -nt "$target" ]; then
    cp -p "$src" "$target"
    copied=$((copied + 1))
  fi
done < <(find "$REPO/outputs/world_model" "$REPO/outputs/a1" "$REPO/outputs/a2" \
              "$REPO/outputs/probe" "$REPO/refine-logs" \
              \( -name '*.json' -o -name '*.jsonl' -o -name '*.yaml' \
                 -o -name '*.npz' -o -name '*.pkl' -o -name '*.md' \) \
              -type f -print0 2>/dev/null)

echo "snapshot -> $DEST"
echo "  copied  : $copied"
echo "  skipped : $skipped  (larger than ${MAX_MB}MB)"
echo "  total   : $(find "$DEST" -type f | wc -l) files, $(du -sh "$DEST" | cut -f1)"
