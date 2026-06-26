#!/usr/bin/env bash
# sync_to_work.sh — Sync LS2D results to the DALES work directory
#
# Usage:
#   ./tools/sync_to_work.sh [expnr]
#
# Examples:
#   ./tools/sync_to_work.sh        # syncs experiment 019 (default)
#   ./tools/sync_to_work.sh 19     # syncs experiment 019
#   ./tools/sync_to_work.sh 5      # syncs experiment 005

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_ROOT="/Users/yunpeichu/LS2D-ICON/results/mpcseed"
WORK_DIR="/Users/yunpeichu/work_dales/mpcseed/input"

# ── Experiment number ─────────────────────────────────────────────────────────
EXPNR="${1:-19}"
EXPNR_FMT=$(printf "%03d" "$EXPNR")

SRC="${RESULTS_ROOT}/run_site_based_${EXPNR_FMT}/"

# ── Validate source ───────────────────────────────────────────────────────────
if [[ ! -d "$SRC" ]]; then
    echo "ERROR: Source directory does not exist: $SRC" >&2
    exit 1
fi

mkdir -p "$WORK_DIR"

# ── Sync ──────────────────────────────────────────────────────────────────────
echo "Syncing experiment ${EXPNR_FMT}"
echo "  From: $SRC"
echo "  To:   $WORK_DIR"
echo ""

rsync -av --progress "$SRC" "$WORK_DIR"

echo ""
echo "Done."
