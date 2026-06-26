#!/usr/bin/env bash
# sync_to_server.sh — Sync LS2D results to the remote DALES work directory
#
# Usage:
#   ./tools/sync_to_server.sh [expnr]
#
# Examples:
#   ./tools/sync_to_server.sh        # syncs experiment 019 (default)
#   ./tools/sync_to_server.sh 19     # syncs experiment 019
#   ./tools/sync_to_server.sh 5      # syncs experiment 005

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
# Replace with your actual SSH alias (e.g., "hpc") or "username@hostname"
SERVER_HOST="vrlab" 

RESULTS_ROOT="/Users/yunpeichu/LS2D-ICON/results/mpcseed"
WORK_DIR="/home/yunpei/work_dales/mpcseed/input"

# ── Experiment number ─────────────────────────────────────────────────────────
EXPNR="${1:-19}"
EXPNR_FMT=$(printf "%03d" "$EXPNR")

SRC="${RESULTS_ROOT}/run_site_based_${EXPNR_FMT}/"

# ── Validate source ───────────────────────────────────────────────────────────
if [[ ! -d "$SRC" ]]; then
    echo "ERROR: Source directory does not exist: $SRC" >&2
    exit 1
fi

# Ensure the remote directory exists before syncing
echo "Verifying remote directory..."
ssh "$SERVER_HOST" "mkdir -p '$WORK_DIR'"

# ── Sync ──────────────────────────────────────────────────────────────────────
echo "Syncing experiment ${EXPNR_FMT}"
echo "  From: $SRC"
echo "  To:   ${SERVER_HOST}:${WORK_DIR}"
echo ""

rsync -av --progress --exclude='scalar.inp.${EXPNR_FMT}' "$SRC" "${SERVER_HOST}:${WORK_DIR}"

echo ""
echo "Done."