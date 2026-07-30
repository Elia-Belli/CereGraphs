#!/usr/bin/env bash
# Launches run_snap_sweep.sh inside a detached `screen` session on cer-usn-01,
# so a long-running compile+run job survives independently of whatever ssh
# connection/tool session started it (an ssh-attached background job dies
# with the connection; this doesn't).
#
# Usage (from anywhere -- always cd's to the repo root itself first):
#   ./bfs/bool_diag_spmv/run_snap_sweep_screen.sh [name ...]
#     No args: runs the full 7-graph NAMES list in run_snap_sweep.sh.
#     Named subset (e.g. `skitter patents`): passed straight through.
#
# Check on it later with:
#   ssh cer-usn-01 'screen -r snap_sweep'     # reattach (Ctrl-A D to detach again)
#   ssh cer-usn-01 'screen -list'             # confirm it's still running
#   tail -f bfs/bool_diag_spmv/run_snap_sweep.log   # or just tail the log directly
#
# NOTE: this script itself must be run FROM cer-usn-01 (it needs real
# appliance/cerebras.sdk.client access, same requirement run_snap_sweep.sh
# already has) -- ssh there first, or invoke via:
#   ssh cer-usn-01 'source ~/cs_appliance_sdk/bin/activate && \
#     bash /home/elia/CereGraphs/spmv-hypersparse/bfs/bool_diag_spmv/run_snap_sweep_screen.sh skitter patents'

set -uo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"

SESSION="snap_sweep"

if screen -list | grep -q "\.${SESSION}[[:space:]]"; then
  echo "A screen session named '$SESSION' is already running -- reattach with:" >&2
  echo "  ssh cer-usn-01 'screen -r $SESSION'" >&2
  exit 1
fi

# -dmS: start detached, named session. -L: log session output too (belt and
# braces alongside run_snap_sweep.sh's own log file).
screen -dmS "$SESSION" -L -Logfile "bfs/bool_diag_spmv/run_snap_sweep_screen.$SESSION.out" \
  bash -c "source ~/cs_appliance_sdk/bin/activate && bash bfs/bool_diag_spmv/run_snap_sweep.sh $*"

echo "started detached screen session '$SESSION' running: run_snap_sweep.sh $*"
echo "reattach:  ssh cer-usn-01 'screen -r $SESSION'"
echo "log file:  bfs/bool_diag_spmv/run_snap_sweep_screen.$SESSION.out (or run_snap_sweep.log)"
