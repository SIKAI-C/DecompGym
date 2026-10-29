#!/usr/bin/env bash
# Start, stop and resume the harder-family campaign without losing finished work.
#
#   scripts/harder_campaign.sh start       # resume the campaign and supervise it in the background
#   scripts/harder_campaign.sh stop        # graceful: start no new pair, finish the pairs in flight, then pause
#   scripts/harder_campaign.sh stop-now    # immediate: kill the agents and solvers; the interrupted attempt resumes later
#   scripts/harder_campaign.sh status      # progress per model, controller state and the last resource sample
#   scripts/harder_campaign.sh dashboard   # serve the progress dashboard at http://localhost:8765
#
# Gemini (agy) runs only when DECOMPBENCH_AGY_KEYRING_PASSPHRASE is set in the shell that calls `start`.
# Before closing the laptop lid, run `stop-now`: sleep would suspend Docker and the timed solver runs.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-/Users/scheng326/Desktop/GT_ML_PhD/z_oo/code/DecompBench/.venv/bin/python}
ROOT=${ROOT:-outputs/experiments/harder_small_all_20261029_r1}
ROOT=$(cd "$ROOT" && pwd)

watchdog_pid() {
  # Print the live watchdog PID of this root, or nothing.
  [ -f "$ROOT/watchdog.json" ] || return 0
  local pid
  pid=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' "$ROOT/watchdog.json")
  if ps -p "$pid" -o command= 2>/dev/null | grep -q -- "watch --root $ROOT"; then
    echo "$pid"
  fi
}

models() {
  # Config order; Gemini only with its keyring passphrase.
  "$PY" - "$ROOT/campaign.json" <<'EOF'
import json, os, sys
for item in json.load(open(sys.argv[1]))["model_queue"]:
    if item["backend"] != "agy-cli" or os.environ.get("DECOMPBENCH_AGY_KEYRING_PASSPHRASE"):
        print(item["model"])
EOF
}

case "${1:-}" in
  start)
    if [ -n "$(watchdog_pid)" ]; then
      echo "Already running (watchdog PID $(watchdog_pid))."; exit 0
    fi
    docker info >/dev/null 2>&1 || { echo "Docker is not running; start Docker Desktop first."; exit 1; }
    args=()
    while read -r model; do args+=(--model "$model"); done < <(models)
    "$PY" src/02_experiment.py resume --root "$ROOT"
    nohup "$PY" src/02_experiment.py watch --root "$ROOT" "${args[@]}" >>"$ROOT/watch.log" 2>&1 &
    pid=$!
    # Keep the Mac awake (not the display) while the watchdog lives.
    nohup caffeinate -i -w "$pid" >/dev/null 2>&1 &
    echo "Started watchdog PID $pid for: ${args[*]//--model /}"
    echo "Log: $ROOT/watch.log"
    ;;
  stop)
    touch "$ROOT/STOP"
    echo "Stop requested: no new pair starts; the pairs in flight finish (minutes to about an hour), then the campaign pauses."
    echo "Use 'stop-now' to stop immediately instead."
    ;;
  stop-now)
    pid=$(watchdog_pid)
    if [ -z "$pid" ]; then echo "No watchdog is running."; exit 0; fi
    kill -TERM "$pid"
    for _ in $(seq 1 60); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 2
    done
    if kill -0 "$pid" 2>/dev/null; then echo "Watchdog $pid is still shutting down; check again with 'status'."; exit 1; fi
    left=$(docker ps -q --filter "label=io.decompbench.runtime=v2-worker" 2>/dev/null | wc -l | tr -d ' ')
    echo "Stopped. Benchmark agent containers still running: $left. Finished attempts are kept; 'start' resumes from the interrupted attempt."
    ;;
  status)
    "$PY" src/02_experiment.py status --root "$ROOT"
    pid=$(watchdog_pid)
    echo "watchdog: ${pid:-not running}"
    [ -f "$ROOT/status.json" ] && "$PY" -c 'import json,sys; s=json.load(open(sys.argv[1])); print("controller:", s.get("state"), "| active:", [(v["model"], v["instance"], v.get("mode"), v.get("attempt"), v.get("phase")) for v in s.get("active_instances", {}).values()])' "$ROOT/status.json"
    [ -f "$ROOT/monitor/status.json" ] && "$PY" -c 'import json,sys; s=json.load(open(sys.argv[1])); print("resources:", s.get("state"), "cpu %.0f%%" % s.get("cpu_percent", 0), "rss %.0f MiB" % s.get("rss_mib", 0), "| peaks:", s.get("peaks"))' "$ROOT/monitor/status.json"
    [ -f "$ROOT/STOP" ] && echo "graceful stop pending"
    true
    ;;
  dashboard)
    nohup "$PY" src/02_experiment.py dashboard --root "$ROOT" >>"$ROOT/dashboard.log" 2>&1 &
    echo "Dashboard PID $!: http://localhost:8765"
    ;;
  *)
    sed -n '2,12p' "$0"; exit 2
    ;;
esac
