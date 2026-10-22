#!/bin/bash
# Lane driver v3 (2026-10-02).  usage:  ./run_lane.sh <queue_dir> <log_prefix> <driver_log_name>
#   e.g.  ./run_lane.sh queues ""  driver.log      (lane A)
#         ./run_lane.sh queues_b B_ driver_b.log   (lane B)
#         ./run_lane.sh queues_c C_ driver_c.log   (lane C)
#
# Why v3: v1/v2 trusted the python exit code / a done/ marker.  On this shared unified-memory box the kernel OOM-killer
# killed 46 of our processes (cgroup memory.events oom_kill); each kill ended attrib.py mid-queue, v1 touched done/ anyway,
# and the lane silently raced through its remaining queue files (2026-10-02: ~15 queue files "finished" in hours with
# almost no results).  Source of truth is now the RESULTS THEMSELVES:
#   * a queue file is complete only when every cell has results/attrib/<name>.json (queue_missing.py);
#   * done/ markers are ignored; incomplete files are re-run (attrib.py skips finished cells and resumes training from
#     the per-cell checkpoint), so an OOM kill just costs <=A_CKPT_EVERY steps;
#   * a file is parked in failed/ only after 3 consecutive attempts that made NO progress (no new result AND no new
#     checkpoint) -- deterministic failures can't loop forever, OOM kills (which do make progress) keep retrying;
#   * an attrib.py already running on this lane's queue dir (orphan from an older driver) is adopted: we wait for it.
Q=$1; PFX=$2; DL=logs/$3
cd ~/p22 || exit 1
export HF_HOME=~/hf_cache
export A_OUT=results/attrib
mkdir -p "$Q/failed" "$Q/parked" logs "$A_OUT/ckpt"
log() { echo "[$(date '+%F %T')] $*" >> "$DL"; }
log "lane driver v3 start ($Q)"
while true; do
  [ -f "$Q/STOP" ] && { log "STOP seen, exiting"; break; }
  while pgrep -f "scripts/attrib.py $Q/" > /dev/null; do sleep 30; done   # adopt an orphaned run
  next=""
  for f in "$Q"/*.json; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    [ -e "$Q/failed/$b" ] && continue
    m=$(python3 scripts/queue_missing.py "$f") || m=-1
    [ "$m" = "0" ] && continue
    next="$f"; before=$m; break
  done
  if [ -z "$next" ]; then sleep 60; continue; fi
  b=$(basename "$next" .json)
  AF="$Q/failed/.attempts_$b"
  np=$(cat "$AF" 2>/dev/null || echo 0)
  stamp=$(mktemp);
  log "running $next (missing cells: $before, no-progress attempts so far: $np)"
  ~/venvs/p22/bin/python scripts/attrib.py "$next" >> "logs/${PFX}$b.log" 2>&1
  rc=$?
  after=$(python3 scripts/queue_missing.py "$next") || after=-1
  newckpt=$(find "$A_OUT/ckpt" -type f -newer "$stamp" 2>/dev/null | head -1)
  if [ "$after" = "0" ]; then
    rm -f "$AF"; log "COMPLETE $next (exit $rc)"
  else
    if [ "$after" != "-1" ] && [ "$after" -lt "$before" ] || [ -n "$newckpt" ]; then np=0; else np=$((np+1)); fi
    echo "$np" > "$AF"
    log "INCOMPLETE $next: $after cell(s) still missing (exit $rc); no-progress attempts=$np"
    if [ "$np" -ge 3 ]; then
      touch "$Q/failed/$b"; log "*** GAVE UP on $next after 3 attempts with no progress -- parked in failed/, needs a human ***"
    else
      sleep 20
    fi
  fi
  rm -f "$stamp"
done
