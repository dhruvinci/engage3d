#!/usr/bin/env bash
# The engagement-demo queue: one clip at a time, one GPU stage at a time, every layer checked before the next clip.
#   bash run_queue.sh            (from anywhere; reads queue.tsv next to this script)
# queue.tsv: NAME <tab> SEQ <tab> CAM <tab> FRAMES <tab> note   (lines starting with # are skipped)
#   Harmony4D: SEQ = sequence name, CAM = camNN, FRAMES = first:last:step
#   folder clip (engage_clip_make.py): SEQ = ../benchmarks/engage_h4d/clips/NAME, CAM and FRAMES = -
# A clip is done when NAME/GATE.json says passed. A stage error or a failed gate writes queue.blocked and stops the
# queue (fix, delete queue.blocked, start again). touch queue.pause to hold between clips; touch queue.stop to end.
set -uo pipefail
Q=$(cd "$(dirname "$0")" && pwd)
cd "$Q/../../../scripts"
B=../benchmarks/engage_h4d/demo_layers
export PYTORCH_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') $*" | tee -a "$Q/queue.log"; }
while pgrep -f "bash run_clip.sh" > /dev/null; do sleep 30; done           # the hand-started clips finish first
while true; do
  [[ -e $Q/queue.stop ]] && { log "stop requested"; exit 0; }
  [[ -e $Q/queue.blocked ]] && { log "blocked: $(cat $Q/queue.blocked)"; exit 1; }
  while [[ -e $Q/queue.pause ]]; do sleep 30; done
  next=""
  while IFS=$'\t' read -r NAME SEQ CAM WIN NOTE; do
    [[ -z "$NAME" || "$NAME" == \#* ]] && continue
    grep -q '"passed": true' "$B/$NAME/GATE.json" 2>/dev/null && continue
    next="$NAME"; break
  done < "$Q/queue.tsv"
  [[ -z "$next" ]] && { log "queue empty"; exit 0; }
  S=$B/${NAME}_seed; F=$B/$NAME; L=$Q/$NAME.log
  mkdir -p $S $F
  A="--seq $SEQ"; [[ "$CAM" != "-" ]] && A="$A --cam $CAM"; [[ "$WIN" != "-" ]] && A="$A --frames $WIN"
  log "== $NAME ($NOTE)"
  t0=$(date +%s)
  stage() {   # env out stage [args]: skip when its output exists (resume), else run; any error blocks the queue
    local env=$1 out=$2 st=$3; shift 3
    local done_file=""
    case $st in detect) done_file=detect.json;; track|samtrack) done_file=track.json;; pose|pose_pmpose) done_file=pose.npz;;
                segment) done_file=segment.npz;; body3d) done_file=body3d.npz;; place) done_file=place.json;; contact) done_file=contact.json;;
                context) done_file=context.json;; score) done_file=scorecard.json;; render) done_file=demo.gif;; esac
    if [[ -n "$done_file" && -e $out/$done_file ]]; then echo "  $st: kept" >> $L; return 0; fi
    echo "$(date '+%T') $st $*" >> $L
    ../.venv-$env/bin/python engage_demo_layers.py $st --out $out $A "$@" >> $L 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
      log "   $st FAILED (exit $rc) - see $NAME.log"; echo "$NAME: stage $st failed (exit $rc)" > $Q/queue.blocked; return 1
    fi
    grep -E "^(lens|detect|track|samtrack|pose|segment|body3d|place|contact|wrote)" $L | tail -1 | sed 's/^/   /' | tee -a $Q/queue.log
  }
  [[ "$CAM" == "-" ]] && { stage depth $F lens || continue; }
  if [[ ! -e $F/track.json ]]; then   # seed chain: boxes, identities, and masks where the two stand apart
    stage rfdetr $S detect && stage depth $S track && stage pose $S pose --seeds-only && stage bmp $S segment --seeds-only || continue
  fi
  stage sam31 $F samtrack --seed $S && stage bmp $F pose_pmpose --variant PMPose-b && stage s3d $F body3d \
    && stage depth $F place && stage depth $F contact && stage depth $F context && stage depth $F score && stage depth $F render || continue
  if ../.venv-depth/bin/python engage_demo_layers.py check --out $F $A >> $L 2>&1; then
    log "   PASSED all layer gates in $(( ($(date +%s) - t0) / 60 )) min"
  else
    log "   GATE FAILED: $(tr '\n' ';' < $F/GATE_FAILED.txt)"; echo "$NAME: gate failed" > $Q/queue.blocked
  fi
done
