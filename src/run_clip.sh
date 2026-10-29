#!/usr/bin/env bash
# One clip through the engagement demo's final stack (scripts/engage_demo_layers.py), one GPU stage at a time.
#   bash run_clip.sh NAME SEQ CAM FIRST:LAST:STEP
# Seed chain (RF-DETR -> tracker -> ViTPose -> SAM-pose2seg) in demo_layers/NAME_seed, then the final stack
# (SAM 3.1 video tracking, PMPose, SAM 3D Body, contact, Astra, score, render) in demo_layers/NAME.
set -uo pipefail
NAME=$1; SEQ=$2; CAM=$3; WIN=$4
cd "$(dirname "$0")/../../../scripts"
B=../benchmarks/engage_h4d/demo_layers; S=$B/${NAME}_seed; F=$B/$NAME
mkdir -p $S $F
export PYTORCH_ALLOC_CONF=expandable_segments:True
A="--seq $SEQ --cam $CAM --frames $WIN"
log() { echo "$(date '+%F %T') [$NAME] $*"; }
run() { local env=$1; shift; log "$*"; ../.venv-$env/bin/python engage_demo_layers.py "$@" $A 2>&1 | grep -vE "Warning|warn|INFO|^\s*$" | tail -3; }
run rfdetr detect --out $S
run depth track --out $S
run pose pose --out $S
run bmp segment --out $S
run sam31 samtrack --seed $S --out $F
run bmp pose_pmpose --variant PMPose-b --out $F
run s3d body3d --out $F
run depth contact --out $F
log "context (Astra)"; ../.venv-depth/bin/python engage_demo_layers.py context --out $F $A > $F/context.log 2>&1; tail -2 $F/context.log
run depth score --out $F
run depth render --out $F
log done
