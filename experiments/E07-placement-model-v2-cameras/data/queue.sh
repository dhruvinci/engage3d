#!/bin/bash
# Published copy of queue.sh as run: absolute paths replaced by $WORKSPACE (the research workspace's dataset/ folder) and $ENGAGE3D_H4D (the Harmony4D benchmark tree).
# Overnight 2026-10-24/25: more Harmony4D cameras for the learned placement model. GPU stages strictly one at a time.
# Each stage: unpack frames (CPU) -> SAM 3D Body with pose tokens -> ViTPose+ -> MoGe-2 maps. Markers in night/*.done.
cd $WORKSPACE/scripts
export PYTORCH_ALLOC_CONF=expandable_segments:True
B=../benchmarks/engage_h4d; N=$B/night
DEV="025_grappling2 028_grappling2 030_grappling2 032_grappling2 033_grappling2 034_grappling2 035_grappling2 036_grappling2 038_grappling2 039_grappling2 040_grappling2 041_grappling2"
TEST="031_grappling2 037_grappling2 042_grappling2 043_grappling2"
log() { echo "$(date '+%F %T') $*" >> $N/NIGHT.log; }
disk_ok() { a=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); [ "$a" -ge 35 ] || { log "STOP: only ${a}G free"; touch $N/STOPPED_DISK; exit 1; }; }
stage() {   # name cams step s3d_out maps_out seqs...
  name=$1 cams=$2 step=$3 s3d=$4 maps=$5; shift 5
  [ -f $N/$name.done ] && return
  disk_ok; log "$name: extract $cams"
  ../.venv-depth/bin/python engage_h4d_extract.py --cams $cams --step $step "$@" >> $N/$name.log 2>&1
  disk_ok; log "$name: SAM 3D Body"
  systemd-run --user --scope -q -p MemoryMax=7G ../.venv-s3d/bin/python engage_bench_s3d.py --out $s3d --step $step --cams $cams --save-tokens "$@" >> $N/$name.log 2>&1
  log "$name: ViTPose"
  systemd-run --user --scope -q -p MemoryMax=6G ../.venv-pose/bin/python engage_pose2d_h4d.py $s3d >> $N/$name.log 2>&1
  disk_ok; log "$name: MoGe-2 maps"
  systemd-run --user --scope -q -p MemoryMax=6G ../.venv-depth/bin/python engage_depth_probe.py --ref $s3d --out $maps --model moge --every 1 --save-maps --maps-only "$@" >> $N/$name.log 2>&1
  log "$name: done"; touch $N/$name.done
}
log "queue start"
stage wave1 cam03,cam08,cam13,cam18 2 $B/s3d_box_xcam $B/depth_maps_xcam $DEV
stage xtest cam03,cam08,cam13,cam18 2 $B/s3d_box_xtest $B/depth_maps_xtest $TEST
if [ ! -f $N/basetok.done ]; then
  log "basetok: SAM 3D Body tokens for cam01/06/11/16"
  systemd-run --user --scope -q -p MemoryMax=7G ../.venv-s3d/bin/python engage_bench_s3d.py --out $B/s3d_box_tok --step 2 --cams cam01,cam06,cam11,cam16 --save-tokens $DEV $TEST 016_mma4 >> $N/basetok.log 2>&1
  log "basetok: done"; touch $N/basetok.done
fi
stage wave2 cam02,cam07,cam12,cam17,cam04,cam09,cam14,cam19 2 $B/s3d_box_xcam $B/depth_maps_xcam $DEV
stage wave3 cam05,cam10,cam15,cam20 2 $B/s3d_box_xcam $B/depth_maps_xcam $DEV
stage mma cam02,cam03,cam04,cam05,cam07,cam08,cam09,cam10,cam12,cam13,cam14,cam15,cam17,cam18,cam19,cam20 4 $B/s3d_box_xcam $B/depth_maps_xcam 016_mma4
log "queue finished"; touch $N/queue.done
