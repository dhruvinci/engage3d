#!/bin/bash
# Published copy of queue.sh as run: absolute paths replaced by $WORKSPACE (the research workspace's dataset/ folder) and $ENGAGE3D_H4D (the Harmony4D benchmark tree).
# Diversity test data (diversity_prereg.md): after the fetches, SAM 3D Body -> ViTPose+ -> MoGe-2 -> features, one GPU job at a time.
cd $WORKSPACE/scripts
export PYTORCH_ALLOC_CONF=expandable_segments:True
B=../benchmarks/engage_h4d; T=$ENGAGE3D_H4D
until [ -f $B/div/fetch.done ]; do sleep 30; done
TESTSEQS="025_grappling2 028_grappling2 030_grappling2 031_grappling2 032_grappling2 033_grappling2 034_grappling2 035_grappling2 036_grappling2 037_grappling2 038_grappling2 039_grappling2 040_grappling2 041_grappling2 042_grappling2 043_grappling2"
SEQS=$( (ls $T/03_grappling2; ls $T/02_grappling) | grep -v -x -F -f <(echo $TESTSEQS | tr ' ' '\n') | tr '\n' ' ')
echo "$(date '+%F %T') sequences: $(echo $SEQS | wc -w)" >> $B/div/DIV.log
echo "$(date '+%F %T') SAM 3D Body" >> $B/div/DIV.log
systemd-run --user --scope -q -p MemoryMax=7G ../.venv-s3d/bin/python engage_bench_s3d.py --out $B/s3d_box_div --step 1 --cams cam01,cam06,cam11,cam16 --save-tokens $SEQS > $B/div/s3d.log 2>&1
echo "$(date '+%F %T') ViTPose" >> $B/div/DIV.log
systemd-run --user --scope -q -p MemoryMax=6G ../.venv-pose/bin/python engage_pose2d_h4d.py $B/s3d_box_div > $B/div/vitpose.log 2>&1
echo "$(date '+%F %T') MoGe-2" >> $B/div/DIV.log
systemd-run --user --scope -q -p MemoryMax=6G ../.venv-depth/bin/python engage_depth_probe.py --ref $B/s3d_box_div --out $B/depth_maps_div --model moge --every 1 --save-maps --maps-only > $B/div/moge.log 2>&1
echo "$(date '+%F %T') features" >> $B/div/DIV.log
systemd-run --user --scope -q -p MemoryMax=7G ../.venv-depth/bin/python engage_place_features.py --maps $B/depth_maps_div/moge/maps --ref $B/s3d_box_div --out $B/place_feats2 --workers 6 --overlap > $B/div/feats.log 2>&1
echo "$(date '+%F %T') done" >> $B/div/DIV.log; touch $B/div/data.done
