#!/bin/bash
# Published copy of train.sh as run: absolute paths replaced by $WORKSPACE (the research workspace's dataset/ folder) and $ENGAGE3D_H4D (the Harmony4D benchmark tree).
# diversity_prereg.md: models A, A+same, A+new (v2 settings), evaluated on MMA + the new pair's held-out sequences.
cd $WORKSPACE/scripts
B=../benchmarks/engage_h4d; source $B/div/sets.sh
REF=$B/s3d_box,$B/s3d_box_xcam,$B/s3d_box_div
run() {  # name exclude
  TORCH_THREADS=12 systemd-run --user --scope -q -p MemoryMax=7G ../.venv-depth/bin/python engage_place_train.py --feats $B/place_feats2 --drop-overlap \
    --ref $REF --out $B/div/$1 --epochs 60 --seeds 3 --width 512 --test --test-seqs 016_mma4,$NEWTEST --exclude-seqs $2 > $B/div/$1.log 2>&1
  for w in model blend; do WORKERS=8 ../.venv-pose/bin/python engage_bench_oracle.py $B/div/$1/$w --out $B/div/$1/score_$w > $B/div/$1/score_$w.log 2>&1; done
}
run A $TEST4,$SAME,$NEWTRAIN
run A_same $TEST4,$NEWTRAIN
run A_new $TEST4,$SAME
touch $B/div/train.done
