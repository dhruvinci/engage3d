"""E16: fine-tune RF-DETR on Harmony4D pair crops: the keypoint model (boxes + 17 COCO keypoints) and a segmentation
model (boxes + masks) (GPU, .venv-rfdetr). Experimental. Pre-registration: `docs/engagement/results/rfdetr_prereg.md`.

Data from `engage_rfdetr_data.py` (COCO format, train/ and valid/). Weights start from Roboflow's COCO-pretrained
checkpoints, kept in ~/Code/shared/models/rfdetr. The checkpoint used at test is the one RF-DETR selects on the
validation sequences (025 / 035; development data, never the test views): `checkpoint_best_ema.pth`.

  PYTORCH_ALLOC_CONF=expandable_segments:True ../.venv-rfdetr/bin/python engage_rfdetr_train.py \
      --task keypoints|seg --data D --out O [--epochs N] [--batch B] [--accum A] [--res R]
"""
import argparse
import json
import os
import time
from pathlib import Path

WEIGHTS = Path.home()/'Code/shared/models/rfdetr'
TASKS = {
    'keypoints': ('RFDETRKeypointPreview', 'rf-detr-keypoint-preview-xlarge.pth'),
    'seg': ('RFDETRSegMedium', 'rf-detr-seg-medium.pt'),
}


def build(task, res=None, checkpointing=True, queries=None):
    import rfdetr
    cls, weights = TASKS[task]
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    kw = dict(pretrain_weights=str(WEIGHTS/weights), gradient_checkpointing=checkpointing)
    if res:
        kw['resolution'] = res
    if queries:
        kw.update(num_queries=queries, num_select=queries)
    return getattr(rfdetr, cls)(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=list(TASKS), required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--res', type=int, default=None)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--queries', type=int, default=None, help='person queries (pretrained: 100 keypoints, 200 seg)')
    ap.add_argument('--fixed-scale', action='store_true', help='train at the model resolution only (no multi-scale)')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = build(a.task, a.res, checkpointing=not os.environ.get('RFDETR_NO_CKPT'), queries=a.queries)
    t0 = time.time()
    model.train(dataset_dir=a.data, output_dir=str(out), epochs=a.epochs, batch_size=a.batch,
                grad_accum_steps=a.accum, lr=a.lr, num_workers=4, tensorboard=False, progress_bar=os.environ.get('RFDETR_BAR'),
                checkpoint_interval=max(1, a.epochs // 3),
                **(dict(multi_scale=False, expanded_scales=False) if a.fixed_scale else {}))
    json.dump({'task': a.task, 'epochs': a.epochs, 'batch': a.batch, 'accum': a.accum, 'res': a.res, 'lr': a.lr,
               'minutes': round((time.time() - t0) / 60, 1)}, open(out/'train_settings.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
