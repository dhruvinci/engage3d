"""Copy the publishable part of the research workspace into this repo, refusing on any privacy hit.

The workspace (`~/Code/kakashi-research/dataset`) is not a git repository and holds private material, so this repo is
a curated export. Only the allowlist below is copied. Every copied file, and every text file already in the repo, is
then scanned:
  - hard hits (refuse to sync): emails, access tokens, absolute home paths, private data markers;
  - soft hits (printed for review): names of products, third parties and footage that need a human look.
Names that must never appear are not listed in this public file: they are read from a private term list (one regex per
line, prefixed "hard " or "soft "), `~/.config/engage3d/private_terms.txt` or `$ENGAGE3D_PRIVATE_TERMS`.
Nothing is committed here; review `git diff`, then commit.

  python3 tools/sync_from_workspace.py [--check-only]
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WS = Path.home()/'Code/kakashi-research/dataset'
CODE = ['engage_*.py', 'engage_paths.py', 'tangle_h4d.py', 'autolabel_common.py', 'autolabel_body3d.py',
        'autolabel_pose2d.py']
DOCS = {'docs/engagement/MEASUREMENT_SPEC.v0.md': 'docs/MEASUREMENT_SPEC.v0.md',
        'docs/engagement/MEASUREMENT_SPEC.v1.md': 'docs/MEASUREMENT_SPEC.v1.md',
        'docs/engagement/CAPTURE_DESIGN.v0.md': 'docs/CAPTURE_DESIGN.v0.md',
        'docs/RESEARCH_QUEUE.md': 'docs/RESEARCH_QUEUE.md'}
R = 'docs/engagement/results/'
EXPERIMENTS = {   # repo folder: {file in the folder: workspace source}
    'E00-benchmark-baseline': {'RESULTS-harmony4d.md': R + 'h4d_s3d_results.md', 'PREREG-vicos.md': R + 'vicos_prereg.md',
                               'RESULTS-vicos.md': R + 'vicos_results.md'},
    'E01-joint-rigid-fit': {'PREREG.md': R + 'jointfit_prereg.md', 'RESULTS.md': R + 'jointfit_results.md'},
    'E02-prompthmr': {'PREREG.md': R + 'phmr_prereg.md', 'RESULTS.md': R + 'phmr_results.md'},
    'E03-depth-map': {'PREREG.md': R + 'depthmap_prereg.md', 'RESULTS.md': R + 'depthmap_results.md'},
    'E04-combined-depth-fresh-test': {'PREREG.md': R + 'depthblend_prereg.md'},
    'E05-vicos-combined-depth': {'PREREG.md': R + 'vicos_depth_prereg.md', 'RESULTS.md': R + 'vicos_depth_results.md'},
    'E06-placement-model-v1': {'PREREG.md': R + 'placemodel_prereg.md', 'RESULTS.md': R + 'placemodel_results.md'},
    'E07-placement-model-v2-cameras': {'PREREG.md': R + 'placemodel_v2_prereg.md', 'RESULTS.md': R + 'placemodel_v2_results.md'},
    'E08-new-people': {'PREREG.md': R + 'newpeople_prereg.md', 'RESULTS.md': R + 'newpeople_results.md'},
    'E09-athlete-variety': {'PREREG.md': R + 'diversity_prereg.md', 'RESULTS.md': R + 'diversity_results.md'},
    'E10-limb-head': {'PREREG.md': R + 'limbhead_prereg.md', 'RESULTS.md': R + 'limbhead_results.md'},
    'E11-decoder-finetune': {'PREREG.md': R + 's3dft_prereg.md', 'RESULTS.md': R + 's3dft_results.md'},
    'E12-vicos-synced-phones': {'PREREG.md': R + 'vicos_sync_prereg.md', 'RESULTS.md': R + 'vicos_sync_results.md'},
    'E18-oracle-clicks': {'PREREG.md': R + 'clicks_prereg.md', 'RESULTS.md': R + 'clicks_results.md'},
    'E19-contact-2d-keypoints': {'PREREG.md': R + 'vitposeft_prereg.md', 'RESULTS.md': R + 'vitposeft_results.md'},
    'E20-calmer-contact': {'PREREG.md': R + 'easy_prereg.md', 'RESULTS.md': R + 'easy_results.md'},
    'E21-engagement-at-range': {'PREREG.md': R + 'sword_prereg.md', 'RESULTS.md': R + 'sword_results.md'},
    'E17-confidence': {'PREREG.md': R + 'confidence_prereg.md', 'RESULTS.md': R + 'confidence_results.md'},
    'E14-temporal-limbs': {'PREREG.md': R + 'temporal_prereg.md', 'RESULTS.md': R + 'temporal_results.md'},
    'E16-end-to-end-rfdetr': {'PREREG.md': R + 'rfdetr_prereg.md', 'RESULTS.md': R + 'rfdetr_results.md', 'STAGE0.md': R + 'ceilings_results.md'},
    'E22-layered-engagement': {'RESULTS.md': R + 'layered_results.md'},
    'E17b-trust-flag-deeper': {'PREREG.md': R + 'trustb_prereg.md', 'RESULTS.md': R + 'trustb_results.md'},
    'E23-engagement-measures': {'PREREG.md': R + 'measures_prereg.md'},
}
EXTRA = {'benchmarks/engage_h4d/demo_layers/run_queue.sh': 'src/run_queue.sh',   # workspace path: repo path
         'benchmarks/engage_h4d/demo_layers/run_clip.sh': 'src/run_clip.sh'}
H, V = 'benchmarks/engage_h4d/', 'benchmarks/engage_vicos/'
DATA = {   # repo folder: {file in its data/: workspace source}. Aggregate scores only: no per-item ViCoS or MAMMA data
    'E00-benchmark-baseline': {'harmony4d_summary.json': H + 's3d_box/score/summary.json',
                               'vicos_summary.json': V + 's3d/score/summary.json', 'vicos_summary_v2.json': V + 's3d/score_v2/summary.json'},
    'E01-joint-rigid-fit': {f'harmony4d_{v}_summary.json': H + f'{v}/score/summary.json' for v in ('fit_full', 'fit_no_pen', 'fit_no_2d')}
    | {f'vicos_{v}_summary.json': V + f'{v}/score/summary.json' for v in ('fit_full', 'fit_no_pen', 'fit_no_2d')}
    | {'harmony4d_multiview_summary.json': H + 's3d_box_multiview/summary.json', 'harmony4d_oracle_summary.json': H + 's3d_box_oracle/summary.json',
       'vicos_s3d_clicks_summary.json': V + 's3d_clicks/score/summary.json', 'vicos_clicks_fit_full.score.log': V + 'clicks_fit_full.score.log'},
    'E02-prompthmr': {'harmony4d_phmr_summary.json': H + 'phmr_box/score/summary.json', 'harmony4d_phmr_oracle_summary.json': H + 'phmr_box/oracle/summary.json',
                      'harmony4d_hybrid_s3dpose_phmrplace_summary.json': H + 'hybrid_s3dpose_phmrplace/score/summary.json',
                      'harmony4d_hybrid_centroid_summary.json': H + 'hybrid_centroid/score/summary.json',
                      'vicos_phmr_summary.json': V + 'phmr/score/summary.json', 'vicos_phmr.score.log': V + 'phmr.score.log'},
    'E03-depth-map': {f'{m}_{v}.score.log': H + f'depth_probe/{m}/{v}.score.log' for m in ('moge', 'depthpro')
                      for v in ('pred', 'gt_rel', 'gt_blend', 'pred_rel', 'pred_abs', 'pred_blend')}
    | {f'{m}_blend_weights.json': H + f'depth_probe/{m}/blend_weights.json' for m in ('moge', 'depthpro')},
    'E04-combined-depth-fresh-test': {f'moge_{v}.score.log': H + f'depth_probe_o4/moge/{v}.score.log' for v in ('pred', 'pred_rel', 'pred_blend', 'gt_blend')},
    'E05-vicos-combined-depth': {f'vicos_{v}.score.log': V + f'{v}.score.log' for v in
                                 ('depth_s3d.blend', 'depth_s3d.rel', 'depth_s3d_clicks.blend', 'depth_s3d_clicks.rel', 's3d_clicks')},
    'E06-placement-model-v1': {'cv_summary.json': H + 'place_cv/summary.json', 'cv_model.score.log': H + 'place_cv/model.score.log',
                               'cv_blend.score.log': H + 'place_cv/blend.score.log', 'test_summary.json': H + 'place_test/summary.json',
                               'test_model.score.log': H + 'place_test/model.score.log', 'test_blend.score.log': H + 'place_test/blend.score.log',
                               'test_s3d.score.log': H + 'place_test/s3d.score.log', 'test_s3d_summary.json': H + 'place_test/s3d_score/summary.json'},
    'E07-placement-model-v2-cameras': {'queue.sh': H + 'night/queue.published.sh'},
    'E08-new-people': {'summary.json': H + 'place_newpeople/summary.json', 'model_summary.json': H + 'place_newpeople/score_model/summary.json',
                       'blend_summary.json': H + 'place_newpeople/score_blend/summary.json'},
    'E09-athlete-variety': {f'{a}_summary.json': H + f'div/{a}/summary.json' for a in ('A', 'A_same', 'A_new')}
    | {f'{q}.sh': H + f'div/{q}.published.sh' for q in ('queue', 'train', 'sets')},
    'E10-limb-head': {'cv.json': H + 'limb_head/cv.json', 'test.json': H + 'limb_head/test.json'},
    'E11-decoder-finetune': {'limb_head_test.json': H + 'limb_head_e11/test.json', 'eval.json': H + 's3d_ft/eval.json',
                             'train_log.json': H + 's3d_ft/train_log.json'},
    'E12-vicos-synced-phones': {'agree.json': V + 'sync/agree.json', 'place_summary.json': V + 'sync/score_place/summary.json',
                                'clicks_summary.json': V + 'sync/score_clicks/summary.json'},
    'E14-temporal-limbs': {'temporal.json': H + 'temporal/temporal.json'},
    'E17-confidence': {'confidence.json': H + 'confidence/confidence.json'},
    'E18-oracle-clicks': {'clicks.json': H + 'clicks/clicks.json'},
    'E19-contact-2d-keypoints': {'train_log.json': H + 'vitpose_ft/train_log.json', 'predict.json': H + 'vitpose_ft/predict.json',
                                 'clicks_heldout.json': H + 'vitpose_ft/clicks_heldout/clicks.json'},
    'E20-calmer-contact': {f'{v}_summary.json': H + f'easy/{v}/summary.json' for v in
                           ('place', 'place_120ep', 'score_s3d', 'score_place_model', 'score_place_blend', 'score_place_model_120ep', 'score_place_blend_120ep')},
    'E21-engagement-at-range': {f'{v}_summary.json': H + f'sword/{v}/summary.json' for v in ('place', 'score_s3d', 'score_place_model', 'score_place_blend')},
    'E17b-trust-flag-deeper': {'trust_b.json': H + 'trust_b/trust_b.json', 'run.log': H + 'trust_b/run.log'},
    'E16-end-to-end-rfdetr': {'stage0_prompts.json': H + 'rfdetr/ceiling/ceiling.json', 'stage0_fit_dev.json': H + 'rfdetr/fit_ceiling/fit_ceiling.json',
                              'stage0_fit_noise.json': H + 'rfdetr/fit_ceiling/fit_noise.json', 'stage0_fit_mma.json': H + 'fit_ceiling_mma/fit_ceiling.json',
                              'stage0_fit_hugging_ballroom.json': H + 'fit_ceiling_l1/fit_ceiling.json',
                              'detect.json': H + 'rfdetr/eval/detect.json', 's3d.json': H + 'rfdetr/eval/s3d.json',
                              'train_keypoints_metrics.csv': H + 'rfdetr/kp/metrics.csv', 'train_segmentation_metrics.csv': H + 'rfdetr/seg/metrics.csv'},
}
HARD = [r'[\w.+-]+@[\w-]+\.(com|org|net|ai|io|edu)\b', r'\bhf_[A-Za-z0-9]{20,}', r'\bgh[opsu]_[A-Za-z0-9]{20,}',
        r'\bsk-[A-Za-z0-9]{20,}', r'/home/[a-z]', r'supabase', r'user_id', r'customer']
SOFT = [r'kakashi', r'youtube']
TERMS = Path(os.environ.get('ENGAGE3D_PRIVATE_TERMS', Path.home()/'.config/engage3d/private_terms.txt'))
TEXT = {'.md', '.py', '.txt', '.cff', '.json', '.yaml', '.yml', '.cfg', '.toml', '.sh', ''}


def load_terms():
    if not TERMS.exists():
        sys.exit(f'missing private term list {TERMS}: refusing to scan without it')
    for line in TERMS.read_text().splitlines():
        kind, _, pat = line.strip().partition(' ')
        if kind in ('hard', 'soft') and pat.strip():
            (HARD if kind == 'hard' else SOFT).append(pat.strip())


def scan(path, label, hard, soft):
    for i, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
        for pat in HARD:
            if re.search(pat, line, re.I):
                hard.append(f'{label}:{i}: {line.strip()[:120]}')
        for pat in SOFT:
            if re.search(pat, line, re.I):
                soft.append(f'{label}:{i}: {line.strip()[:120]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true')
    a = ap.parse_args()
    load_terms()
    plan = [(p, REPO/'src'/p.name) for pat in CODE for p in sorted((WS/'scripts').glob(pat))]
    plan += [(WS/s, REPO/d) for s, d in DOCS.items()]
    plan += [(WS/s, REPO/d) for s, d in EXTRA.items()]
    plan += [(WS/s, REPO/'experiments'/e/f) for e, files in EXPERIMENTS.items() for f, s in files.items()]
    plan += [(WS/s, REPO/'experiments'/e/'data'/f) for e, files in DATA.items() for f, s in files.items() if (WS/s).exists()]
    missing = [s for files in DATA.values() for s in files.values() if not (WS/s).exists()]
    for m in missing:
        print('missing (skipped):', m)
    hard, soft = [], []
    for src, _ in plan:
        scan(src, src.relative_to(WS), hard, soft)
    targets = {dst for _, dst in plan} | {Path(__file__).resolve()}
    for p in sorted(REPO.rglob('*')):   # hand-written files (READMEs, findings, citation) are scanned too
        if p.is_file() and '.git' not in p.parts and p.suffix.lower() in TEXT and p not in targets:
            scan(p, p.relative_to(REPO), hard, soft)
    for s in soft:
        print('review:', s)
    if hard:
        for h in hard:
            print('BLOCKED:', h)
        sys.exit(f'{len(hard)} privacy hits: nothing copied')
    print(f'{len(plan)} files clean ({len(soft)} lines flagged for review)')
    if a.check_only:
        return
    for src, dst in plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print('copied; review `git status` / `git diff`, then commit')


if __name__ == '__main__':
    main()
