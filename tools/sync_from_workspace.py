"""Copy the publishable part of the research workspace into this repo, refusing on any privacy hit.

The workspace (`~/Code/kakashi-research/dataset`) is not a git repository and holds private material, so this repo is
a curated export. Only the allowlist below is copied. Every copied file is then scanned:
  - hard hits (refuse to sync): emails, access tokens, absolute home paths, private data markers;
  - soft hits (printed for review): names of products, third parties and footage that need a human look.
Nothing is committed here; review `git diff`, then commit.

  python3 tools/sync_from_workspace.py [--check-only]
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WS = Path.home()/'Code/kakashi-research/dataset'
CODE = ['engage_*.py', 'engage_paths.py', 'tangle_h4d.py', 'autolabel_common.py', 'autolabel_body3d.py',
        'autolabel_pose2d.py']
DOCS = {'docs/engagement/MEASUREMENT_SPEC.v0.md': 'docs/MEASUREMENT_SPEC.v0.md',
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
}
HARD = [r'[\w.+-]+@[\w-]+\.(com|org|net|ai|io|edu)\b', r'\bhf_[A-Za-z0-9]{20,}', r'\bgh[opsu]_[A-Za-z0-9]{20,}',
        r'\bsk-[A-Za-z0-9]{20,}', r'/home/[a-z]', r'supabase', r'user_id', r'customer']
SOFT = [r'kakashi', r'youtube']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true')
    a = ap.parse_args()
    plan = [(p, REPO/'src'/p.name) for pat in CODE for p in sorted((WS/'scripts').glob(pat))]
    plan += [(WS/s, REPO/d) for s, d in DOCS.items()]
    plan += [(WS/s, REPO/'experiments'/e/f) for e, files in EXPERIMENTS.items() for f, s in files.items()]
    hard, soft = [], []
    for src, _ in plan:
        text = src.read_text(errors='replace')
        for i, line in enumerate(text.splitlines(), 1):
            for pat in HARD:
                if re.search(pat, line, re.I):
                    hard.append(f'{src.relative_to(WS)}:{i}: {line.strip()[:120]}')
            for pat in SOFT:
                if re.search(pat, line, re.I):
                    soft.append(f'{src.relative_to(WS)}:{i}: {line.strip()[:120]}')
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
