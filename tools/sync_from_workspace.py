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
    'E11-decoder-finetune': {'PREREG.md': R + 's3dft_prereg.md'},
}
HARD = [r'[\w.+-]+@[\w-]+\.(com|org|net|ai|io|edu)\b', r'\bhf_[A-Za-z0-9]{20,}', r'\bgh[opsu]_[A-Za-z0-9]{20,}',
        r'\bsk-[A-Za-z0-9]{20,}', r'/home/[a-z]', r'supabase', r'user_id', r'customer']
SOFT = [r'kakashi', r'youtube']
TERMS = Path(os.environ.get('ENGAGE3D_PRIVATE_TERMS', Path.home()/'.config/engage3d/private_terms.txt'))
TEXT = {'.md', '.py', '.txt', '.cff', '.json', '.yaml', '.yml', '.cfg', '.toml', ''}


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
    plan += [(WS/s, REPO/'experiments'/e/f) for e, files in EXPERIMENTS.items() for f, s in files.items()]
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
