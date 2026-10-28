"""Index of the engagement-demo queue (CPU, any venv with numpy): one row per clip in demo_layers/queue.tsv with its
engagement stop, source, gate result and each layer's headline number, written as INDEX.md and a local gallery
(index.html: every demo video side by side with its numbers). Local only: MAMMA renders may not be uploaded anywhere,
and internet clips are internal research material.

  python3 engage_demo_index.py
"""
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

D = Path(__file__).resolve().parent.parent/'benchmarks/engage_h4d/demo_layers'


def load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def fmt(v, f='.2f', pct=False):
    if v is None:
        return 'n/a'
    return f'{v:.0%}' if pct else format(v, f)


def row(name, seq, note):
    o = D/name
    G, S, X = load(o/'GATE.json'), load(o/'scorecard.json'), load(o/'context.json')
    r = {'name': name, 'note': note, 'state': 'queued', 'truth': None, 'cells': ['', '', '', '', '', ''], 'video': None}
    if G is not None:
        r['state'] = 'passed' if G['passed'] else 'GATE FAILED: ' + '; '.join(k for k, v in G['gates'].items() if not v['ok'])
    elif (o/'track.json').exists() or (D/f'{name}_seed/detect.json').exists():
        r['state'] = 'running or stopped'
    if (o/'demo.mp4').exists():
        from engage_demo_render import to_h264
        import subprocess
        codec = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name',
                                '-of', 'csv=p=0', str(o/'demo.mp4')], capture_output=True, text=True).stdout.strip()
        if codec != 'h264':                                         # browsers cannot play OpenCV's mpeg4
            to_h264(o/'demo.mp4')
        r['video'] = f'{name}/demo.mp4'
    if S:
        r['truth'] = S.get('truth', True)
        L1, L2, L3, L4, L5 = (S.get(k, {}) for k in ('L1_detection_tracking', 'L2_segmentation', 'L3_skeleton', 'L4_3d', 'L5_contact'))
        if r['truth']:
            r['cells'] = [f'both found {fmt(L1.get("both_found_iou50"), pct=True)}, {L1.get("identity_switches")} switches',
                          f'mask IoU {fmt(L2.get("mask_iou_median"))}',
                          f'{fmt(L3.get("visible_joints_px"), ".0f")} px visible / {fmt(L3.get("hidden_joints_px"), ".0f")} px hidden',
                          f'limbs {fmt(L4.get("limb_joints_other_cm"), ".1f")} cm; pair dist. {fmt(L4.get("pair_distance_err_cm"), ".1f")} cm off',
                          f'touch F1 {fmt(L5.get("touching_frame_f1"))}, agree {fmt(L5.get("touching_frame_agreement"), pct=True)}', '']
        else:
            r['cells'] = [f'unmasked {fmt(L1.get("held_person_frames"), pct=True)}',
                          f'overlap {fmt(L2.get("mask_overlap_median"), pct=True)}',
                          f'kp score {fmt(L3.get("keypoint_score_median"))}',
                          f'fit {fmt(L4.get("fit_residual_share_of_person_height"), ".1%")} of height; pelvises {fmt(L4.get("pair_pelvis_distance_m_median"))} m',
                          f'touching {fmt(L5.get("touching_share"), pct=True)} of frames', '']
    if X and X.get('astra'):
        r['cells'][5] = X['astra'].get('interaction_type', '')
    return r


def main():
    rows = []
    for line in open(D/'queue.tsv'):
        if not line.strip() or line.startswith('#'):
            continue
        p = line.rstrip('\n').split('\t')
        rows.append(row(p[0], p[1], p[4] if len(p) > 4 else ''))
    done = sum(r['state'] == 'passed' for r in rows)
    md = [f'# Engagement demo queue: {done} of {len(rows)} clips passed every layer gate', '',
          '| # | clip | engagement | truth | state | L1 people | L2 masks | L3 skeletons | L4 3D | L5 contact | L6 Astra |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for i, r in enumerate(rows, 1):
        md.append(f'| {i} | {r["name"]} | {r["note"]} | {"yes" if r["truth"] else ("no" if r["truth"] is False else "")} | {r["state"]} | '
                  + ' | '.join(c.replace('|', '/') for c in r['cells']) + ' |')
    (D/'INDEX.md').write_text('\n'.join(md) + '\n')
    cards = []
    for i, r in enumerate(rows, 1):
        vid = f'<video src="{html.escape(r["video"])}" controls muted loop preload="metadata"></video>' if r['video'] else '<div class="none">no video yet</div>'
        cells = ''.join(f'<li><b>L{k + 1}</b> {html.escape(c)}</li>' for k, c in enumerate(r['cells']) if c)
        cards.append(f'<section><h2>{i}. {html.escape(r["name"])}</h2><p class="note">{html.escape(r["note"])}</p>'
                     f'<p class="state {"ok" if r["state"] == "passed" else "bad"}">{html.escape(r["state"])}</p>{vid}<ul>{cells}</ul></section>')
    page = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Engagement demo queue</title><style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--ok:#1f7a3a;--bad:#a8321f;--card:#fff;--line:#e4e4df}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ececea;--mut:#9a9a94;--ok:#5cc27a;--bad:#ef7a64;--card:#1f1f1d;--line:#33332f}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0;padding:24px 16px}}
h1{{font-size:20px;margin:0 0 16px}} main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:16px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}} h2{{font-size:16px;margin:0 0 4px}}
.note{{color:var(--mut);margin:0 0 4px}} .state{{margin:0 0 8px;font-weight:600}} .ok{{color:var(--ok)}} .bad{{color:var(--bad)}}
video{{width:100%;border-radius:4px;background:#000}} ul{{margin:8px 0 0;padding-left:18px}} .none{{color:var(--mut);padding:40px 0;text-align:center}}
@media (max-width:560px){{main{{grid-template-columns:1fr}}}}
</style></head><body><h1>Engagement demo queue: {done} of {len(rows)} clips passed every layer gate</h1><main>{''.join(cards)}</main></body></html>'''
    (D/'index.html').write_text(page)
    print(f'{done}/{len(rows)} passed; wrote {D/"INDEX.md"} and {D/"index.html"}')


if __name__ == '__main__':
    main()
