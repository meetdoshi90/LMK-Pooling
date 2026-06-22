#!/usr/bin/env python3
"""
aggregate_ruler_results.py

Reads ruler_results/ (from eval_ruler_full_doc.py) and produces:

  1. Per-(task, pool) recall@10 heatmaps — best checkpoint + best LMK gran
  2. Per-task 3-panel comparison heatmaps (mean | last | lmk) side by side
  3. Summary CSV table: rows=pool, cols=task×context_length
     value = avg recall@10 across depths (best ckpt, best gran for LMK)

LMK granularity selection:
  For each (task, context_len, checkpoint): choose the granularity that
  maximises avg recall@10 across ALL depths, then pick the best checkpoint.

Usage:
  python aggregate_ruler_results.py --results_dir ruler_results --out_dir ruler_agg
"""

import csv, re, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print('[WARN] matplotlib not available — skipping plots')

ap = argparse.ArgumentParser()
ap.add_argument('--results_dir', default='ruler_results')
ap.add_argument('--out_dir',     default='ruler_agg')
ap.add_argument('--metric',      default='recall@10',
                choices=['recall@1', 'recall@5', 'recall@10', 'mrr@10'])
a = ap.parse_args()

RESULTS_DIR = Path(a.results_dir)
OUT_DIR     = Path(a.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRIC = a.metric

# ── 1. Parse all *_summary.csv files ──────────────────────────────────────────
#
# Filename format produced by eval_ruler_full_doc.py:
#   {pool_tag}_corp{N}_{model_tag}_summary.csv
#   pool_tag : lmk_gran{N} | mean | last
#
FILE_RE = re.compile(r'^(lmk_gran(\d+)|mean|last)_corp\d+_(.+?)_summary\.csv$')

# raw[(pool, gran, model, task, context_len, depth)] = metric_value
raw: dict = {}

for p in sorted(RESULTS_DIR.glob('*_summary.csv')):
    m = FILE_RE.match(p.name)
    if not m:
        print(f'[WARN] cannot parse: {p.name}')
        continue
    pool_tag, gran_s, model = m.group(1), m.group(2), m.group(3)
    # print(pool_tag, gran_s, model)
    print(p)
    pool = 'lmk' if 'lmk' in pool_tag else pool_tag
    gran = int(gran_s) if gran_s else None
    with p.open() as f:
        for row in csv.DictReader(f):
            if METRIC not in row:
                print(f'[WARN] column "{METRIC}" missing in {p.name}')
                break
            try:
                v = float(row[METRIC])
            except ValueError:
                continue
            # Store context_len as raw token count (handle both old / new CSV format)
            lk_raw = int(row['length_k'])
            # If stored as tokens//1024, reconstruct best guess
            # (new format stores raw tokens; old format stores tokens//1024)
            # context_len = lk_raw if lk_raw > 100 else lk_raw * 1024
            context_len = lk_raw if lk_raw >= 512 else lk_raw * 1024
            depth = round(float(row['depth'].rstrip('%')) / 100.0, 4)
            if depth==1.0 or model.endswith('checkpoint-1500') or model.endswith('checkpoint-2000'):
                # print('Skipping', pool, gran, model, row['task'], context_len, depth)
                continue
            raw[(pool, gran, model, row['task'], context_len, depth)] = v

if not raw:
    print(f'[ERROR] No data found in {RESULTS_DIR}')
    exit(1)

tasks        = sorted({k[3] for k in raw})
context_lens = sorted({k[4] for k in raw})
depths       = sorted({k[5] for k in raw})
pools        = sorted({k[0] for k in raw})
models       = sorted({k[2] for k in raw})

def ctx_label(c): return f"{c//1024}k" if c >= 1024 else ("512" if c == 0 else str(c))

print(f'\nParsed {len(raw):,} records')
print(f'  tasks        : {tasks}')
print(f'  context_lens : {[ctx_label(c) for c in context_lens]}')
print(f'  depths       : {[f"{d:.0%}" for d in depths]}')
print(f'  pools        : {pools}')
print(f'  checkpoints  : {len(models)}')


# ── 2. Aggregate: best (gran, checkpoint) per (pool, task, context_len) ────────
#
# Step A: average across depths
#   avg_d[(pool, gran, model, task, cl)] = mean over depths
avg_d: dict = defaultdict(list)
for (pool, gran, model, task, cl, depth), v in raw.items():
    avg_d[(pool, gran, model, task, cl)].append(v)
avg_d = {k: float(np.mean(vs)) for k, vs in avg_d.items()}

# Step B: best gran per (pool, model, task, cl)
#   For mean/last gran=None is the only option so the "best" is trivially itself
best_gran: dict = {}   # (pool, model, task, cl) → (gran, avg)
for (pool, gran, model, task, cl), avg in avg_d.items():
    key = (pool, model, task, cl)
    if key not in best_gran or avg > best_gran[key][1]:
        best_gran[key] = (gran, avg)

# Step C: best checkpoint per (pool, task, cl)
#   best[(pool, task, cl)] = (model, gran, avg_across_depths)
best: dict = {}
for (pool, model, task, cl), (gran, avg) in best_gran.items():
    key = (pool, task, cl)
    if key not in best or avg > best[key][2]:
        best[key] = (model, gran, avg)


# ── 3. Helper: build depth×context_len grid for best (model, gran) ─────────────
def make_grid(pool: str, task: str):
    """
    Returns (grid, depths_rev) where
      grid[depth_idx, length_idx] = metric value
    using the best checkpoint (and best gran for LMK) per context_len column.
    """
    depths_rev = list(reversed(depths))
    G = np.full((len(depths_rev), len(context_lens)), np.nan)
    for j, cl in enumerate(context_lens):
        bk = best.get((pool, task, cl))
        if bk is None:
            continue
        bm, bg, _ = bk
        for i, d in enumerate(depths_rev):
            v = raw.get((pool, bg, bm, task, cl, d))
            if v is not None:
                G[i, j] = v
    return G, depths_rev

'''
# ── 4. Plotting ────────────────────────────────────────────────────────────────
def _draw_heatmap(ax, G, depths_rev, title, show_ylabel=True, show_xlabel=True):
    im = ax.imshow(G, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03)

    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            if not np.isnan(G[i, j]):
                c = 'black' if 0.2 < G[i, j] < 0.8 else 'white'
                ax.text(j, i, f'{G[i,j]:.2f}', ha='center', va='center',
                        fontsize=6, color=c, fontweight='bold')

    ax.set_xticks(range(len(context_lens)))
    ax.set_xticklabels([ctx_label(c) for c in context_lens], fontsize=7)
    ax.set_yticks(range(len(depths_rev)))
    ax.set_yticklabels([f'{d:.0%}' for d in depths_rev] if show_ylabel
                        else ['']*len(depths_rev), fontsize=7)
    if show_xlabel:
        ax.set_xlabel('Context Length', fontsize=8)
    if show_ylabel:
        ax.set_ylabel('Needle Depth  (0%=start, 100%=end)', fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')


# def _pool_label(pool: str, task: str) -> str:
#     if pool != 'lmk':
#         return pool.upper()
#     grans = sorted({best.get((pool, task, cl), (None,None,None))[1]
#                     for cl in context_lens} - {None})
#     if len(grans) == 1:
#         return f'LMK  (gran={grans[0]})'
#     gran_str = '+'.join(map(str, grans))
#     return f'LMK  (gran={gran_str}, best/col)'

def _pool_label(pool: str, task: str) -> str:
    if pool != 'lmk':
        return pool.upper()
    # Pull from raw so ALL evaluated grans are shown, not just the winner
    grans = sorted({k[1] for k in raw if k[0] == pool and k[3] == task and k[1] is not None})
    if len(grans) == 1:
        return f'LMK  (gran={grans[0]})'
    gran_str = ','.join(map(str, grans))
    return f'LMK  (best of gran∈{{{gran_str}}})'

if HAS_MPL:
    ORDERED_POOLS = [p for p in ['mean', 'last', 'lmk'] if p in pools]

    # ── Plot A: individual heatmaps per (task, pool) ──────────────────────────
    for task in tasks:
        for pool in ORDERED_POOLS:
            G, depths_rev = make_grid(pool, task)
            if np.all(np.isnan(G)):
                continue
            fig, ax = plt.subplots(figsize=(max(5, len(context_lens)*1.4),
                                            2.5 + len(depths)*0.55))
            _draw_heatmap(ax, G, depths_rev, _pool_label(pool, task))
            fig.suptitle(f'{task}  |  {METRIC}  (best ckpt)',
                         fontsize=10, fontweight='bold')
            plt.tight_layout()
            fname = f'{task}_{pool}_{METRIC.replace("@","_at_")}.png'
            plt.savefig(OUT_DIR / fname, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'[PLOT] {fname}')

    # ── Plot B: side-by-side comparison (mean | last | lmk) per task ──────────
    for task in tasks:
        n = len(ORDERED_POOLS)
        if n == 0:
            continue

        w_per = max(4.5, len(context_lens) * 1.2)
        h = 2.5 + len(depths) * 0.55
        fig, axes = plt.subplots(1, n, figsize=(w_per * n, h), sharey=True)
        if n == 1:
            axes = [axes]

        for idx, (ax, pool) in enumerate(zip(axes, ORDERED_POOLS)):
            G, depths_rev = make_grid(pool, task)
            _draw_heatmap(
                ax, G, depths_rev,
                title=_pool_label(pool, task),
                show_ylabel=(idx == 0),
                show_xlabel=True,
            )

        fig.suptitle(
            f'{task}  |  {METRIC}  —  Mean  vs  Last  vs  LMK  (best ckpt per pool)',
            fontsize=11, fontweight='bold', y=1.02,
        )
        plt.tight_layout()
        fname = f'{task}_comparison_{METRIC.replace("@","_at_")}.png'
        plt.savefig(OUT_DIR / fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[PLOT] {fname}')

    print(f'\n[INFO] All plots saved to {OUT_DIR}/')
'''
# ── 4. Plotting ────────────────────────────────────────────────────────────────
#
# Paper-ready settings for EMNLP (two-column, ~3.3in per column, 300 dpi)

import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt

# ── Step 1: apply classic style FIRST so our overrides are not wiped ──────────
# plt.style.use must come before rcParams.update — classic resets everything.
plt.style.use('classic')

# ── Step 2: register Palatino TTFs ────────────────────────────────────────────
_FONT_DIR = '/u/meet/.local/share/fonts'
_PALATINO_FONTS = [
    f'{_FONT_DIR}/palr45w.ttf',
    f'{_FONT_DIR}/fonnts.com-Palatino-LT-Bold.ttf',
    f'{_FONT_DIR}/fonnts.com-Palatino-LT-Roman.ttf',
]
for _fp in _PALATINO_FONTS:
    try:
        font_manager.fontManager.addfont(_fp)
    except Exception as e:
        print(f'[WARN] Could not load font {_fp}: {e}')

_available  = {f.name for f in font_manager.fontManager.ttflist}
_FONT_FAMILY = 'Palatino LT' if 'Palatino LT' in _available else 'Palatino'
if _FONT_FAMILY not in _available:
    _FONT_FAMILY = next(
        (f for f in ['Georgia', 'DejaVu Serif'] if f in _available), 'serif'
    )
    print(f'[WARN] Palatino LT not found — falling back to {_FONT_FAMILY!r}')
else:
    print(f'[INFO] Using font: {_FONT_FAMILY}')

# ── Step 3: override rcParams AFTER style is applied ─────────────────────────
matplotlib.rcParams.update({
    'font.family':       'serif',
    'font.serif':        [_FONT_FAMILY, 'Palatino', 'Georgia',
                          'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset':  'dejavusans',
    'text.usetex':       False,

    'font.size':         11,   # was 14
    'axes.titlesize':    12,   # was 15
    'axes.labelsize':    11,   # was 14
    'xtick.labelsize':   10,   # was 12
    'ytick.labelsize':   10,   # was 12
    'legend.fontsize':   10,   # was 12

    'axes.linewidth':    1.0,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size':  3.0,
    'ytick.major.size':  3.0,
    'patch.edgecolor':   'black',

    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.03,
})

# ── Layout constants ───────────────────────────────────────────────────────────
_COL_W = 3.35
_TW    = 7.0

# ── Tick label helpers ─────────────────────────────────────────────────────────

def _depth_label(d: float) -> str:
    return f'{round(d * 100)}%'

def _ctx_label_paper(c: int) -> str:
    k = c // 1024
    return f'{k}K' if k > 0 else 512 if c==0 else str(c)


# ── Core heatmap drawing ───────────────────────────────────────────────────────

def _draw_heatmap(ax, G, depths_rev, title,
                  show_ylabel=True, show_xlabel=True, cbar=True):
    im = ax.imshow(G, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto',
                   interpolation='nearest')

    if cbar:
        cb = plt.colorbar(im, ax=ax, shrink=0.90, pad=0.02,
                          aspect=20, fraction=0.046)
        cb.set_label(METRIC, fontsize=10, labelpad=4)
        cb.ax.tick_params(labelsize=9, length=2.5, width=0.7)
        cb.outline.set_linewidth(0.6)

    # Cell value annotations
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            if not np.isnan(G[i, j]):
                val   = G[i, j]
                txt_c = 'white' if (val < 0.25 or val > 0.82) else 'black'
                ax.text(j, i, f'{val:.2f}',
                        ha='center', va='center',
                        fontsize=7, color=txt_c, fontweight='bold')

    # Cell separator grid lines
    n_rows, n_cols = G.shape
    for x in np.arange(-0.5, n_cols, 1):
        ax.axvline(x, color='white', linewidth=0.5, zorder=2)
    for y in np.arange(-0.5, n_rows, 1):
        ax.axhline(y, color='white', linewidth=0.5, zorder=2)

    # X axis
    ax.set_xticks(range(len(context_lens)))
    ax.set_xticklabels([_ctx_label_paper(c) for c in context_lens],
                       fontsize=10, rotation=0)
    if show_xlabel:
        ax.set_xlabel('Context Length', fontsize=11, labelpad=4)

    # Y axis — ALWAYS set ticks so sharey panels aren't broken;
    # control visibility explicitly via tick_params instead of empty labels
    ax.set_yticks(range(len(depths_rev)))
    ax.set_yticklabels([_depth_label(d) for d in depths_rev], fontsize=10)
    if show_ylabel:
        ax.set_ylabel('Needle Depth', fontsize=11, labelpad=4)
        ax.tick_params(axis='y', which='both', labelleft=True)
    else:
        # Hide tick labels on non-first panels WITHOUT using sharey
        ax.tick_params(axis='y', which='both', labelleft=False)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)


# ── Panel label helper ─────────────────────────────────────────────────────────

def _pool_label(pool: str, task: str) -> str:
    if pool != 'lmk':
        return pool.upper()
    grans = sorted({best.get((pool, task, cl), (None, None, None))[1]
                    for cl in context_lens} - {None})
    return 'LMK'
    # if len(grans) == 1:
    #     return f'LMK (gran={grans[0]})'
    # return f'LMK (gran={"+".join(map(str, grans))})'


if HAS_MPL:
    ORDERED_POOLS = [p for p in ['mean', 'last', 'lmk'] if p in pools]

    # ── Plot A: individual heatmaps per (task, pool) ───────────────────────────
    for task in tasks:
        for pool in ORDERED_POOLS:
            G, depths_rev = make_grid(pool, task)
            if np.all(np.isnan(G)):
                continue

            n_ctx = len(context_lens)
            n_dep = len(depths_rev)
            fig_w = min(_TW, max(_COL_W, n_ctx * 0.55 + 1.4))
            fig_h = 1.2 + n_dep * 0.38

            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            _draw_heatmap(ax, G, depths_rev,
                          title=_pool_label(pool, task),
                          show_ylabel=True, show_xlabel=True, cbar=True)
            fig.suptitle(
                f'{task.replace("_", " ")}  —  {METRIC}  (best checkpoint)',
                fontsize=14, fontweight='bold', y=1.02,
            )
            plt.tight_layout()
            fname = f'{task}_{pool}_{METRIC.replace("@", "_at_")}'
            plt.savefig(OUT_DIR / f'{fname}.pdf')
            # plt.savefig(OUT_DIR / f'{fname}.png')
            plt.close()
            print(f'[PLOT] {fname}.pdf')

    # ── Plot B: side-by-side comparison (mean | last | lmk) per task ──────────
    for task in tasks:
        n = len(ORDERED_POOLS)
        if n == 0:
            continue

        n_dep = len(depths_rev)
        fig_h = 1.2 + n_dep * 0.38

        # sharey=False — we control y-tick visibility manually in _draw_heatmap
        # so that ALL panels show the depth labels, not just the first one.
        # We then sync ylim manually below.
        fig, axes = plt.subplots(
            1, n,
            figsize=(_TW, fig_h),
            sharey=False,                      # <-- intentionally False
            gridspec_kw={'wspace': 0.25},      # a bit more space so % labels fit
        )
        if n == 1:
            axes = [axes]

        for idx, (ax, pool) in enumerate(zip(axes, ORDERED_POOLS)):
            G, depths_rev = make_grid(pool, task)
            _draw_heatmap(
                ax, G, depths_rev,
                title=_pool_label(pool, task),
                show_ylabel=(idx == 0),              # show depth % on EVERY panel
                show_xlabel=True,
                cbar=(idx == n - 1),
            )

        # Sync ylim across panels (since sharey=False)
        all_ylims = [ax.get_ylim() for ax in axes]
        ymin = min(y[0] for y in all_ylims)
        ymax = max(y[1] for y in all_ylims)
        for ax in axes:
            ax.set_ylim(ymin, ymax)

        fig.suptitle(
            f'{task.replace("_", " ")}  —  {METRIC}:'
            f'  Mean  vs.  Last-token  vs.  LMK',
            fontsize=13, fontweight='bold',
        )
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        fname = f'{task}_comparison_{METRIC.replace("@", "_at_")}'
        plt.savefig(OUT_DIR / f'{fname}.pdf')
        # plt.savefig(OUT_DIR / f'{fname}.png')
        plt.close()
        print(f'[PLOT] {fname}.pdf')

    print(f'\n[INFO] All plots saved to {OUT_DIR}/')

# ── 5. Summary CSV table ───────────────────────────────────────────────────────
#
# Rows   : pool (mean / last / lmk)
# Columns: task × context_length  (two header rows)
# Value  : avg recall@10 across depths (best ckpt + gran)
#
metric_safe = METRIC.replace('@', '_at_')
table_path  = OUT_DIR / f'summary_table_{metric_safe}.csv'

with table_path.open('w', newline='') as f:
    w = csv.writer(f)

    # Header row 1: task names (span len(context_lens) columns each)
    h1 = ['pool', 'lmk_gran']
    for task in tasks:
        h1.append(task)
        h1.extend([''] * (len(context_lens) - 1))
    w.writerow(h1)

    # Header row 2: context lengths
    h2 = ['', '']
    for _ in tasks:
        h2.extend(ctx_label(c) for c in context_lens)
    w.writerow(h2)

    # Data rows — one per pool
    for pool in ['mean', 'last', 'lmk']:
        if pool not in pools:
            continue
        grans_used = sorted({best.get((pool, t, c), (None,None,None))[1]
                             for t in tasks for c in context_lens} - {None})
        # gran_cell = '/'.join(map(str, grans_used)) if grans_used else ''
        gran_cell = '/'.join(map(str, grans_used)) if grans_used else 'N/A'
        row = [pool, gran_cell]
        for task in tasks:
            for cl in context_lens:
                bk = best.get((pool, task, cl))
                row.append(f'{bk[2]:.4f}' if bk else '')
        w.writerow(row)

print(f'[CSV]  {table_path.name}')

# Flat CSV for further analysis / debugging
flat_path = OUT_DIR / f'flat_best_{metric_safe}.csv'
with flat_path.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['pool','gran','model','task',
                                       'context_len', f'avg_depth_{METRIC}'])
    w.writeheader()
    for (pool, task, cl), (model, gran, avg) in sorted(best.items()):
        w.writerow({'pool': pool, 'gran': gran or '', 'model': model,
                    'task': task, 'context_len': ctx_label(cl),
                    f'avg_depth_{METRIC}': f'{avg:.4f}'})
print(f'[CSV]  {flat_path.name}')


# ── 6. Console summary ─────────────────────────────────────────────────────────
print(f'\n{"="*90}')
print(f'  {METRIC.upper()}  —  avg across depths | best ckpt + gran per (pool, task, ctx_len)')
print(f'{"="*90}')

for pool in ['mean', 'last', 'lmk']:
    if pool not in pools:
        continue
    print(f'\n  ── {pool.upper()} ──')
    header = f'  {"task":<24}' + ''.join(f'  {ctx_label(c):>5}' for c in context_lens) + '  avg'
    print(header)
    print('  ' + '─' * (len(header) - 2))
    for task in tasks:
        line  = f'  {task:<24}'
        vals  = []
        for cl in context_lens:
            bk = best.get((pool, task, cl))
            if bk:
                line += f'  {bk[2]:.3f}'
                vals.append(bk[2])
            else:
                line += '    -- '
        line += f'  {np.mean(vals):.3f}' if vals else '   --'
        print(line)

print('\nDone.')