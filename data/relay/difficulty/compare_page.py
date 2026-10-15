#!/usr/bin/env python3
"""benchmark.json (+ judged window labels) -> compare.html, a self-contained page of method comparisons with SVG figures.

    python compare_page.py --bench <dir with benchmark.json, windows.jsonl, labels*.jsonl> --summary summary.md --out compare.html
"""
import argparse
import collections
import glob
import html
import json
import os

MEASURE_LABEL = dict(
    nll='raw NLL', gap='gap to student top-1', excess='surprisal − entropy', z='standardized log-prob (z)',
    pval='surprise p-value', rank='clipped rank', conflict='confident conflict', entropy='student entropy',
    D='turn curvature D', rsr='rank/surprisal (RSR)', sub1='share p<1%', length='turn length',
    dnr='Δ no-refeed − full', xstudent='two-student mean NLL', position='early in turn',
    sft_stays_hard='(truth) stays hard after SFT')
SETS = [('dk0921/full_history', '09-21 · full history', '#3b5bdb'), ('dk0921/norefeed', '09-21 · no-refeed', '#91a7ff'),
        ('rlvr1s12/full_history', 'RLVR1 · full history', '#e8590c'), ('rlvr1s12/norefeed', 'RLVR1 · no-refeed', '#ffa94d')]
CATS = [('K', 'missing knowledge', '#2f9e44'), ('S', 'strategy / action choice', '#37b24d'),
        ('O', 'reading an observation', '#69db7c'), ('R', 'relies on earlier reasoning', '#a9e34b'),
        ('T', 'transition / format', '#adb5bd'), ('W', 'surface wording', '#ffd43b'),
        ('A', 'arbitrary specifics', '#ff922b'), ('C', 'teacher confusion / error', '#f03e3e')]
INFORMATIVE = {'K', 'S', 'O', 'R'}


def esc(s):
    return html.escape(str(s))


def hbars(rows, groups, title, xmin=0.0, xmax=1.0, ref=None, fmt='{:.2f}', width=760):
    """rows: [(label, {group: (value, [lo, hi]) or None})]; groups: [(key, label, color)]."""
    bh, gap, left, top = 9, 7, 210, 34
    gh = bh * len(groups) + gap
    h = top + gh * len(rows) + 30
    x = lambda v: left + (width - left - 60) * (v - xmin) / (xmax - xmin)  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img" aria-label="{esc(title)}">',
           f'<text x="0" y="14" class="ct">{esc(title)}</text>']
    step = (width - left) / max(1, len(groups))
    for i, (key, lab, col) in enumerate(groups):
        out.append(f'<rect x="{left + i * step}" y="20" width="10" height="8" fill="{col}"/>'
                   f'<text x="{left + i * step + 14}" y="28" class="lg">{esc(lab)}</text>')
    for t in [xmin + (xmax - xmin) * k / 4 for k in range(5)]:
        out.append(f'<line x1="{x(t)}" x2="{x(t)}" y1="{top}" y2="{h - 26}" class="grid"/>'
                   f'<text x="{x(t)}" y="{h - 12}" class="ax" text-anchor="middle">{fmt.format(t)}</text>')
    if ref is not None:
        out.append(f'<line x1="{x(ref)}" x2="{x(ref)}" y1="{top - 4}" y2="{h - 26}" class="ref"/>')
    for r, (lab, vals) in enumerate(rows):
        y0 = top + r * gh
        out.append(f'<text x="{left - 8}" y="{y0 + gh / 2 + 2}" class="rl" text-anchor="end">{esc(lab)}</text>')
        for i, (key, _, col) in enumerate(groups):
            v = vals.get(key)
            if not v or v[0] is None:
                continue
            val, ci = v
            y = y0 + i * bh
            base = x(max(xmin, min(ref if ref is not None else xmin, xmax)))
            xv = x(max(xmin, min(val, xmax)))
            out.append(f'<rect x="{min(base, xv)}" y="{y}" width="{abs(xv - base)}" height="{bh - 2}" fill="{col}">'
                       f'<title>{esc(lab)} · {esc(key)}: {fmt.format(val)}{f" [{fmt.format(ci[0])}, {fmt.format(ci[1])}]" if ci and ci[0] is not None else ""}</title></rect>')
            if ci and ci[0] is not None:
                out.append(f'<line x1="{x(ci[0])}" x2="{x(ci[1])}" y1="{y + bh / 2 - 1}" y2="{y + bh / 2 - 1}" class="ci"/>')
    out.append('</svg>')
    return ''.join(out)


def lines(series, xs, title, width=520, height=230, ylab='mean NLL'):
    """series: [(label, color, [y...])]."""
    left, top, bot = 46, 30, 34
    ys = [v for _, _, s in series for v in s]
    ymin, ymax = 0, max(ys) * 1.1
    X = lambda i: left + (width - left - 110) * i / (len(xs) - 1)  # noqa: E731
    Y = lambda v: top + (height - top - bot) * (1 - (v - ymin) / (ymax - ymin))  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{esc(title)}">',
           f'<text x="0" y="14" class="ct">{esc(title)}</text>']
    for k in range(5):
        v = ymin + (ymax - ymin) * k / 4
        out.append(f'<line x1="{left}" x2="{width - 110}" y1="{Y(v)}" y2="{Y(v)}" class="grid"/>'
                   f'<text x="{left - 6}" y="{Y(v) + 4}" class="ax" text-anchor="end">{v:.2f}</text>')
    for i, xl in enumerate(xs):
        out.append(f'<text x="{X(i)}" y="{height - 14}" class="ax" text-anchor="middle">{esc(xl)}</text>')
    for lab, col, s in series:
        pts = ' '.join(f'{X(i)},{Y(v)}' for i, v in enumerate(s))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        for i, v in enumerate(s):
            out.append(f'<circle cx="{X(i)}" cy="{Y(v)}" r="3" fill="{col}"><title>{esc(lab)} {xs[i]}: {v:.3f}</title></circle>')
        out.append(f'<text x="{X(len(xs) - 1) + 8}" y="{Y(s[-1]) + 4}" class="lg" fill="{col}">{esc(lab)}</text>')
    out.append(f'<text x="{left}" y="{top - 6}" class="ax">{esc(ylab)}</text></svg>')
    return ''.join(out)


def stacked(rows, title, width=760):
    """rows: [(label, Counter of category codes)] -> 100 % stacked bars."""
    bh, left, top = 16, 210, 40
    h = top + (bh + 6) * len(rows) + 20
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img" aria-label="{esc(title)}">',
           f'<text x="0" y="14" class="ct">{esc(title)}</text>']
    for i, (c, lab, col) in enumerate(CATS):
        out.append(f'<rect x="{left + (i % 4) * 150}" y="{18 + (i // 4) * 11}" width="9" height="8" fill="{col}"/>'
                   f'<text x="{left + (i % 4) * 150 + 12}" y="{26 + (i // 4) * 11}" class="lg">{esc(lab)}</text>')
    W = width - left - 80
    for r, (lab, cnt) in enumerate(rows):
        y = top + r * (bh + 6)
        n = sum(cnt.values()) or 1
        out.append(f'<text x="{left - 8}" y="{y + bh - 4}" class="rl" text-anchor="end">{esc(lab)}</text>')
        x0 = left
        for c, clab, col in CATS:
            w = W * cnt.get(c, 0) / n
            if w > 0:
                out.append(f'<rect x="{x0}" y="{y}" width="{w}" height="{bh}" fill="{col}"><title>{esc(lab)}: {esc(clab)} {cnt.get(c, 0)}/{n}</title></rect>')
            x0 += w
        k_inf = sum(cnt.get(c, 0) for c in INFORMATIVE)
        out.append(f'<text x="{left + W + 6}" y="{y + bh - 4}" class="rl">{k_inf}/{n} inf.</text>')
    out.append('</svg>')
    return ''.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', required=True)
    ap.add_argument('--summary', required=True, help='markdown-ish text: the verdict, written after reading results')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    B = json.load(open(os.path.join(a.bench, 'benchmark.json')))
    wins = {}
    for line in open(os.path.join(a.bench, 'windows.jsonl')) if os.path.exists(os.path.join(a.bench, 'windows.jsonl')) else []:
        w = json.loads(line)
        wins[w['id']] = w
    labels = {}
    for f in glob.glob(os.path.join(a.bench, 'labels*.jsonl')):
        for line in open(f):
            if line.strip():
                lab = json.loads(line)
                labels[lab['id']] = lab
    parts = []
    # B1
    measures = list(B['config']['token_measures']) + list(B['config']['turn_extra'])
    rows = [(MEASURE_LABEL.get(m, m), {k: (B['b1'][k][m]['auroc'], B['b1'][k][m]['ci']) for k, _, _ in SETS if k in B['b1']})
            for m in measures]
    rows.sort(key=lambda r: -(r[1].get('rlvr1s12/full_history') or r[1].get('dk0921/full_history') or (0, 0))[0])
    parts.append(('b1', 'Benchmark 1 · Does the measure tell the teacher apart from the student itself?',
                  '<p>For each teacher turn, the student wrote 8 replies of its own to the same history (temperature 1). '
                  'AUROC = probability the teacher turn scores as more off-policy than one of the student\'s own replies '
                  '(mean over turns; whiskers: 95 % bootstrap over trajectories). 0.5 = indistinguishable; below 0.5 = the '
                  'teacher looks <i>more</i> typical than the student\'s own samples.</p>'
                  + hbars(rows, SETS, 'Turn-level AUROC: teacher turn vs the student\'s own replies at the same state',
                          xmin=0.0, xmax=1.0, ref=0.5)))
    tk = B.get('b1_tokens', {})
    if tk:
        cls = ['reasoning', 'prose', 'command']
        cell = lambda k, c, m: tk.get(k, {}).get(f'{c}/{m}', {})  # noqa: E731
        t = ['<table class="grid"><tr><th>measure</th>' + ''.join(f'<th>{esc(lab)}<br>{c}</th>' for k, lab, _ in SETS if k in tk for c in cls) + '</tr>']
        for m in B['config']['token_measures']:
            t.append(f'<tr><td>{esc(MEASURE_LABEL[m])}</td>' + ''.join(
                f'<td style="background:{heat(cell(k, c, m).get("auroc"))}">{fmtv(cell(k, c, m).get("auroc"))}</td>'
                for k, _, _ in SETS if k in tk for c in cls) + '</tr>')
        t.append('</table>')
        means = ['<table class="grid"><tr><th>class</th>' + ''.join(f'<th>{esc(lab)}<br>teacher / own</th>' for k, lab, _ in SETS if k in tk) + '</tr>']
        for c in cls:
            means.append(f'<tr><td>{c} NLL</td>' + ''.join(
                f'<td>{fmtv(cell(k, c, "nll").get("teacher_mean"))} / {fmtv(cell(k, c, "nll").get("sample_mean"))}</td>' for k, _, _ in SETS if k in tk) + '</tr>')
        means.append('</table>')
        parts.append(('b1t', 'Benchmark 1b · Token level, per class',
                      '<p>Pooled AUROC of teacher tokens against the student\'s own tokens of the same class (same turns).</p>'
                      + ''.join(t) + '<p>Mean NLL, teacher tokens vs the student\'s own tokens:</p>' + ''.join(means)))
    # B2
    b2 = B.get('b2') or {}
    if b2:
        cps = ['base'] + B['config']['sft']
        xs = ['base', 'pass 1', 'pass 2', 'pass 3']
        series = [(c, col, [b2['by_class'][c]['base']] + [b2['by_class'][c][s] for s in B['config']['sft']])
                  for c, col in (('reasoning', '#7048e8'), ('prose', '#c9a227'), ('command', '#2f9e44'))]
        series.append(('all content', '#212529', [b2['mean_nll_by_checkpoint'][k] for k in cps]))
        cats = b2['categories']
        catbar = ''.join(f'<div class="cat"><b>{v:.1%}</b><span>{k[0]}→{k[1]}</span></div>' for k, v in cats.items())
        bm = b2.get('predict_stay_hard_base_matched', {})
        prow = [(MEASURE_LABEL.get(p, p), {'x': (v['auroc'], v['ci']), 'm': (bm.get(p), None)})
                for p, v in b2['predict_stay_hard'].items()]
        prow.sort(key=lambda r: -r[1]['x'][0])
        sprow = sorted(b2['spearman_residual'].items(), key=lambda kv: -kv[1])
        parts.append(('b2', 'Benchmark 2 · Which tokens does SFT fail to teach?',
                      f'<p>The clean-pair vanilla SFT (09-21, lr 3e-4, 3 passes over 1,877 CalibForge teacher episodes) '
                      f'trained on these exact 20 trajectories. Teacher-forced NLL of the same tokens after each pass. '
                      f'H = NLL ≥ {B["config"]["tau"]} nats (p &lt; 0.1). {b2["n_hard"]:,} of {b2["n_tokens"]:,} content '
                      f'tokens start hard; {b2["stay_hard_rate"]:.1%} of those are still hard after three passes.</p>'
                      + '<div class="row">' + lines(series, xs, 'Mean NLL of the teacher tokens across SFT passes') +
                      f'<div><div class="cats">{catbar}</div><p class="hint">Rho-1 style categories, base → pass 3. '
                      f'Learned at pass 1 then hard again at pass 3: {b2["forgetting_events"]:.2%} of tokens.</p></div></div>'
                      + hbars(prow, [('x', 'AUROC among tokens that start hard', '#2b8a3e'),
                                     ('m', 'same, at equal starting NLL (within deciles)', '#8ce99a')],
                              'Predicting which initially-hard tokens stay hard after 3 passes (base-model measures)', 0.3, 0.9, 0.5)
                      + '<p>Spearman of each base measure with the token\'s NLL after pass 3 (all content tokens): '
                      + ', '.join(f'{esc(MEASURE_LABEL.get(p, p))} {v:.2f}' for p, v in sprow) + '.</p>'))
    # B3
    if B.get('b3'):
        groups = [(k, k.replace('dk0921', '09-21').replace('rlvr1s12', 'RLVR1').replace('/full_history', '').replace('/norefeed', ' no-refeed'), col)
                  for k, col in zip(B['b3'], ('#495057', '#adb5bd'))]
        rows = [(MEASURE_LABEL[m], {k: (B['b3'][k][m], None) for k, _, _ in groups}) for m in B['config']['token_measures']]
        parts.append(('b3', 'Benchmark 3 · Is the measure a property of the text or of one student?',
                      '<p>Token-level Spearman of each measure between two students, and between the two history views.</p>'
                      + hbars(rows, groups, 'Token-level Spearman between views', 0.0, 1.0)))
    # B4
    if labels and wins:
        per = collections.defaultdict(collections.Counter)
        for wid, w in wins.items():
            lab = labels.get(wid)
            if not lab:
                continue
            for m in w['methods']:
                per[m][lab['cat']] += 1
        rows = sorted(per.items(), key=lambda kv: -sum(kv[1].get(c, 0) for c in INFORMATIVE) / max(1, sum(kv[1].values())))
        parts.append(('b4', 'Benchmark 4 · Are the flagged spans worth teaching?',
                      f'<p>The 25 top tokens of each measure (one per turn) with their context, judged blind to the method '
                      f'by Opus into the categories below ({len(labels)} windows). Informative = missing knowledge, strategy, '
                      f'reading an observation, or relying on earlier reasoning.</p>'
                      + stacked([(MEASURE_LABEL.get(m, m), c) for m, c in rows], 'What the top-flagged spans are, per measure')))
        ex = []
        for m, _ in rows:
            ws = [w for w in wins.values() if m in w['methods'] and w['id'] in labels][:3]
            items = ''.join(
                f'<div class="ex"><div class="exm">{esc(w["id"])} · <b>{esc(dict((c, l) for c, l, _ in CATS).get(labels[w["id"]]["cat"], "?"))}</b> — '
                f'{esc(labels[w["id"]].get("why", ""))} · <a href="index.html#{w["traj"]}:{w["turn"]}">open</a></div>'
                f'<div class="ext">…{esc(w["before"][-220:])}<mark>{esc(w["token"])}</mark>{esc(w["after"][:120])}…</div></div>' for w in ws)
            ex.append(f'<details><summary>{esc(MEASURE_LABEL.get(m, m))}</summary>{items}</details>')
        parts.append(('ex', 'Examples per measure', ''.join(ex)))
    summary = open(a.summary).read() if os.path.exists(a.summary) else ''
    nav = ''.join(f'<a href="#{k}">{esc(t.split("·")[0].strip())}</a>' for k, t, _ in parts)
    body = ''.join(f'<section id="{k}"><h2>{esc(t)}</h2>{b}</section>' for k, t, b in parts)
    page = TEMPLATE.replace('{{SUMMARY}}', summary).replace('{{NAV}}', nav).replace('{{BODY}}', body)
    open(a.out, 'w').write(page)
    print('wrote', a.out, len(page))


def heat(v):
    if v is None:
        return 'transparent'
    d = max(-1, min(1, (v - 0.5) * 2.5))
    return f'rgba(214,40,40,{0.6 * d:.2f})' if d > 0 else f'rgba(34,102,214,{-0.6 * d:.2f})'


def fmtv(v):
    return '–' if v is None else f'{v:.2f}'


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SFTA Method Benchmark</title>
<style>
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e2e2de;--accent:#3b5bdb}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#161618;--panel:#1e1e21;--ink:#e8e8ea;--muted:#9a9aa2;--line:#333338;--accent:#8ea2ff}}
:root[data-theme="dark"]{--bg:#161618;--panel:#1e1e21;--ink:#e8e8ea;--muted:#9a9aa2;--line:#333338;--accent:#8ea2ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif}
header{position:sticky;top:0;background:var(--panel);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;z-index:5}
header b{font-size:15px}header a{color:var(--accent);text-decoration:none;font-size:13px}
main{max-width:1000px;margin:0 auto;padding:10px 16px 60px}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:16px 0}
h2{font-size:16px;margin:2px 0 8px}.summary{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:16px 0}
.summary h3{margin:4px 0 6px;font-size:15px}.summary li{margin:4px 0}
svg.chart{width:100%;height:auto;margin:6px 0}svg .ct{font-weight:600;font-size:13px;fill:var(--ink)}svg .lg,svg .rl{font-size:11px;fill:var(--ink)}
svg .ax{font-size:10px;fill:var(--muted)}svg .grid{stroke:var(--line)}svg .ref{stroke:var(--ink);stroke-dasharray:3 3}svg .ci{stroke:var(--ink);stroke-width:1.2}
table.grid{border-collapse:collapse;font-size:12px;margin:8px 0;width:100%}table.grid td,table.grid th{border:1px solid var(--line);padding:3px 6px;text-align:center}
table.grid td:first-child{text-align:left}.row{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}.row>*{flex:1 1 300px}
.cats{display:flex;gap:8px;margin-top:30px}.cat{border:1px solid var(--line);border-radius:8px;padding:6px 10px;text-align:center}.cat b{display:block;font-size:16px}.cat span{font-size:12px;color:var(--muted)}
.hint{color:var(--muted);font-size:12px}details{margin:6px 0}summary{cursor:pointer;font-weight:600}
.ex{border-left:3px solid var(--line);padding:4px 10px;margin:6px 0}.exm{font-size:12px;color:var(--muted)}.exm a{color:var(--accent)}
.ext{font:12px/1.5 ui-monospace,Menlo,monospace;white-space:pre-wrap;word-break:break-word}mark{background:#ffd8a8;color:#000}
@media (max-width:640px){main{padding:8px 16px 40px}section{padding:12px}}
</style></head><body>
<header><b>SFTA method benchmark</b>{{NAV}}<a href="index.html">← token viewer</a></header>
<main><div class="summary">{{SUMMARY}}</div>{{BODY}}</main></body></html>"""


if __name__ == '__main__':
    main()
