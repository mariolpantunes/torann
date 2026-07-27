"""Full session report: what was measured, what it means, what is next.

Reads the machine-written results where they exist —
``out/bench_base.json`` (the standing sweep) and both
``out/recall_ablation*.json`` — and carries the `exploration/` tables as
labelled constants, since those scripts print rather than serialise. Every
constant block names the script that produced it so it can be re-derived.

Usage::

    python examples/report_session.py
    python examples/report_session.py --out out/session_report.html
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from report_recall_ablation import (ARM_ORDER, CSS, aggregate,  # noqa: E402
                                    bars, table)

OUT = os.path.join(os.path.dirname(__file__), "out")

# --- Constants transcribed from exploration/ output ------------------------
# Each block is reproducible by running the named script; they print tables
# rather than JSON, so they are carried here rather than read.

# exploration/exp_cellsize.py — force weight by neighbour rank, on
# ESS-converged points, using the gaussian law esa defaults to.
FORCE_BY_RANK = {
    2: {"r1": 0.1586, "r8": 0.02083, "r64": 3.663e-27, "share_1_8": 99.95},
    8: {"r1": 0.1235, "r8": 0.1024, "r64": 0.01929, "share_1_8": 23.58},
    32: {"r1": 0.3490, "r8": 0.3296, "r64": 0.2989, "share_1_8": 13.27},
}

# exploration/exp_cellsize.py — capture (w >= R) vs selectivity (n/B^d).
CELLS = [
    # d, n, R, B_capture, B_occupy, cells (1+2d), verdict
    (2, 512, 0.0442, 22.63, 10.12, 5, "works"),
    (4, 2048, 0.1956, 5.11, 4.50, 9, "works"),
    (8, 2048, 0.7355, 1.36, 2.12, 17, "impossible"),
    (16, 2048, 2.2118, 0.45, 1.46, 33, "impossible"),
    (32, 10000, 5.1095, 0.20, 1.27, 65, "impossible"),
    (64, 2048, 12.4236, 0.08, 1.10, 129, "impossible"),
]

# exploration/exp_metric_contrast.py — (r64-r1)/r1 on ESS-converged points.
CONTRAST = {8: (24.0, 49.0, 74.8), 16: (7.3, 21.6, 36.7),
            32: (2.9, 11.5, 20.1), 64: (1.8, 6.4, 11.5)}
POWERS = (1.0, 0.5, 0.25)

# exploration/exp_metric_contrast.py — sigma/r1 for 10:1 rank-1 vs rank-64.
SHARPNESS = {8: (0.341, 11), 16: (0.181, 33), 32: (0.113, 81), 64: (0.088, 133)}

# exploration/exp_lp_recall.py + the live-LSH check: recall of the true
# L^0.5 top-5 from an L1 retrieval of increasing width, d=32, n=4000.
LP_RECALL = [(5, 16.0), (50, 74.3), (200, 93.2), (500, 93.2)]
LP_RANKS = {"median": 18, "p90": 55, "max": 147}

# Hubness: k-occurrence at k=5, n=2000. (label, d, p, mean, max, skew)
HUBNESS = [
    ("uniform", 8, 1.0, 5.0, 13, 0.32), ("ESS-converged", 8, 1.0, 5.0, 15, 0.47),
    ("uniform", 32, 1.0, 5.0, 14, 0.36), ("ESS-converged", 32, 1.0, 5.0, 13, 0.40),
    ("ESS-converged", 32, 0.5, 5.0, 14, 0.37),
]

# examples/bench_ess_suite.py --repeat 3, this session.
SUITE = [
    # d, anchors, cands, L, per-query total, batched total, speed-up, recall
    (2, 0, 256, 0, 0.17, 0.14, None, "exact"),
    (2, 256, 512, 4, 0.24, 0.21, 1.14, "1.000"),
    (8, 0, 1024, 8, 0.48, 0.39, 1.24, "0.991"),
    (8, 1024, 2048, 8, 1.18, 0.85, 1.38, "1.000"),
    (32, 0, 10000, 24, 17.65, 12.75, 1.38, "0.689"),
    (32, 10000, 20000, 24, 101.12, 64.82, 1.56, "0.861"),
]
SUITE_TOTAL = (120.83, 79.16, 1.53, 1.65)  # per-query, batched, total x, query x

# The tuner's own formula at d=32, for three collision targets.
TUNER_L = [(0.05, 45, 23, 14), (0.10, 22, 12, 7)]

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")      # light: slots 1-3
SERIES_DARK = ("#3987e5", "#d95926", "#199e70")  # dark steps of the same hues

EXTRA_CSS = """
.tiles{display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}
.tile{background:var(--card); border:1px solid var(--rule); border-radius:4px;
  padding:18px 20px; display:flex; flex-direction:column; gap:5px}
.tile .n{font-family:ui-monospace,Menlo,monospace; font-size:26px;
  font-variant-numeric:tabular-nums; color:var(--ink); letter-spacing:-0.02em}
.tile .n.warn{color:var(--critical)}
.tile .lab{font-size:12.5px; color:var(--muted); line-height:1.45}
.chart{background:var(--card); border:1px solid var(--rule); border-radius:4px;
  padding:20px 22px; overflow-x:auto}
.chart svg{display:block; min-width:520px; width:100%; height:auto}
.s1{fill:var(--s1)} .s2{fill:var(--s2)} .s3{fill:var(--s3)}
.gridline{stroke:var(--rule); stroke-width:1}
.axis{fill:var(--muted); font-size:11px;
  font-family:ui-monospace,Menlo,monospace}
.dlabel{fill:var(--ink-2); font-size:10.5px;
  font-family:ui-monospace,Menlo,monospace}
.lineser{fill:none; stroke-width:2; stroke-linecap:round;
  stroke-linejoin:round}
:root{--s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){--s1:#3987e5; --s2:#d95926; --s3:#199e70}
}
:root[data-theme="dark"]{--s1:#3987e5; --s2:#d95926; --s3:#199e70}
.nav{display:flex; flex-wrap:wrap; gap:8px 18px; font-size:13px}
.nav a{color:var(--ink-2); text-decoration:none; border-bottom:1px solid var(--rule)}
.nav a:hover{color:var(--accent); border-bottom-color:var(--accent)}
.verdict.warn{border-left-color:var(--critical)}
.stack{display:flex; flex-direction:column; gap:12px}
.note{font-size:13.5px; color:var(--muted); max-width:68ch}
"""


def tiles(items):
    """Headline numbers. `items` are (value, label, warn)."""
    out = ['<div class="tiles">']
    for value, label, warn in items:
        cls = "n warn" if warn else "n"
        out.append(f'<div class="tile"><div class="{cls}">{value}</div>'
                   f'<div class="lab">{label}</div></div>')
    out.append("</div>")
    return "\n".join(out)


def grouped_bars(cats, series_names, values, ymax, unit="%", width=760,
                 height=250):
    """Grouped vertical bars: `values[s][c]`. One legend, direct labels."""
    left, bottom, top = 46, 34, 14
    plot_w, plot_h = width - left - 12, height - bottom - top
    ns, nc = len(series_names), len(cats)
    group_w = plot_w / nc
    bar_w = min(26, (group_w - 12) / ns)
    svg = [f'<svg viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="grouped bar chart">']
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * (1 - frac)
        svg.append(f'<line class="gridline" x1="{left}" y1="{y:.1f}" '
                   f'x2="{width - 12}" y2="{y:.1f}"/>')
        svg.append(f'<text class="axis" x="{left - 8}" y="{y + 4:.1f}" '
                   f'text-anchor="end">{ymax * frac:.0f}{unit}</text>')
    for ci, cat in enumerate(cats):
        gx = left + ci * group_w
        for si in range(ns):
            v = values[si][ci]
            h = plot_h * (v / ymax)
            x = gx + (group_w - ns * bar_w - (ns - 1) * 3) / 2 \
                + si * (bar_w + 3)
            y = top + plot_h - h
            svg.append(f'<rect class="s{si + 1}" x="{x:.1f}" y="{y:.1f}" '
                       f'width="{bar_w:.1f}" height="{max(h, 1):.1f}" rx="2"/>')
            svg.append(f'<text class="dlabel" x="{x + bar_w / 2:.1f}" '
                       f'y="{y - 4:.1f}" text-anchor="middle">{v:g}</text>')
        svg.append(f'<text class="axis" x="{gx + group_w / 2:.1f}" '
                   f'y="{height - 12}" text-anchor="middle">{cat}</text>')
    svg.append("</svg>")
    legend = '<div class="legend">' + "".join(
        f'<span class="key"><span class="swatch" '
        f'style="background:var(--s{i + 1})"></span>{n}</span>'
        for i, n in enumerate(series_names)) + "</div>"
    return f'<div class="chart">{"".join(svg)}</div>{legend}'


def line_chart(xs, ys, xlabels, ymax=100, width=760, height=240,
               annotate=None):
    """Single-series line with emphasised points and direct labels."""
    left, bottom, top = 46, 34, 16
    plot_w, plot_h = width - left - 16, height - bottom - top
    n = len(xs)
    px = [left + (plot_w * i / max(n - 1, 1)) for i in range(n)]
    py = [top + plot_h * (1 - y / ymax) for y in ys]
    svg = [f'<svg viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="retrieval curve">']
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * (1 - frac)
        svg.append(f'<line class="gridline" x1="{left}" y1="{y:.1f}" '
                   f'x2="{width - 16}" y2="{y:.1f}"/>')
        svg.append(f'<text class="axis" x="{left - 8}" y="{y + 4:.1f}" '
                   f'text-anchor="end">{ymax * frac:.0f}%</text>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{px[i]:.1f},{py[i]:.1f}"
                    for i in range(n))
    svg.append(f'<path class="lineser" d="{path}" stroke="var(--s1)"/>')
    for i in range(n):
        svg.append(f'<circle cx="{px[i]:.1f}" cy="{py[i]:.1f}" r="4.5" '
                   f'fill="var(--s1)" stroke="var(--card)" stroke-width="2"/>')
        svg.append(f'<text class="dlabel" x="{px[i]:.1f}" '
                   f'y="{py[i] - 11:.1f}" text-anchor="middle">{ys[i]:g}%</text>')
        svg.append(f'<text class="axis" x="{px[i]:.1f}" y="{height - 12}" '
                   f'text-anchor="middle">{xlabels[i]}</text>')
    if annotate:
        y = top + plot_h * (1 - annotate / ymax)
        svg.append(f'<line class="gridline" x1="{left}" y1="{y:.1f}" '
                   f'x2="{width - 16}" y2="{y:.1f}" '
                   f'stroke="var(--critical)" stroke-dasharray="4 3"/>')
    svg.append("</svg>")
    return f'<div class="chart">{"".join(svg)}</div>'


def simple_table(head, rows, cls=""):
    out = [f'<div class="tablewrap"><table class="{cls}"><thead><tr>'
           + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def sweep_summary(rows):
    """Geometric mean ms/epoch by engine, from the standing sweep."""
    out = {}
    for engine in ("brute", "lsh"):
        v = [r["ms_per_epoch"] for r in rows if r["engine"] == engine]
        q = [r["query_ms_per_epoch"] for r in rows if r["engine"] == engine]
        out[engine] = (len(v), float(np.exp(np.mean(np.log(v)))),
                       100 * sum(q) / sum(v))
    return out


def render(sweep, abl_rows):
    stats = aggregate(abl_rows)
    shapes = sorted(stats, key=lambda s: (-s[0], s[1]))
    sw = sweep_summary(sweep)

    def pct(shape, arm):
        s = stats[shape]
        return 100 * (s[arm]["ce"] - s["exact"]["ce"]) / s["exact"]["ce"]

    empty32 = next(s for s in shapes if s[0] == 32 and s[1] == 0)
    fill32 = next((s for s in shapes if s[0] == 32 and s[1] > 0), None)

    # --- ablation facets ---------------------------------------------------
    facets = []
    for shape in shapes:
        dim, anchors, cands = shape
        start = "filled start" if anchors else "empty start"
        facets.append(
            f'<div class="facet"><h3>d = {dim} <span>&nbsp;·&nbsp; {start}'
            f'&nbsp;·&nbsp; {anchors}+{cands}</span></h3>'
            f'{bars(stats[shape], "ce", stats[shape]["uniform"]["ce"])}</div>')

    abl_tables = "".join(
        f'<section><h3>d = {d} &nbsp;·&nbsp; '
        f'{"filled" if a else "empty"} start &nbsp;·&nbsp; {a}+{c}</h3>'
        f'{table(stats[(d, a, c)])}</section>' for (d, a, c) in shapes)

    # --- charts ------------------------------------------------------------
    contrast_chart = grouped_bars(
        [f"d={d}" for d in sorted(CONTRAST)],
        [f"p = {p}" for p in POWERS],
        [[CONTRAST[d][i] for d in sorted(CONTRAST)] for i in range(3)],
        ymax=80)

    force_chart = grouped_bars(
        [f"d={d}" for d in sorted(FORCE_BY_RANK)],
        ["share of the vote from ranks 1–8"],
        [[FORCE_BY_RANK[d]["share_1_8"] for d in sorted(FORCE_BY_RANK)]],
        ymax=100)

    lp_chart = line_chart([w for w, _ in LP_RECALL],
                          [r for _, r in LP_RECALL],
                          [f"k={w}" for w, _ in LP_RECALL],
                          annotate=93.2)

    return f"""<title>torann × ESS — session report</title>
<style>{CSS}{EXTRA_CSS}</style>
<div class="wrap">

<header class="prose">
  <p class="eyebrow">torann · branch query-batched-join · 2026-07-27</p>
  <h1>What the index is actually being asked to do</h1>
  <p class="sub">One open question answered, one new wall found. Recall has
  a floor rather than a price — and above d≈8 the force law consuming those
  neighbours stops being able to tell them apart, which changes the
  problem.</p>
  <nav class="nav">
    <a href="#baseline">Baselines</a>
    <a href="#recall">Recall ablation</a>
    <a href="#force">The force-law wall</a>
    <a href="#grid">Grid feasibility</a>
    <a href="#metric">Metric contrast</a>
    <a href="#lp">L^p retrieval ceiling</a>
    <a href="#tuner">The tuner</a>
    <a href="#ruled-out">Ruled out</a>
    <a href="#next">What is next</a>
  </nav>
</header>

<hr class="rule">

<section>
  {tiles([
      (f"{abs(pct(empty32, 'top2k')):.2f}%",
       "CE cost of halving recall at d=32, empty start — the slack the "
       "index can spend", False),
      (f"{abs(pct(empty32, 'rank1-4k')):.0f}%",
       "CE cost of losing all true neighbours, though substitutes are only "
       "5% further away", True),
      ("86%",
       "how hard the 64th neighbour pushes relative to the 1st at d=32 — "
       "the force law cannot discriminate", True),
      ("93.2%",
       "ceiling on L^0.5 recall from the L1 hash, at any retrieval width", True),
  ])}
</section>

<section class="prose" id="baseline">
  <h2>Baselines, and the trap underneath them</h2>
  <p>ESS imports the <em>installed</em> torann, not the repository. Every
  number here was taken after rebuilding and reinstalling the wheel, with
  the installed <code>.so</code> hash checked against the one built from
  this branch. An earlier session lost an hour to measuring a 12-day-old
  build twice.</p>
  <p>The standing sweep, 48 configurations, reported per epoch:
  <strong>brute {sw['brute'][1]:.1f} ms/epoch</strong> geometric mean
  ({sw['brute'][2]:.0f}% of it query), <strong>LSH {sw['lsh'][1]:.2f}
  ms/epoch</strong> ({sw['lsh'][2]:.0f}% query). One sweep run was
  discarded and repeated because another job overlapped ~90 s of it.</p>
</section>

<section>
  <div class="prose"><h3>The six ESS shapes, batched join vs per-query</h3>
  <p class="note">Every shape's trajectory fingerprint matched — results are
  bit-identical across the two paths.</p></div>
  {simple_table(
      ("d", "shape", "L", "per-query", "batched", "speed-up", "recall"),
      [(d, f"{a}+{c}", ell or "—", f"{pq:.2f}s", f"{b:.2f}s",
        f"{su:.2f}×" if su else "n/a", rec)
       for d, a, c, ell, pq, b, su, rec in SUITE]
      + [("<strong>all</strong>", "<strong>six shapes</strong>", "",
          f"<strong>{SUITE_TOTAL[0]:.2f}s</strong>",
          f"<strong>{SUITE_TOTAL[1]:.2f}s</strong>",
          f"<strong>{SUITE_TOTAL[2]:.2f}×</strong>",
          f"query {SUITE_TOTAL[3]:.2f}×")])}
</section>

<section class="prose" id="recall">
  <h2>Does ESS need recall, or only plausible repellers?</h2>
  <p>The evidence was ambiguous because it confounded two things: cutting
  LSH tables degrades recall <em>and</em> locality together. This separates
  them. The index is exact throughout, so recall is <strong>imposed</strong>
  rather than measured — the neighbour list is corrupted between the query
  and the force kernel, and the true toroidal-L1 distance of every
  substituted point is passed through, so only the selection changes and
  force magnitudes are never confounded with it.</p>
</section>

<section>
  <div class="prose">
    <div class="verdict">
      <p class="eyebrow">Answer</p>
      <p><strong>Recall has a floor, not a price.</strong> Half the true
      neighbours can go almost free — {abs(pct(empty32, 'top2k')):.2f}% CE at
      d=32{"" if not fill32 else
      f", {abs(pct(fill32, 'top2k')):.2f}% from the filled start"} — but
      losing all of them collapses the result to the uniform null, even
      when the substitutes are only 5% further away. A distance
      <em>ranking</em> is the product. Redesigning the index around cost per
      <em>plausible local neighbour</em> is ruled out.</p>
      {"" if not fill32 else
       f'<p>The filled start — anchors that cannot be pushed, which is the '
       f'case ESS is actually for — is uniformly <em>less</em> sensitive '
       f'({abs(pct(fill32, "rank1-4k")):.1f}% at recall 0 against '
       f'{abs(pct(empty32, "rank1-4k")):.1f}% empty), so the slack measured '
       f'from an empty start is a lower bound on the slack in production.</p>'}
    </div>
  </div>
  <div class="legend">
    <span class="key"><span class="swatch"></span>at or above the uniform
    null</span>
    <span class="key"><span class="swatch crit"></span>below it — actively
    clustered</span>
    <span class="key"><span class="swatch null"></span>uniform null</span>
  </div>
  <div class="facets">{"".join(facets)}</div>
</section>

<section>
  <div class="prose"><h3>The full record</h3></div>
  {abl_tables}
</section>

<section class="prose" id="force">
  <h2>The new wall: above d≈8 the force law stops discriminating</h2>
  <p>Everything above treats the index's job as "return the true k
  nearest". This is about the fact that the thing consuming those
  neighbours can no longer tell them apart. Measured on ESS-converged
  points with the Gaussian law <code>esa</code> defaults to:</p>
</section>

<section>
  {force_chart}
  {simple_table(
      ("d", "rank 1 weight", "rank 8", "rank 64", "share from ranks 1–8"),
      [(d, f"{v['r1']:.4g}", f"{v['r8']:.4g}", f"{v['r64']:.3g}",
        f"{v['share_1_8']:.2f}%") for d, v in sorted(FORCE_BY_RANK.items())])}
  <div class="prose stack">
    <p>At d=32 the 64th neighbour pushes <strong>86% as hard as the
    nearest</strong>, because all 64 distances lie in [6.13, 6.31] — a 3%
    spread. Three consequences:</p>
    <p><strong>Magnitude carries almost no information at high d.</strong>
    The entire signal is <em>which</em> points come back; a wrongly-returned
    far neighbour votes at nearly full strength. That is the mechanism
    behind the collapse above.</p>
    <p><strong>Radius mode cannot work at d=32 either</strong>, since
    "inside R" stops being a distinction. Related: at convergence
    <em>zero</em> neighbours lie inside R at any shape tested — the nearest
    sits at 1.15–1.4·R, because ESS pushes to equilibrium just outside the
    interaction radius.</p>
    <p><strong>ESS destroys its own contrast as it converges.</strong>
    Relative spread at d=32 is 24.2% on uniform points against 2.9% on
    converged ones — making points equidistant is precisely its job. Any
    benchmark on random data measures an easier problem than the real one.</p>
  </div>
</section>

<section class="prose" id="grid">
  <h2>Cells sized from the force law: exact at d≤4, impossible at d≥8</h2>
  <p>Look in the query's own cell plus the face-adjacent ones — von
  Neumann, <code>1+2d</code> cells; the L1 ball makes diagonal cells
  pointless. That needs <strong>capture</strong> (width <code>w ≥ R</code>,
  since L∞ ≤ L1) and <strong>selectivity</strong> (occupancy
  <code>n/B<sup>d</sup></code> small) simultaneously. They cross between
  d=4 and d=8.</p>
</section>

<section>
  {simple_table(
      ("d", "n", "R", "cells/dim allowed (capture)",
       "needed (occupancy)", "1+2d", "verdict"),
      [(d, n, f"{r:.4f}", f"{bc:.2f}", f"{bo:.2f}", cells,
        f'<span style="color:var(--critical)">{v}</span>'
        if v == "impossible" else v)
       for d, n, r, bc, bo, cells, v in CELLS])}
  <p class="note">At d≥16 the capturing cell is <em>larger than the whole
  domain</em>. This is the same wall as the measured geometry elsewhere,
  now in the force law's own terms — and it is why a ring-search engine
  stays scoped to d≤4.</p>
</section>

<section class="prose" id="metric">
  <h2>Lower p restores contrast — about two octaves of dimension</h2>
  <p>Relative spread <code>(r₆₄−r₁)/r₁</code> on ESS-converged points. p=0.25
  at d=32 beats L1 at d=16, and approaches L1 at d=8.</p>
</section>

<section>
  {contrast_chart}
  <div class="prose"><p class="note">The required force sharpness relaxes
  with it — σ/r₁ for 10:1 discrimination goes from
  {SHARPNESS[32][0]:.3f} at p=1 to 0.309 at p=0.25 — so the two fixes
  reinforce rather than compete. A power law is not the route: reaching the
  same 10:1 needs r<sup>−{SHARPNESS[32][1]}</sup> at d=32 and
  r<sup>−{SHARPNESS[64][1]}</sup> at d=64.</p></div>
</section>

<section class="prose" id="lp">
  <h2>But the L1 hash cannot retrieve L^p neighbours</h2>
  <p>The grid hash collides with probability <code>1 − B·δ</code> per
  sampled dimension, so <code>log P(collide) ~ −B·Σδⱼ</code> — <strong>a
  function of L1 and nothing else</strong>. Two points with equal L1 and
  very different L^0.5 are retrieved with equal probability, so ranking its
  candidates by L^p reranks a set that was never selected for it.</p>
</section>

<section>
  {lp_chart}
  <div class="prose stack">
    <p>Measured against the real LSH at d=32, n=4000. The dashed line is
    the plateau: <strong>it is not retrieval width, it is the hash's own L1
    recall</strong> (0.689 here). The missing ~7% never collide, so no
    rerank at any width reaches them.</p>
    <p>The mitigating fact is that the L^p-nearest are not <em>far</em> in
    L1 — median L1 rank {LP_RANKS['median']}, 90th percentile
    {LP_RANKS['p90']}, max {LP_RANKS['max']} — and the refine kernel already
    scores ~4118 candidates per query, so the rerank itself costs one
    <code>vsqrtps</code> per lane plus a bigger heap, not extra retrieval.
    <strong>But do not ship it as an L^p index.</strong> The cap is
    structural.</p>
  </div>
</section>

<section class="prose" id="tuner">
  <h2>The tuner optimises a target nobody chose</h2>
  <p><code>_tune</code> sets <code>L = ceil(log(0.10) / log(1 − p1))</code>,
  clamped to [4, 24]. That <code>0.10</code> is a hardcoded <strong>90%
  collision target for a true k-NN</strong> — a recall objective never
  derived from what ESS needs. Query is 68–85% of ESS wall time and L drives
  probe count linearly:</p>
</section>

<section>
  {simple_table(
      ("per-table collision p1", "L for 90% (current)",
       "for 69% (delivered)", "for 50%"),
      [(f"{p:.2f}", a, b, f"<strong>{c}</strong>")
       for p, a, b, c in TUNER_L])}
  <div class="prose stack">
    <div class="verdict warn">
      <p class="eyebrow">The contradiction worth resolving first</p>
      <p>Imposed recall 0.5 costs <strong>0.45%</strong> CE. But cutting
      tables to L=12 costs <strong>4.3%</strong> and L=8 costs
      <strong>7.3%</strong>. Those cannot both describe "less recall", so
      cutting tables must be doing something else.</p>
      <p>Two candidates, both testable in ~20 minutes: misses are replaced
      by <em>far</em> points rather than omitted (the −18% regime), or low
      recall flattens the force EMA and fires the early-stop prematurely, so
      the loss is a convergence artifact. The ablation's epoch counts favour
      the second — corrupted arms stopped at 40 epochs against 151 for
      exact. <strong>The measurement:</strong> run at L=24 and L=8 and record
      the true rank of every returned neighbour.</p>
    </div>
    <p class="note">What is missing is not a knob —
    <code>num_tables</code>, <code>resolution</code>,
    <code>dims_per_table</code>, <code>target_bucket_size</code>,
    <code>probes</code>, <code>query_block_size</code> and
    <code>brute_threshold</code> all override the tuner — but an
    <em>objective</em>: no way to say "I need recall 0.5, not 0.9".</p>
  </div>
</section>

<section class="prose" id="ruled-out">
  <h2>Ruled out this session</h2>
</section>

<section>
  {simple_table(
      ("condition", "d", "p", "mean k-occurrence", "max", "skew"),
      [(lab, d, p, f"{m:.1f}", mx, f"+{sk:.2f}")
       for lab, d, p, m, mx, sk in HUBNESS])}
  <div class="prose stack">
    <p><strong>Hubness.</strong> No runaway hubs anywhere — skew +0.32…+0.47
    with a maximum k-occurrence of 13–15 against a mean of 5, for uniform
    and converged points, at both dimensions, under p=1 and p=0.5. So
    hubness-corrected metrics (mutual proximity, local scaling) are not a
    lead.</p>
    <p><strong>A power-law force.</strong> r<sup>−81</sup> at d=32 for 10:1
    discrimination. The Gaussian's r² in the exponent is what makes the same
    job reachable.</p>
    <p><strong>The original "within 2× the k-th distance" arm.</strong> At
    d=32 it agrees with uniform random to four decimals on every column —
    the ball has swallowed the whole point set. Replaced by rank-window arms
    that pin recall at zero while varying locality independently.</p>
  </div>
</section>

<section class="prose" id="next">
  <h2>What is next</h2>
  <p><strong>A hash family designed natively for p &lt; 1.</strong> The
  target is per-dimension collision <code>~exp(−c·δ^p)</code>. The lever is
  randomising the cell width: <code>E_w[max(0, 1−δ/w)]</code> represents any
  convex decreasing f, and <code>exp(−c·δ^p)</code> is convex for p&lt;1.
  The obstacle is toroidal — the wrap needs <code>w = 1/B</code> with
  <em>integer</em> B, capping the mixing distribution at w=0.5 instead of a
  continuum. That is theory to settle before code. The bar to beat is the
  93.2% rerank ceiling; below it, the construction has bought nothing.</p>
  <p>A sharper force law (σ 0.5 → ~0.11 at d=32) is the other half of the
  fix and belongs to ESS, not to the index.</p>
  <p><strong>Not yet measured.</strong> ESS is a refinement method — called
  repeatedly, each call handed the points it produced before — so the
  filled start is the normal case and the static tier grows without bound.
  Every benchmark here measures a single call, and each call re-fits the
  index and re-tunes (B,K,L) from scratch.
  <code>bench_refine_rounds.py</code> is written for exactly this and has
  never been run. Separately, <strong>there is no FAISS comparison on the
  ESS workload at all</strong> — the existing scripts use uniform random
  data at k=2·D — so "competitive with FAISS on ESS" should be treated as
  unestablished.</p>
</section>

<footer>
  <p>Generated by <code>examples/report_session.py</code> from
  <code>bench_base.json</code> and <code>recall_ablation*.json</code>;
  the <code>exploration/</code> tables are carried as labelled constants
  reproducible by re-running those scripts. Clark-Evans and separation are
  toroidal L1, with the exact fixed-n null rather than the Poisson
  asymptotic. Ablation figures are 3 seeds, paired by seed.</p>
</footer>
</div>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default=os.path.join(OUT, "bench_base.json"))
    ap.add_argument("--ablation", nargs="+",
                    default=[os.path.join(OUT, "recall_ablation.json"),
                             os.path.join(OUT, "recall_ablation_anchored.json")])
    ap.add_argument("--out", default=os.path.join(OUT, "session_report.html"))
    args = ap.parse_args()

    with open(args.sweep) as fh:
        sweep = json.load(fh)
    abl = [r for p in args.ablation for r in json.load(open(p))]

    os.makedirs(OUT, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(render(sweep, abl))
    print(f"[wrote {args.out}]")
