"""Render `bench_recall_ablation.py`'s JSON as a standalone HTML report.

Reads ``out/recall_ablation.json`` and writes ``out/recall_ablation.html`` —
one self-contained file, no external assets, readable in light or dark.

The report is built around the ladder: the arms are ordered by how local
their neighbour selection is, so the reader can see *where* along that
ladder quality falls off rather than only that it does.

Usage::

    python examples/report_recall_ablation.py
    python examples/report_recall_ablation.py --json out/recall_ablation.json
"""

import argparse
import collections
import html
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "out")

# The ladder, most local first, with the one-line gloss the report prints.
ARM_ORDER = ("exact", "top2k", "rank1-4k", "rank8-16k", "ratio2x", "uniform")
ARM_GLOSS = {
    "exact": "True k nearest neighbours. The control.",
    "top2k": "k drawn from the true 2k nearest — half the true neighbours "
             "survive, and locality is essentially untouched.",
    "rank1-4k": "k drawn from ranks k..4k. Not one true neighbour, but the "
                "nearest points that are not true neighbours.",
    "rank8-16k": "k drawn from ranks 8k..16k. Still local relative to n, "
                 "several times the true neighbour distance.",
    "ratio2x": "k drawn from everything within 2x the k-th distance — the "
               "arm as originally specified.",
    "uniform": "k uniformly random points. The null.",
}


def load(path):
    with open(path) as fh:
        return json.load(fh)


def aggregate(rows):
    """Mean/sd per (shape, arm), keyed in ladder order.

    Returns:
        dict: ``{(dim, anchors, cands): {arm: stats}}``, seeds collapsed.
    """
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[(r["dim"], r["anchors"], r["candidates"])][r["arm"]].append(r)
    out = {}
    for shape, arms in by.items():
        out[shape] = {}
        for arm, rs in arms.items():
            ce = np.array([r["clark_evans"] for r in rs])
            out[shape][arm] = {
                "ce": float(ce.mean()),
                "ce_sd": float(ce.std(ddof=1)) if len(ce) > 1 else 0.0,
                "sep": float(np.mean([r["separation"] for r in rs])),
                "recall": float(np.mean([r["recall"] for r in rs])),
                "ratio": float(np.mean([r["mean_ratio"] for r in rs])),
                "epochs": float(np.mean([r["epochs"] for r in rs])),
                "seeds": len(rs),
            }
    return out


def bars(shape_stats, key, null_value):
    """One facet's worth of `<div>` bars, as an HTML string.

    A bar is drawn against the arm's own scale maximum; the uniform null is
    marked with a rule so "below random" is visible as a position, not only
    as a number. Bars that fall under it take the critical colour and say
    so in words — colour is never the only carrier.

    Args:
        shape_stats (dict): ``{arm: stats}`` for one shape.
        key (str): ``"ce"`` or ``"sep"``.
        null_value (float): The uniform arm's value, drawn as the rule.

    Returns:
        str: HTML for the facet body.
    """
    vals = [shape_stats[a][key] for a in ARM_ORDER if a in shape_stats]
    top = max(vals + [null_value]) * 1.08
    null_pct = 100.0 * null_value / top
    # Unitless: the rule's offset is a length times a number, which calc
    # allows — a percentage times a length, which it does not, is the easy
    # way to get a silently unplaced rule here.
    out = [f'<div class="plot" style="--null:{null_pct / 100:.4f}">',
           f'<div class="nullrule" aria-hidden="true"></div>']
    for arm in ARM_ORDER:
        if arm not in shape_stats:
            continue
        s = shape_stats[arm]
        v = s[key]
        pct = 100.0 * v / top
        below = v < null_value * 0.995 and arm != "uniform"
        cls = "bar below" if below else "bar"
        note = ' <span class="flag">below the null</span>' if below else ""
        out.append(
            f'<div class="row"><div class="rowname">{html.escape(arm)}</div>'
            f'<div class="track"><div class="{cls}" '
            f'style="width:{pct:.2f}%"></div>'
            f'<span class="val">{v:.4f}{note}</span></div></div>')
    out.append("</div>")
    return "\n".join(out)


def table(shape_stats):
    """The full numeric record for one shape — every column, no rounding games."""
    head = ("arm", "recall", "mean ratio", "Clark-Evans", "vs exact",
            "separation", "epochs")
    base = shape_stats["exact"]["ce"] if "exact" in shape_stats else None
    rows = ["<div class=\"tablewrap\"><table><thead><tr>"
            + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for arm in ARM_ORDER:
        if arm not in shape_stats:
            continue
        s = shape_stats[arm]
        delta = ("—" if base is None or arm == "exact"
                 else f"{100 * (s['ce'] - base) / base:+.2f}%")
        sd = f" <span class=\"sd\">±{s['ce_sd']:.4f}</span>" if s["seeds"] > 1 else ""
        rows.append(
            f"<tr{' class=\"ctl\"' if arm == 'exact' else ''}>"
            f"<td class=\"arm\">{html.escape(arm)}</td>"
            f"<td>{s['recall']:.3f}</td><td>{s['ratio']:.2f}</td>"
            f"<td>{s['ce']:.4f}{sd}</td><td>{delta}</td>"
            f"<td>{s['sep']:.4f}</td><td>{s['epochs']:.0f}</td></tr>")
    rows.append("</tbody></table></div>")
    return "\n".join(rows)


CSS = """
:root{
  color-scheme: light;
  --ground:#f7f8fa; --card:#ffffff; --ink:#10141b; --ink-2:#545c6b;
  --muted:#7b8394; --rule:#dfe3ea; --accent:#2a78d6; --critical:#e34948;
  --track:#eceff4;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --ground:#0e1116; --card:#161b22; --ink:#eef1f6; --ink-2:#aab3c0;
    --muted:#7d8797; --rule:#262d38; --accent:#3987e5; --critical:#e66767;
    --track:#1e242d;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --ground:#0e1116; --card:#161b22; --ink:#eef1f6; --ink-2:#aab3c0;
  --muted:#7d8797; --rule:#262d38; --accent:#3987e5; --critical:#e66767;
  --track:#1e242d;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:16px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1080px; margin:0 auto; padding:clamp(28px,5vw,72px) clamp(18px,4vw,40px) 96px;
  display:flex; flex-direction:column; gap:44px;}
.prose{max-width:68ch; display:flex; flex-direction:column; gap:16px;}
h1{
  font-family:ui-serif,Georgia,"Times New Roman",serif;
  font-weight:600; font-size:clamp(30px,4.4vw,46px); line-height:1.12;
  letter-spacing:-0.015em; margin:0; text-wrap:balance;
}
h2{
  font-family:ui-serif,Georgia,serif; font-weight:600;
  font-size:clamp(21px,2.6vw,27px); letter-spacing:-0.01em;
  margin:0; text-wrap:balance;
}
h3{font-size:15px; font-weight:650; margin:0; letter-spacing:-0.005em;}
p{margin:0; color:var(--ink-2);}
.prose p{max-width:68ch}
strong{color:var(--ink); font-weight:640}
.eyebrow{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:11.5px; letter-spacing:0.14em; text-transform:uppercase;
  color:var(--muted); margin:0;
}
.sub{font-size:17.5px; color:var(--ink-2); max-width:64ch; margin:0}
.rule{height:1px; background:var(--rule); border:0; margin:0}
section{display:flex; flex-direction:column; gap:20px}
.verdict{
  background:var(--card); border:1px solid var(--rule);
  border-left:3px solid var(--accent);
  border-radius:3px; padding:24px 26px;
  display:flex; flex-direction:column; gap:12px; max-width:72ch;
}
.verdict p{color:var(--ink)}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:0.9em; font-variant-numeric:tabular-nums}
code{background:var(--track); padding:0.12em 0.36em; border-radius:3px}
.facet{background:var(--card); border:1px solid var(--rule); border-radius:4px;
  padding:22px 24px; display:flex; flex-direction:column; gap:18px}
.facets{display:grid; gap:18px; grid-template-columns:1fr}
@media(min-width:900px){.facets{grid-template-columns:1fr 1fr}}
.facet h3 span{color:var(--muted); font-weight:450}
.plot{position:relative; display:flex; flex-direction:column; gap:9px}
.nullrule{position:absolute; left:calc(96px + (100% - 96px) * var(--null));
  top:0; bottom:0; width:0; border-left:1px dashed var(--muted); opacity:.75}
.row{display:grid; grid-template-columns:96px 1fr; align-items:center; gap:0}
.rowname{font-family:ui-monospace,Menlo,monospace; font-size:12px;
  color:var(--ink-2); padding-right:10px; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis}
.track{position:relative; height:22px; display:flex; align-items:center;
  background:linear-gradient(var(--track),var(--track)) no-repeat;
  background-size:100% 2px; background-position:0 50%}
.bar{height:14px; background:var(--accent); border-radius:0 3px 3px 0;
  box-shadow:0 0 0 2px var(--card)}
.bar.below{background:var(--critical)}
.val{font-family:ui-monospace,Menlo,monospace; font-size:12px;
  font-variant-numeric:tabular-nums; color:var(--ink-2); padding-left:9px;
  white-space:nowrap}
.flag{color:var(--critical); font-family:system-ui,sans-serif; font-size:11px;
  letter-spacing:.02em}
.legend{display:flex; flex-wrap:wrap; gap:16px; align-items:center;
  font-size:12.5px; color:var(--muted)}
.key{display:inline-flex; align-items:center; gap:7px}
.swatch{width:14px; height:8px; border-radius:2px; background:var(--accent);
  display:inline-block}
.swatch.crit{background:var(--critical)}
.swatch.null{width:0; height:13px; border-left:1px dashed var(--muted);
  border-radius:0}
.tablewrap{overflow-x:auto; border:1px solid var(--rule); border-radius:4px;
  background:var(--card)}
table{border-collapse:collapse; width:100%; min-width:640px; font-size:13.5px}
th{
  font-family:ui-monospace,Menlo,monospace; font-size:10.5px;
  letter-spacing:.1em; text-transform:uppercase; color:var(--muted);
  text-align:right; font-weight:500; padding:13px 16px;
  border-bottom:1px solid var(--rule); white-space:nowrap;
}
th:first-child{text-align:left}
td{padding:11px 16px; text-align:right; border-bottom:1px solid var(--rule);
  font-family:ui-monospace,Menlo,monospace; font-variant-numeric:tabular-nums;
  color:var(--ink-2); white-space:nowrap}
td.arm{text-align:left; color:var(--ink)}
tr:last-child td{border-bottom:0}
tr.ctl td{color:var(--ink); background:color-mix(in srgb,var(--accent) 7%,transparent)}
.sd{color:var(--muted); font-size:11px}
.ladder{display:flex; flex-direction:column; gap:0; max-width:72ch}
.rung{display:grid; grid-template-columns:26px 1fr; gap:14px;
  padding:14px 0; border-top:1px solid var(--rule)}
.rung:first-child{border-top:0}
.rungno{font-family:ui-monospace,Menlo,monospace; font-size:11.5px;
  color:var(--muted); padding-top:2px}
.rung b{font-family:ui-monospace,Menlo,monospace; font-size:13.5px;
  color:var(--ink); font-weight:500}
.rung p{font-size:14px; margin-top:3px}
footer{color:var(--muted); font-size:13px; max-width:68ch}
a{color:var(--accent)}
a:focus-visible,[tabindex]:focus-visible{outline:2px solid var(--accent);
  outline-offset:3px}
"""


def render(rows, source):
    stats = aggregate(rows)
    # Deepest first, and within a dimension the empty start before the
    # filled one — the two are different jobs, not repeats.
    shapes = sorted(stats, key=lambda s: (-s[0], s[1]))
    seeds = max(v["seeds"] for s in stats.values() for v in s.values())

    facets_ce, facets_sep = [], []
    for shape in shapes:
        dim, anchors, cands = shape
        s = stats[shape]
        null_ce = s["uniform"]["ce"]
        null_sep = s["uniform"]["sep"]
        start = "filled start" if anchors else "empty start"
        label = (f'<h3>d = {dim} <span>&nbsp;·&nbsp; {start} &nbsp;·&nbsp; '
                 f'{anchors}+{cands}</span></h3>')
        facets_ce.append(f'<div class="facet">{label}'
                         f'{bars(s, "ce", null_ce)}</div>')
        facets_sep.append(f'<div class="facet">{label}'
                          f'{bars(s, "sep", null_sep)}</div>')

    ladder = "".join(
        f'<div class="rung"><div class="rungno">{i}</div><div>'
        f'<b>{html.escape(a)}</b><p>{html.escape(ARM_GLOSS[a])}</p>'
        f'</div></div>'
        for i, a in enumerate(ARM_ORDER, 1))

    tables = "".join(
        f'<section><h3>d = {d} &nbsp;·&nbsp; '
        f'{"filled" if a else "empty"} start &nbsp;·&nbsp; {a}+{c}</h3>'
        f'{table(stats[(d, a, c)])}</section>'
        for (d, a, c) in shapes)

    legend = ('<div class="legend">'
              '<span class="key"><span class="swatch"></span>'
              'at or above the uniform null</span>'
              '<span class="key"><span class="swatch crit"></span>'
              'below the uniform null — actively clustered</span>'
              '<span class="key"><span class="swatch null"></span>'
              'uniform null</span></div>')

    # Headline numbers per dimension: the answer is not the same at d=8 and
    # d=32, and a report that quoted only one of them would be wrong about
    # the other.
    def pct(shape, arm):
        s = stats[shape]
        return 100 * (s[arm]["ce"] - s["exact"]["ce"]) / s["exact"]["ce"]

    def sep_pct(shape, arm):
        s = stats[shape]
        return 100 * (s[arm]["sep"] - s["exact"]["sep"]) / s["exact"]["sep"]

    hi, lo = shapes[0], shapes[-1]          # deepest and shallowest shape
    d_hi, d_lo = hi[0], lo[0]

    return f"""<title>Does ESS need recall, or only plausible repellers?</title>
<style>{CSS}</style>
<div class="wrap">
<header class="prose">
  <p class="eyebrow">torann · ESS · neighbour-selection ablation</p>
  <h1>Does ESS need recall, or only plausible repellers?</h1>
  <p class="sub">An exact index, corrupted on purpose. {seeds} seeds per arm,
  paired by seed, {len(shapes)} shapes. Recall turns out to have a
  <em>floor</em>, not a price — and the floor is low.</p>
</header>

<hr class="rule">

<section class="prose">
  <div class="verdict">
    <p class="eyebrow">Verdict</p>
    <p><strong>Half the true neighbours can go almost free; all of them
    cannot go at all.</strong> Dropping to recall 0.5 while holding locality
    fixed costs {abs(pct(hi, 'top2k')):.2f}% Clark-Evans at d={d_hi} and
    {abs(pct(lo, 'top2k')):.2f}% at d={d_lo}. Dropping to recall 0 costs
    {abs(pct(hi, 'rank1-4k')):.1f}% at d={d_hi} and
    {abs(pct(lo, 'rank1-4k')):.1f}% at d={d_lo} — collapsing to the uniform
    null or below it — <em>even when the substituted neighbours are only
    {100 * (stats[hi]['rank1-4k']['ratio'] - 1):.0f}&ndash;{
    100 * (stats[lo]['rank1-4k']['ratio'] - 1):.0f}% further away</em>.</p>
    <p>So this lands on the branch where a distance <em>ranking</em> is the
    product. The index must return real nearest neighbours, but it has
    genuine slack in how many of them are true — which is roughly where the
    tuner already sits. Redesigning around cost per <em>plausible local
    neighbour</em> is ruled out: that arm is indistinguishable from random.</p>
  </div>
</section>

<section class="prose">
  <h2>Why the earlier evidence looked ambiguous</h2>
  <p>Two measurements pointed opposite ways. Recall 0.69 at d=32 costs only
  1.28% CE against perfect recall, which suggests recall is cheap to give
  up. But cutting LSH tables to lower recall costs 4.3% CE, which suggests
  it is not. Both can be true, because fewer tables degrades recall
  <em>and</em> locality together — the experiment confounds them.</p>
  <p>This ablation separates the two. The index is exact throughout, so
  recall is not a property of the index here; it is imposed. Between the
  query and the force kernel the neighbour list is replaced by a controlled
  corruption of it, and the <strong>true toroidal-L1 distance of whatever
  was substituted is passed through</strong>. Only the selection changes.
  Force magnitudes are never confounded with it.</p>
  <p><code>mean ratio</code> below is the delivered neighbour distance over
  the true k-th distance — the locality knob, made comparable across
  dimensions. <code>recall</code> is measured against the exact answer the
  same index computed.</p>
</section>

<section>
  <div class="prose"><h2>The ladder</h2>
  <p>Ordered by how local the selection is. Only the first two arms contain
  true neighbours at all.</p></div>
  <div class="ladder">{ladder}</div>
</section>

<section>
  <div class="prose"><h2>Clark-Evans regularity</h2>
  <p>1.0 is the uniform null at every dimension. Higher is more regular —
  the thing ESS exists to produce.</p></div>
  {legend}
  <div class="facets">{"".join(facets_ce)}</div>
</section>

<section>
  <div class="prose"><h2>Separation</h2>
  <p>The minimum pairwise distance, toroidal L1. This is where the mechanism
  shows: when the true nearest neighbours are withheld, nothing repels the
  pairs that are actually touching, and they fuse.</p></div>
  {legend}
  <div class="facets">{"".join(facets_sep)}</div>
</section>

<section class="prose">
  <h2>Reading the collapse</h2>
  <p><code>rank1-4k</code> is the informative arm: its neighbours are only
  {100 * (stats[lo]['rank1-4k']['ratio'] - 1):.0f}% further away than the
  true ones at d={d_lo}, and {100 * (stats[hi]['rank1-4k']['ratio'] - 1):.0f}%
  at d={d_hi}. Being nearly-nearest buys nothing. The pairs at risk of fusing
  are exactly the ones the force law must see, and plausible mid-range
  repulsion does not compensate for missing them — separation falls
  {abs(sep_pct(lo, 'rank1-4k')):.0f}% at d={d_lo} and
  {abs(sep_pct(hi, 'rank1-4k')):.0f}% at d={d_hi}.</p>
  <p>The severity is strongly dimension-dependent, and this is the part
  worth not over-reading. At d={d_lo} a near-miss neighbour list is
  <em>worse than random</em> — CE {stats[lo]['rank1-4k']['ce']:.4f} against a
  null of {stats[lo]['uniform']['ce']:.4f} — because it actively holds points
  together while pushing on the wrong pairs, where random neighbours at
  least apply an isotropic mean-field pressure. At d={d_hi} that gap nearly
  vanishes ({stats[hi]['rank1-4k']['ce']:.4f} against
  {stats[hi]['uniform']['ce']:.4f}): once recall is zero, every arm lands on
  the null and how local the substitutes were stops mattering.</p>
  <p>The arm as originally specified confirms why it needed replacing. At
  d={d_hi}, <code>ratio2x</code> and <code>uniform</code> agree to four
  decimal places on every column — the ball of radius 2x the k-th distance
  has swallowed the entire point set, exactly the concentration argument
  that governs the rest of this index's design. The rank-window arms were
  added for that reason: they pin recall at zero while varying locality
  independently, which is the separation the original design intended but
  could not achieve at these dimensions.</p>
</section>

<section>
  <div class="prose"><h2>The full record</h2></div>
  {tables}
</section>

<footer>
  <p>Generated by <code>examples/report_recall_ablation.py</code> from
  <code>{html.escape(source)}</code>. Reproduce with
  <code>python examples/bench_recall_ablation.py --seeds {seeds}</code>.
  Clark-Evans and separation are both toroidal L1; the CE null is the exact
  fixed-n expectation, not the Poisson asymptotic.</p>
</footer>
</div>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", nargs="+",
                    default=[os.path.join(OUT, "recall_ablation.json")])
    ap.add_argument("--out", default=os.path.join(OUT, "recall_ablation.html"))
    args = ap.parse_args()

    rows = [r for path in args.json for r in load(path)]
    os.makedirs(OUT, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(render(rows, ", ".join(os.path.basename(p)
                                        for p in args.json)))
    print(f"[wrote {args.out}]")
