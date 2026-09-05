"""verdict.json -> static HTML. BUILD.md section 5.

No LLM. No network. No JavaScript framework. No web fonts. The output is a single
self-contained file with inline CSS, and rendering the same VerdictReport twice
produces byte-identical HTML -- there is no timestamp, no random ordering and no
generated id anywhere in it.

Layout, in reading order:

  header          what was audited, and the verdict as a badge
  metric cards    observed Sharpe, deflated Sharpe, alpha t-stat
  gate list       one row per gate, with the observed-vs-threshold comparison
  why it failed   the rationale strings verbatim, as prose
  evidence panels reported, explicitly non-voting
  limitations     a full-bleed band at the bottom

The "why it failed" section is the product. Everything above it is orientation and
everything below it is disclosure; the prose is the thing a human actually reads
and the only part that tells them something they could act on.
"""

from __future__ import annotations

import html
from pathlib import Path

from null.contracts import GateResult, Verdict
from null.verdict.engine import VerdictReport

__all__ = ["render_html", "write_report"]

_STATE_STYLE = {
    "PASS": ("PASS", "pass", "✓"),
    "FAIL": ("FAIL", "fail", "✗"),
    # Deliberately neither a tick nor a cross. A reader must not be able to skim
    # this row and come away thinking the gate was cleared.
    "NOT_COMPUTABLE": ("NOT COMPUTABLE", "nc", "–"),
}

_CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#16181d;--muted:#5b6270;--line:#e3e5ea;
--pass:#1a7f4b;--passbg:#e8f5ee;--fail:#b3261e;--failbg:#fdeceb;
--nc:#8a6d1f;--ncbg:#fbf3dd;--band:#3a2f14;--bandink:#f6efdd}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:820px;margin:0 auto;padding:40px 24px 0}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
margin:38px 0 12px;font-weight:600}
.sub{color:var(--muted);font-size:13px;margin:0}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.badge{display:inline-block;padding:6px 14px;border-radius:4px;font-weight:700;
letter-spacing:.06em;font-size:13px}
.badge.reject{background:var(--failbg);color:var(--fail)}
.badge.pass{background:var(--passbg);color:var(--pass)}
.head{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;
padding-bottom:22px;border-bottom:1px solid var(--line)}
.cards{display:flex;gap:12px;margin-top:22px}
.card{flex:1;background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:14px 16px}
.card .k{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.card .v{font-size:26px;font-weight:650;margin-top:4px;letter-spacing:-.02em}
.gate{display:flex;align-items:baseline;gap:10px;background:var(--card);
border:1px solid var(--line);border-radius:6px;padding:11px 14px;margin-bottom:7px}
.gate .icon{width:18px;font-weight:700}
.gate .name{flex:1;font-weight:600}
.gate .cmp{color:var(--muted);font-size:13px;white-space:nowrap}
.gate.pass .icon{color:var(--pass)} .gate.fail .icon{color:var(--fail)}
.gate.nc{border-color:#e6d9a8;background:var(--ncbg)}
.gate.nc .icon{color:var(--nc)} .gate.nc .name{color:var(--nc)}
.tag{font-size:10px;letter-spacing:.06em;font-weight:700;color:var(--nc);
border:1px solid #e0cf94;border-radius:3px;padding:1px 5px;margin-left:8px}
.why p{background:var(--card);border-left:3px solid var(--line);border-radius:0 4px 4px 0;
margin:0 0 10px;padding:13px 16px}
.why p.fail{border-left-color:var(--fail)}
.why p.nc{border-left-color:var(--nc);background:var(--ncbg)}
.why p .g{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.07em;
color:var(--muted);margin-bottom:5px;font-weight:700}
.panel{background:var(--card);border:1px dashed var(--line);border-radius:6px;
padding:13px 16px;margin-bottom:8px;color:#33383f;font-size:14px}
.panel .g{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.07em;
color:var(--muted);margin-bottom:5px;font-weight:700}
.note{color:var(--muted);font-size:13px;margin:-4px 0 12px}
.band{background:var(--band);color:var(--bandink);margin-top:46px;padding:30px 24px 40px}
.band .inner{max-width:820px;margin:0 auto}
.band h2{color:#e5d9b4;margin-top:0}
.band ul{margin:0;padding-left:18px}
.band li{margin-bottom:10px}
.band .sev{font-size:10px;letter-spacing:.07em;font-weight:700;border:1px solid #7d6c3d;
border-radius:3px;padding:1px 5px;margin-right:7px;text-transform:uppercase}
"""


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return f"{value:.4g}"


def _comparison(gate: GateResult) -> str:
    if gate.state == "NOT_COMPUTABLE":
        return "no evidence"
    return f"{_fmt(gate.observed)} vs {_fmt(gate.threshold)}"


def _gate_rows(verdict: Verdict) -> str:
    rows = []
    for gate in verdict.gates:
        label, cls, icon = _STATE_STYLE[gate.state]
        tag = '<span class="tag">NOT COMPUTABLE</span>' if cls == "nc" else ""
        rows.append(
            f'<div class="gate {cls}"><span class="icon">{icon}</span>'
            f'<span class="name">{_esc(gate.name)}{tag}</span>'
            f'<span class="cmp mono">{_esc(_comparison(gate))}</span></div>'
        )
    return "\n".join(rows)


def _why(verdict: Verdict, panels: dict[str, str]) -> str:
    blocks = []
    for gate in verdict.gates:
        if gate.state == "PASS":
            continue
        cls = "nc" if gate.state == "NOT_COMPUTABLE" else "fail"
        blocks.append(
            f'<p class="{cls}"><span class="g">{_esc(gate.name)}</span>'
            f"{_esc(gate.rationale)}</p>"
        )
    sentence = panels.get("expected_max_sharpe")
    if sentence:
        blocks.append(
            '<p><span class="g">selection diagnostic</span>' f"{_esc(sentence)}</p>"
        )
    if not blocks:
        blocks.append("<p>Every gate passed. Nothing failed to explain.</p>")
    return "\n".join(blocks)


def _panels(panels: dict[str, str]) -> str:
    return "\n".join(
        f'<div class="panel"><span class="g">{_esc(name)}</span>{_esc(text)}</div>'
        for name, text in sorted(panels.items())
        if name != "expected_max_sharpe" and text
    )


def _limitations(report: VerdictReport) -> str:
    if not report.limitations:
        return "<li>No limitations registered. This is itself suspicious.</li>"
    return "\n".join(
        f'<li><span class="sev">{_esc(lim.severity)}</span>{_esc(lim.text)}</li>'
        for lim in report.limitations
    )


def render_html(
    report: VerdictReport,
    *,
    observed_sharpe: float,
    deflated_sharpe: float,
    alpha_tstat: float,
    n_observations: int,
) -> str:
    """Render to a single self-contained HTML string. Deterministic."""
    verdict = report.verdict
    run = verdict.generated_from
    badge = "pass" if verdict.result == "PASS" else "reject"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NULL verdict &mdash; {_esc(run.strategy_id)}</title>
<style>{_CSS}</style></head><body>
<div class="wrap">
  <div class="head">
    <div>
      <h1>{_esc(run.strategy_id)}</h1>
      <p class="sub mono">{run.n_trials:,} trials &middot; {n_observations:,}
      observations &middot; evidence {_esc(verdict.evidence_hash[:12])} &middot;
      spec {_esc(verdict.spec_version)}</p>
    </div>
    <div class="badge {badge}">{_esc(verdict.result)}</div>
  </div>

  <div class="cards">
    <div class="card"><div class="k">Observed Sharpe</div>
      <div class="v mono">{observed_sharpe:.2f}</div></div>
    <div class="card"><div class="k">Deflated Sharpe</div>
      <div class="v mono">{deflated_sharpe:.2f}</div></div>
    <div class="card"><div class="k">Alpha t-stat</div>
      <div class="v mono">{alpha_tstat:.2f}</div></div>
  </div>

  <h2>Gates</h2>
{_gate_rows(verdict)}

  <h2>Why it failed</h2>
{_why(verdict, report.panels)}

  <h2>Evidence panels</h2>
  <p class="note">These are reported for context and <strong>do not vote</strong>
  on the verdict.</p>
{_panels(report.panels)}
</div>

<div class="band"><div class="inner">
  <h2>Stated limitations &mdash; all of them</h2>
  <ul>
{_limitations(report)}
  </ul>
</div></div>
</body></html>
"""


def write_report(
    report: VerdictReport,
    path: Path,
    *,
    observed_sharpe: float,
    deflated_sharpe: float,
    alpha_tstat: float,
    n_observations: int,
) -> Path:
    """Write the rendered report. Returns the path written."""
    html_text = render_html(
        report,
        observed_sharpe=observed_sharpe,
        deflated_sharpe=deflated_sharpe,
        alpha_tstat=alpha_tstat,
        n_observations=n_observations,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8", newline="\n")
    return path
