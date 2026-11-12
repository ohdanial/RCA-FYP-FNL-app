"""
app.py - Streamlit dashboard for the FYP2 project
"Real-Time Root Cause Analysis: Using LLMs to Interpret Cloud Security Log Anomalies"

Run in the VS Code terminal (inside the project folder, with .venv active):
    streamlit run app.py

Pages:
  1. Incidents   - list with search/filters + detail view (context, Gemini result, latency)
  2. Evaluation  - results of the 44-incident gold set (with vs without context)
  3. About       - pipeline, scope and safeguards

Safety: the dashboard never writes to llm_cache.json, gold files or scoring files.
Live "Analyse now" answers are saved to llm_cache_live.json only.
"""
import html
import math
import os
import time

import altair as alt
import pandas as pd
import streamlit as st

import rca_core as core

st.set_page_config(page_title="RCA Console", layout="wide", initial_sidebar_state="expanded")

ALERT = {"malicious", "suspicious"}
VARIANT_NAME = {"with_context": "With context", "no_context": "Without context"}
ACCENT = "#3b82f6"
VARIANT_COLOR = {"With context": ACCENT, "Without context": "#5b6270"}   # baseline in neutral grey
DOT = "#2a2f37"

# ======================================================================== icons (inline SVG, line style)
_P = {
    "shield": '<path d="M12 3l7 3v5c0 4.5-3 8.3-7 10-4-1.7-7-5.5-7-10V6l7-3z"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="0.5"/>',
    "alert": '<path d="M12 4l9 16H3L12 4z"/><path d="M12 10v4"/><path d="M12 17.5v.5"/>',
    "clock": '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
    "check": '<circle cx="12" cy="12" r="8.5"/><path d="M8.5 12.2l2.4 2.4 4.6-5"/>',
    "list": '<path d="M9 6h11M9 12h11M9 18h11"/><circle cx="4.5" cy="6" r=".6"/><circle cx="4.5" cy="12" r=".6"/><circle cx="4.5" cy="18" r=".6"/>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2.5v3M15 2.5v3M9 18.5v3M15 18.5v3M2.5 9h3M2.5 15h3M18.5 9h3M18.5 15h3"/>',
}


def icon(name, color):
    return (f'<span class="ico" style="color:{color};background:{color}1f;border-color:{color}33">'
            f'<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{_P[name]}</svg></span>')


# ======================================================================== styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --bg: #0b0d10; --card: #13161b; --card-2: #181c22; --line: #23272f; --line-2: #1c2027;
  --ink: #e6e8ec; --ink-2: #b4b9c3; --muted: #7a818d; --accent: #3b82f6;
  --mal: #f87171; --sus: #fbbf24; --ben: #4ade80; --up: #4ade80; --down: #f87171;
}
html, body, [class*="css"], .stMarkdown, button, input, select, textarea, label {
  font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif !important; }
code, pre, .mono { font-family: 'JetBrains Mono', Consolas, monospace !important; }

header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stMainMenu"], [data-testid="stAppDeployButton"], [data-testid="stToolbarActions"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: flex !important; visibility: visible !important; color: var(--ink-2); }
.block-container { padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1320px; }

/* ---------- sidebar */
section[data-testid="stSidebar"] { background: #0e1014; border-right: 1px solid var(--line-2); }
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
.brand { display: flex; align-items: center; gap: 10px; margin: .2rem 0 1.6rem 0; }
.brand .logo { width: 32px; height: 32px; border-radius: 9px; background: var(--accent); display: grid; place-items: center; }
.brand .name { font-weight: 700; font-size: .95rem; color: var(--ink); line-height: 1.1; }
.brand .sub { font-size: .7rem; color: var(--muted); }
.side-label { font-size: .66rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
  color: var(--muted); margin: 1.5rem 0 .5rem .1rem; }
/* radio as menu */
section[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; width: 100%; }
section[data-testid="stSidebar"] [role="radiogroup"] > div { width: 100%; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] {
  width: 100%; padding: .5rem .7rem; border-radius: 8px; margin: 0; transition: background .15s; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover { background: #171a20; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) { display: none; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] p { font-size: .87rem; color: var(--ink-2); }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] p span[role="img"] { margin-right: 8px; font-size: 1.1rem; color: var(--muted); }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] {
  background: rgba(59,130,246,.13); box-shadow: inset 2px 0 0 var(--accent); }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] p { color: #fff; font-weight: 600; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] p span[role="img"] { color: var(--accent); }
.side-foot { font-size: .72rem; color: var(--muted); margin-top: 2rem; line-height: 1.5; }

/* ---------- page header */
.eyebrow { font-size: .68rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
.page-title { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.025em; color: var(--ink); margin: .1rem 0 .15rem 0; }
.page-sub { font-size: .85rem; color: var(--muted); max-width: 760px; }

/* ---------- cards */
.card, div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .cardmark),
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--card); border: 1px solid var(--line) !important; border-radius: 14px; }
.card { padding: 1rem 1.15rem; }
.card-title { display: flex; justify-content: space-between; align-items: center; font-size: .9rem; font-weight: 600;
  color: var(--ink); margin-bottom: .6rem; }
.card-title .hint { font-size: .72rem; font-weight: 500; color: var(--muted); }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 1.2rem 0 12px 0; }
.kpi .top { display: flex; justify-content: space-between; align-items: center; }
.kpi .kl { font-size: .76rem; color: var(--ink-2); font-weight: 500; }
.kpi .value { font-size: 1.7rem; font-weight: 600; letter-spacing: -0.03em; color: var(--ink); margin-top: .55rem;
  font-variant-numeric: tabular-nums; }
.kpi .sub { font-size: .74rem; color: var(--muted); margin-top: .2rem; }
.up { color: var(--up); font-weight: 600; } .down { color: var(--down); font-weight: 600; } .flat { color: var(--ink-2); font-weight: 600; }
.ico { width: 28px; height: 28px; border-radius: 50%; display: inline-grid; place-items: center; border: 1px solid; }

/* ---------- gauge legend */
.glegend { padding-bottom: .4rem; display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px 14px; margin-top: .4rem; }
.glegend .k { font-size: .72rem; color: var(--muted); display: flex; align-items: center; gap: 6px; }
.glegend .k i { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.glegend .v { font-size: .95rem; font-weight: 600; color: var(--ink); margin-top: 2px; font-variant-numeric: tabular-nums; }

/* ---------- pills */
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
  font-size: .72rem; font-weight: 600; border: 1px solid transparent; white-space: nowrap; }
.pill .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.pill.malicious { color: var(--mal); background: rgba(248,113,113,.1); border-color: rgba(248,113,113,.28); }
.pill.suspicious { color: var(--sus); background: rgba(251,191,36,.1); border-color: rgba(251,191,36,.28); }
.pill.benign, .pill.pass { color: var(--ben); background: rgba(74,222,128,.1); border-color: rgba(74,222,128,.28); }
.pill.fail { color: var(--mal); background: rgba(248,113,113,.1); border-color: rgba(248,113,113,.28); }
.pill.neutral { color: var(--ink-2); background: #1d2128; border-color: var(--line); }

/* ---------- incident facts */
.facts { display: grid; grid-template-columns: repeat(4, 1fr); }
.fact { padding-right: 1rem; } .fact + .fact { border-left: 1px solid var(--line); padding-left: 1rem; }
.fact .k { font-size: .72rem; color: var(--muted); } .fact .v { font-size: .95rem; font-weight: 600; color: var(--ink); margin-top: 3px; }
.idline { font-size: .8rem; color: var(--muted); margin-top: 1rem; line-height: 2; }
.idline .mono { background: #1b1f26; border: 1px solid var(--line); padding: 2px 7px; border-radius: 6px; color: var(--ink-2); font-size: .76rem; }

/* ---------- result */
.result { background: var(--card); border: 1px solid var(--line); border-radius: 14px; overflow: hidden; margin: .6rem 0 1rem 0; }
.result-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: .85rem 1.2rem;
  border-bottom: 1px solid var(--line); }
.result-head .title { font-weight: 600; font-size: .9rem; color: var(--ink); }
.result-head .meta { display: flex; gap: 8px; }
.notice { padding: .55rem 1.2rem; font-size: .76rem; color: #fcd34d; background: rgba(251,191,36,.06);
  border-bottom: 1px solid rgba(251,191,36,.18); }
.result-body { padding: 1.1rem 1.2rem 1.2rem 1.2rem; }
.stats { display: grid; grid-template-columns: repeat(3, 1fr); margin-bottom: 1rem; }
.stat { padding-right: 1rem; } .stat + .stat { border-left: 1px solid var(--line); padding-left: 1rem; }
.stat .k { font-size: .72rem; color: var(--muted); }
.stat .v { font-size: 1.15rem; font-weight: 600; color: var(--ink); margin-top: 2px; font-variant-numeric: tabular-nums; }
.bar { height: 4px; background: #232832; border-radius: 2px; margin-top: 7px; }
.bar > div { height: 4px; background: var(--accent); border-radius: 2px; }
.label { font-size: .68rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin: 1rem 0 .4rem 0; }
.rc { font-size: .98rem; line-height: 1.65; color: var(--ink); }
ol.ev { margin: 0; padding-left: 1.15rem; }
ol.ev li { font-size: .86rem; color: var(--ink-2); line-height: 1.55; padding: 2px 0; }
ol.ev li::marker { color: var(--muted); }
.action { border-left: 2px solid var(--accent); padding: .55rem .9rem; background: rgba(59,130,246,.07);
  border-radius: 0 8px 8px 0; font-size: .88rem; color: var(--ink); }
.latency { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: .73rem; color: var(--muted); padding: .65rem 1.2rem;
  border-top: 1px solid var(--line); background: #11141a; font-variant-numeric: tabular-nums; }
.latency b { color: var(--ink); font-weight: 600; }

.label-card { background: var(--card); border: 1px solid var(--line); border-left: 2px solid var(--ink-2);
  border-radius: 0 14px 14px 0; padding: .85rem 1.15rem; margin: 12px 0; }
.label-card .t { display: flex; gap: 10px; align-items: center; font-size: .8rem; font-weight: 600; color: var(--ink-2); }
.label-card .d { font-size: .9rem; color: var(--ink); margin-top: .4rem; line-height: 1.55; }
.label-card .s { font-size: .74rem; color: var(--muted); margin-top: .45rem; }
.empty { border: 1px dashed var(--line); border-radius: 12px; padding: 1.1rem; color: var(--muted); font-size: .86rem; }
.note { font-size: .76rem; color: var(--muted); margin-top: .3rem; }
.section-title { font-size: 1rem; font-weight: 600; color: var(--ink); margin: 1.8rem 0 .7rem 0; }

/* ---------- widgets */
.stTabs [data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: .6rem 0; font-size: .86rem; color: var(--muted); background: transparent; }
.stTabs [aria-selected="true"] { color: var(--ink) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent); height: 2px; }
.stTabs [data-baseweb="tab-border"] { display: none; }
.stButton > button[kind="primary"] { background: var(--accent); border: none; border-radius: 10px; font-weight: 600; }
.stButton > button[kind="primary"]:hover { background: #2563eb; }
.stTextInput input { border-radius: 999px !important; }
div[data-baseweb="input"], div[data-baseweb="select"] > div { background: #13161b !important; border-color: var(--line) !important; }
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
[data-testid="stExpander"] details { border: 1px solid var(--line); border-radius: 12px; background: var(--card); }
[data-testid="stMultiSelectTagsContainer"] > span { background-color: #1b2533 !important; border: 1px solid #263346 !important; border-radius: 6px !important; }
[data-testid="stMultiSelectTagsContainer"] > span * { color: #a9c7f5 !important; }
pre { background: #0f1216 !important; border: 1px solid var(--line); border-radius: 10px; font-size: .74rem !important; }
</style>
"""

esc = html.escape


def ui(markup, where=st):
    where.markdown(markup, unsafe_allow_html=True)


def pill(verdict):
    v = verdict if verdict in ("malicious", "suspicious", "benign") else "neutral"
    text = verdict.capitalize() if v != "neutral" else "Not analysed"
    return f'<span class="pill {v}"><span class="dot"></span>{text}</span>'


def tag(text, kind="neutral"):
    return f'<span class="pill {kind}">{esc(text)}</span>'


def page_header(eyebrow, title, sub=""):
    ui(f'<div class="eyebrow">{esc(eyebrow)}</div><div class="page-title">{esc(title)}</div>'
       + (f'<div class="page-sub">{sub}</div>' if sub else ""))


def kpi_card(label, value, sub, ico, color):
    return (f'<div class="card kpi"><div class="top"><div class="kl">{esc(label)}</div>{icon(ico, color)}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')


def delta(new, old, fmt, higher_is_better=True, unit="", tol=1e-9):
    d = new - old
    if abs(d) < tol:
        return '<span class="flat">≈ same</span>'
    good = (d > 0) == higher_is_better
    arrow = "↑" if d > 0 else "↓"
    return f'<span class="{"up" if good else "down"}">{arrow} {fmt.format(abs(d))}{unit}</span>'


def gauge(share, big, caption):
    """Semicircle gauge (SVG)."""
    L = math.pi * 80
    return f"""
<svg viewBox="0 0 200 118" width="100%" style="max-width:250px;display:block;margin:0 auto">
  <defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#1d4ed8"/><stop offset="1" stop-color="#60a5fa"/></linearGradient></defs>
  <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#232832" stroke-width="12" stroke-linecap="round"/>
  <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="url(#g)" stroke-width="12" stroke-linecap="round"
        stroke-dasharray="{L * share:.1f} {L:.1f}"/>
  <text x="100" y="86" text-anchor="middle" fill="#e6e8ec" font-size="26" font-weight="700" font-family="Inter">{big}</text>
  <text x="100" y="106" text-anchor="middle" fill="#7a818d" font-size="10.5" font-family="Inter">{esc(caption)}</text>
</svg>"""


def glegend(items):
    cells = "".join(f'<div><div class="k"><i style="background:{c}"></i>{esc(k)}</div><div class="v">{v}</div></div>'
                    for k, v, c in items)
    return f'<div class="glegend">{cells}</div>'


# ======================================================================== charts
def dark(chart, height=280):
    return (chart.properties(height=height, background="transparent")
            .configure(font="Inter")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor="#7a818d", titleColor="#7a818d", labelFontSize=11, titleFontSize=11,
                            domainColor="#23272f", tickColor="#23272f", gridColor="#1c2027", gridDash=[3, 3])
            .configure_legend(labelColor="#b4b9c3", labelFontSize=11.5, symbolType="circle", symbolSize=70,
                              orient="top-right", direction="horizontal", title=None))


def grouped_bars(data, x, y, y_title, percent=True, sort=None, height=280):
    fmt = ".0%" if percent else ".2f"
    c = (alt.Chart(data)
         .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width={"band": 0.36})
         .encode(
             x=alt.X(f"{x}:N", sort=sort, title=None, axis=alt.Axis(labelAngle=0, labelLimit=170, ticks=False, domain=False, labelOverlap=False)),
             xOffset=alt.XOffset("Variant:N", sort=list(VARIANT_COLOR)),
             y=alt.Y(f"{y}:Q", title=y_title, axis=alt.Axis(format=fmt, tickCount=5, domain=False, ticks=False),
                     scale=alt.Scale(domain=[0, 1] if percent else [0, 2])),
             color=alt.Color("Variant:N", scale=alt.Scale(domain=list(VARIANT_COLOR), range=list(VARIANT_COLOR.values()))),
             tooltip=[alt.Tooltip(f"{x}:N", title="Measure"), alt.Tooltip("Variant:N"),
                      alt.Tooltip(f"{y}:Q", title="Value", format=fmt)]))
    return dark(c, height)


def dot_matrix(hours, highlight_hour=None, max_rows=12, height=230):
    """Incidents per month, drawn as stacked dots (one column per month)."""
    months = pd.to_datetime(pd.Series(hours)).dt.tz_localize(None).dt.to_period("M")
    counts = months.value_counts().sort_index()
    full = pd.period_range(counts.index.min(), counts.index.max(), freq="M")
    counts = counts.reindex(full, fill_value=0)
    unit = max(1, math.ceil(counts.max() / max_rows))
    hl = pd.Timestamp(highlight_hour).tz_localize(None).to_period("M") if highlight_hour is not None else None
    rows = []
    for p, n in counts.items():
        label = p.strftime("%b %Y")
        for lvl in range(math.ceil(n / unit)):
            rows.append({"Month": label, "order": p.ordinal, "level": lvl, "Incidents": int(n), "hl": p == hl})
    data = pd.DataFrame(rows)
    every = max(1, len(full) // 8)
    keep = [p.strftime("%b %Y") for i, p in enumerate(full) if i % every == 0]
    c = (alt.Chart(data).mark_circle(size=44, opacity=1)
         .encode(x=alt.X("Month:O", sort=alt.SortField("order"), title=None,
                         axis=alt.Axis(values=keep, labelAngle=0, ticks=False, domain=False, labelFontSize=10)),
                 y=alt.Y("level:O", sort="descending", axis=None),
                 color=alt.condition("datum.hl", alt.value(ACCENT), alt.value("#3a404a")),
                 tooltip=["Month", "Incidents"]))
    return dark(c, height), unit


# ======================================================================== data
@st.cache_data
def load_light_data():
    gold = pd.read_parquet("gold_labelled.parquet")
    final = pd.read_csv("final_evaluation.csv")
    inc = pd.read_parquet(core.INCIDENTS_FILE)
    inc = core.anonymise(inc, core.load_arn_map())
    inc["inc_id"] = "I" + inc.index.astype(str).str.zfill(4)
    inc["gold_id"] = inc.index.map(gold.set_index("inc_index")["gold_id"])
    return gold, final, inc


@st.cache_resource(show_spinner="Loading 1.2 million events for live analysis (first time only)...")
def load_events_cached():
    df, _ = core.load_events()
    return df


# ======================================================================== result card
def show_result(entry, title, source_note):
    t_start = time.perf_counter()
    r = entry["result"]
    conf = float(r["confidence"])
    evidence = "".join(f"<li>{esc(e)}</li>" for e in r["evidence"])
    body = f"""
<div class="result">
  <div class="result-head"><div class="title">{esc(title)}</div>
    <div class="meta">{pill(r['verdict'])}{tag('Severity ' + r['severity'])}</div></div>
  <div class="notice"><b>AI-suggested</b> &middot; verify before acting. This is the model's reading of the log evidence, not a confirmed finding.</div>
  <div class="result-body">
    <div class="stats">
      <div class="stat"><div class="k">Confidence</div><div class="v">{conf:.2f}</div><div class="bar"><div style="width:{conf * 100:.0f}%"></div></div></div>
      <div class="stat"><div class="k">Context build</div><div class="v">{entry['context_seconds']:.2f} s</div></div>
      <div class="stat"><div class="k">LLM response</div><div class="v">{entry['llm_seconds']:.2f} s</div></div>
    </div>
    <div class="label">Root cause</div><div class="rc">{esc(r['root_cause'])}</div>
    <div class="label">Evidence from the log</div><ol class="ev">{evidence}</ol>
    <div class="label">Recommended next action</div><div class="action">{esc(r['next_action'])}</div>
  </div>"""
    display_s = time.perf_counter() - t_start
    total = entry["context_seconds"] + entry["llm_seconds"] + display_s
    body += f"""
  <div class="latency"><span>Total <b>{total:.2f} s</b></span>
    <span>context {entry['context_seconds']:.3f} s + LLM {entry['llm_seconds']:.3f} s + display {display_s:.3f} s</span>
    <span>{esc(source_note)}</span><span>{esc(entry['model'])} &middot; {esc(entry.get('created', ''))}</span></div>
</div>"""
    ui(body)


# ======================================================================== page 1: incidents
def page_incidents(gold, final, inc, cache, live_cache):
    h1, h2 = st.columns([3, 2], vertical_alignment="bottom")
    with h1:
        page_header("Investigation", "Flagged incidents",
                    "Unusual CloudTrail activity flagged by Isolation Forest, grouped by identity, IP and hour. "
                    "Identities appear exactly as the model saw them.")
    search = h2.text_input("Search", placeholder="Search ID, IP or identity...", label_visibility="collapsed")

    ui('<div class="side-label">Incident list</div>', st.sidebar)
    source = st.sidebar.radio("Incident list", ["Gold set (44 labelled)", "All flagged incidents"],
                              label_visibility="collapsed")
    is_gold = source.startswith("Gold")
    if is_gold:
        rows = gold.copy()
        rows["id"] = rows["gold_id"]
    else:
        show_noise = st.sidebar.checkbox("Include routine Lambda logging", value=False)
        rows = (inc if show_noise else inc[~inc["is_routine_noise"]]).copy()
        rows["id"] = rows["gold_id"].fillna(rows["inc_id"])

    def cached(row):
        k = core.cache_key(row, "with_context")
        return cache.get(k) or live_cache.get(k)

    entries = [cached(r) for _, r in rows.iterrows()]
    rows["gemini_verdict"] = [e["result"]["verdict"] if e else "not analysed" for e in entries]
    rows["severity"] = [e["result"]["severity"] if e else "" for e in entries]
    rows["confidence"] = [e["result"]["confidence"] if e else None for e in entries]
    rows["seconds"] = [e["context_seconds"] + e["llm_seconds"] if e else None for e in entries]

    ui('<div class="side-label">Filters</div>', st.sidebar)
    cats = sorted(rows["label_category"].unique())
    pick_cats = st.sidebar.multiselect("Category", cats, default=cats)
    verdicts = ["malicious", "suspicious", "benign", "not analysed"]
    pick_verdicts = st.sidebar.multiselect("Model verdict (with context)", verdicts,
                                           default=[v for v in verdicts if v in set(rows["gemini_verdict"])])
    view = rows[rows["label_category"].isin(pick_cats) & rows["gemini_verdict"].isin(pick_verdicts)]
    if is_gold and st.sidebar.checkbox("Only alert errors"):
        view = view[view["gemini_verdict"].isin(ALERT) != view["true_verdict"].isin(ALERT)]
    if search:
        s = search.lower()
        view = view[view["id"].str.lower().str.contains(s, regex=False)
                    | view["sourceIPAddress"].str.lower().str.contains(s, regex=False)
                    | view["userIdentityarn"].str.lower().str.contains(s, regex=False)]

    # ---- KPI row
    vc = view["gemini_verdict"].value_counts()
    n_done = int((view["gemini_verdict"] != "not analysed").sum())
    n_alert = int(view["gemini_verdict"].isin(ALERT).sum())
    mean_s = view["seconds"].mean()
    ui('<div class="kpis">'
       + kpi_card("Incidents", f"{len(view)}", f"of {len(rows)} in this list", "list", ACCENT)
       + kpi_card("Analysed by model", f"{n_done}", "answers saved in cache", "cpu", "#a78bfa")
       + kpi_card("Raised as alert", f"{n_alert}", "malicious or suspicious", "alert", "#fbbf24")
       + kpi_card("Avg response time", f"{mean_s:.2f} s" if pd.notna(mean_s) else "–",
                  "context + LLM, per incident", "clock", "#4ade80")
       + "</div>")

    if view.empty:
        ui('<div class="empty">No incidents match the current search and filters.</div>')
        return

    # ---- detail selector first (so the timeline can highlight it)
    labels = [f"{r['id']}  ·  {r['label_category']}  ·  {r['userIdentityarn'].split(':')[-1]}  ·  "
              f"{pd.Timestamp(r['hour']):%Y-%m-%d %H:%M}" for _, r in view.iterrows()]
    if st.session_state.get("pick_label") not in labels:   # first load, or the list changed
        st.session_state["pick_label"] = labels[0]
    incident = view.iloc[labels.index(st.session_state["pick_label"])]

    # ---- timeline + gauge
    c1, c2 = st.columns([2, 1], gap="small")
    with c1.container(border=True):
        chart, unit = dot_matrix(view["hour"], incident["hour"])
        ui(f'<div class="card-title">Incidents over time<span class="hint">per month &middot; '
           f'{"each dot = " + str(unit) + " incidents" if unit > 1 else "each dot = 1 incident"} &middot; '
           f'<span style="color:{ACCENT}">&#9679;</span> selected incident</span></div>')
        st.altair_chart(chart, width="stretch")
    with c2.container(border=True):
        share = n_alert / n_done if n_done else 0
        ui(f'<div class="card-title">Model verdicts<span class="hint">with context</span></div>'
           + gauge(share, f"{share:.0%}", "raised as alert")
           + glegend([("Malicious", vc.get("malicious", 0), "#f87171"),
                      ("Suspicious", vc.get("suspicious", 0), "#fbbf24"),
                      ("Benign", vc.get("benign", 0), "#4ade80"),
                      ("Not analysed", vc.get("not analysed", 0), "#5b6270")]))

    # ---- table
    with st.container(border=True):
        ui(f'<div class="card-title">Incident list<span class="hint">{len(view)} shown</span></div>')
        cols = {"id": "ID", "userIdentityarn": "Identity", "sourceIPAddress": "Source IP", "hour": "Hour (UTC)",
                "n_events": "Events", "label_category": "Category", "gemini_verdict": "Model verdict",
                "severity": "Severity", "confidence": "Confidence"}
        if is_gold:
            cols["true_verdict"] = "Analyst label"
        table = view[list(cols)].rename(columns=cols)
        table["Identity"] = table["Identity"].str.replace("arn:aws:", "", regex=False)
        table["Hour (UTC)"] = pd.to_datetime(table["Hour (UTC)"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(table, hide_index=True, width="stretch", height=300,
                     column_config={"Confidence": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})

    ui('<div class="section-title">Incident detail</div>')
    choice = st.selectbox("Open incident", labels, key="pick_label")
    show_detail(view.iloc[labels.index(choice)], final, cache, live_cache)


def show_detail(incident, final, cache, live_cache):
    ui(f"""<div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem">
        <div style="font-size:1.2rem;font-weight:700;letter-spacing:-.02em">Incident {esc(str(incident['id']))}</div>
        {tag(incident['label_category'])}</div>
      <div class="facts">
        <div class="fact"><div class="k">Identity type</div><div class="v">{esc(incident['identity_type'])}</div></div>
        <div class="fact"><div class="k">Flagged events</div><div class="v">{incident['n_events']} <span style="color:var(--muted);font-weight:500">({incident['n_distinct_events']} distinct)</span></div></div>
        <div class="fact"><div class="k">Most common event</div><div class="v">{esc(incident['top_event'])}</div></div>
        <div class="fact"><div class="k">Error rate</div><div class="v">{incident['error_rate']:.0%}</div></div>
      </div>
      <div class="idline">Identity <span class="mono">{esc(incident['userIdentityarn'])}</span> &nbsp;
        IP <span class="mono">{esc(incident['sourceIPAddress'])}</span> &nbsp;
        Hour <span class="mono">{pd.Timestamp(incident['hour']):%Y-%m-%d %H:%M} UTC</span><br>
        The category label is for evaluation only and is never sent to the model.</div>
    </div>""")
    with st.expander("Real identity (analyst view only, never sent to the model)"):
        st.code(incident["arn_real"], language=None)

    if isinstance(incident.get("gold_id"), str) and "true_verdict" in incident.index:
        sc = final[final["gold_id"] == incident["gold_id"]].set_index("variant")
        parts = [f"{VARIANT_NAME[v]}: verdict {'correct' if sc.loc[v, 'exact'] else 'wrong'}, "
                 f"root-cause score {int(sc.loc[v, 'rc_score'])}/2" for v in ["with_context", "no_context"] if v in sc.index]
        ui(f"""<div class="label-card"><div class="t">Analyst label (blind) {pill(incident['true_verdict'])}</div>
          <div class="d">{esc(incident['true_root_cause'])}</div>
          <div class="s">{' &nbsp;&middot;&nbsp; '.join(parts)}</div></div>""")

    tab1, tab2, tab3 = st.tabs(["With context", "Without context (baseline)", "Input sent to the model"])
    for tab, variant in [(tab1, "with_context"), (tab2, "no_context")]:
        with tab:
            entry = cache.get(core.cache_key(incident, variant))
            if entry:
                show_result(entry, f"Model result, {VARIANT_NAME[variant].lower()}", "from cache, timed when created")
            else:
                ui('<div class="empty">No saved answer for this incident yet. Use <b>Analyse now</b> below for a live call.</div>')
            live = live_cache.get(core.cache_key(incident, variant))
            if live:
                show_result(live, "Latest live answer", "live call from this dashboard")
    with tab3:
        entry = cache.get(core.cache_key(incident, "with_context")) or live_cache.get(core.cache_key(incident, "with_context"))
        if entry:
            leaks = core.leak_check(entry["context"])
            ui(f'<div style="margin:.6rem 0">{tag("Leak check passed", "pass") if not leaks else tag("Leak check failed", "fail")} '
               f'<span class="note" style="margin-left:6px">{"no real names found in the text" if not leaks else "found " + esc(str(leaks))}</span></div>')
            st.code(core.INSTRUCTIONS + entry["context"], language=None)
        else:
            ui('<div class="empty">The context is built when the incident is analysed.</div>')
        base = cache.get(core.cache_key(incident, "no_context"))
        ui('<div class="label">Baseline input (without context)</div>')
        st.code(base["context"] if base else core.build_minimal(incident), language=None)

    ui('<div class="section-title">Analyse now</div>')
    with st.container(border=True):
        ui('<div class="note" style="margin:0 0 .7rem 0">Builds the context from the raw events and makes one live call '
           '(uses one free-tier request). Saved to <span class="mono">llm_cache_live.json</span> only; the research cache is never changed.</div>')
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        variant = c1.radio("Variant", ["with_context", "no_context"], format_func=VARIANT_NAME.get,
                           horizontal=True, label_visibility="collapsed")
        go = c2.button("Analyse now", type="primary", width="stretch")
    if go:
        key = st.session_state.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not key:
            st.error("Enter your Gemini API key in the sidebar first.")
            return
        try:
            df = load_events_cached() if variant == "with_context" else None
            with st.spinner("Building context and calling the model..."):
                entry = core.analyse_live(incident, df, core.make_client(key), variant)
        except Exception as e:
            st.error(f"Model call failed: {str(e)[:300]}. If it says 503 or high demand, wait a minute and try again.")
            return
        lc = core.load_cache(core.LIVE_CACHE_FILE)
        lc[core.cache_key(incident, variant)] = entry
        core.save_cache(lc, core.LIVE_CACHE_FILE)
        show_result(entry, "Live result", "live call just now")


# ======================================================================== page 2: evaluation
def metrics_by_variant(final):
    out = {}
    for v, d in final.groupby("variant"):
        mal, ben = d[d["true_verdict"] == "malicious"], d[d["true_verdict"] == "benign"]
        caught = int(mal["llm_verdict"].isin(ALERT).sum())
        fa = int(ben["llm_verdict"].isin(ALERT).sum())
        full = int((d["rc_score"] == 2).sum())
        out[VARIANT_NAME[v]] = {
            "table": {
                "Alert accuracy (malicious + suspicious = alert)": f"{d['alert_ok'].mean():.1%}  ({d['alert_ok'].sum()}/{len(d)})",
                "Exact verdict accuracy": f"{d['exact'].mean():.1%}  ({d['exact'].sum()}/{len(d)})",
                "Attacks caught (true malicious raised as alert)": f"{caught}/{len(mal)}  ({caught / len(mal):.0%})",
                "False alarms on benign": f"{fa}/{len(ben)}  ({fa / len(ben):.0%})",
                "Root-cause score, mean (0-2)": f"{d['rc_score'].mean():.2f}",
                "Root cause fully correct (score 2)": f"{full / len(d):.1%}  ({full}/{len(d)})",
                "Mean confidence": f"{d['confidence'].mean():.2f}",
                "Mean LLM time": f"{d['llm_seconds'].mean():.2f} s",
                "Mean total time (context + LLM)": f"{d['total_seconds'].mean():.2f} s",
            },
            "rates": {"Alert accuracy": d["alert_ok"].mean(), "Exact verdict": d["exact"].mean(),
                      "Attacks caught": caught / len(mal), "Root cause correct": full / len(d),
                      "False alarms": fa / len(ben)},
            "caught": caught, "n_mal": len(mal), "fa": fa, "n_ben": len(ben), "alert": d["alert_ok"].mean(),
            "alert_n": int(d["alert_ok"].sum()), "n": len(d), "full": full / len(d), "time": d["total_seconds"].mean(),
        }
    return out


def page_evaluation(final):
    page_header("Objective III", "Evaluation",
                f"44 gold incidents labelled blind before the run. Same model ({core.MODEL}), same frozen prompt, "
                "temperature 0. Baseline = the same model given only the detector alert, without context.")
    m = metrics_by_variant(final)
    w, n = m["With context"], m["Without context"]
    ui('<div class="kpis">'
       + kpi_card("Attacks caught", f"{w['caught']}/{w['n_mal']}",
                  f"{delta(w['caught'], n['caught'], '{:.0f}')} vs baseline {n['caught']}/{n['n_mal']}", "shield", ACCENT)
       + kpi_card("Alert accuracy", f"{w['alert']:.1%}",
                  f"{delta(w['alert'] * 100, n['alert'] * 100, '{:.1f}', unit=' pts')} vs baseline", "target", "#a78bfa")
       + kpi_card("Root cause fully correct", f"{w['full']:.1%}",
                  f"{delta(w['full'] * 100, n['full'] * 100, '{:.1f}', unit=' pts')} vs baseline", "check", "#4ade80")
       + kpi_card("Avg time per incident", f"{w['time']:.2f} s",
                  f"{delta(w['time'], n['time'], '{:.2f}', higher_is_better=False, unit=' s', tol=0.1)} as baseline ({n['time']:.2f} s)", "clock", "#fbbf24")
       + "</div>")

    c1, c2 = st.columns([2, 1], gap="small")
    with c1.container(border=True):
        ui('<div class="card-title">Key measures<span class="hint">higher is better, except false alarms</span></div>')
        rates = pd.DataFrame([{"Measure": k, "Variant": v, "Rate": r}
                              for v in VARIANT_COLOR for k, r in m[v]["rates"].items()])
        st.altair_chart(grouped_bars(rates, "Measure", "Rate", None, sort=list(w["rates"]), height=300), width="stretch")
    with c2.container(border=True):
        ui('<div class="card-title">Alert accuracy<span class="hint">with context</span></div>'
           + gauge(w["alert"], f"{w['alert']:.1%}", f"{w['alert_n']} of {w['n']} incidents correct")
           + glegend([("Attacks caught", f"{w['caught']}/{w['n_mal']}", ACCENT),
                      ("Attacks missed", f"{w['n_mal'] - w['caught']}/{w['n_mal']}", "#f87171"),
                      ("False alarms", f"{w['fa']}/{w['n_ben']}", "#fbbf24"),
                      ("Benign cleared", f"{w['n_ben'] - w['fa']}/{w['n_ben']}", "#4ade80")]))

    with st.container(border=True):
        ui('<div class="card-title">Full results</div>')
        tbl = pd.DataFrame({k: v["table"] for k, v in m.items()})[list(VARIANT_COLOR)]
        tbl.index.name = "Measure"
        st.dataframe(tbl, width="stretch")

    order = ["benign", "suspicious", "malicious"]
    cc = st.columns(2, gap="small")
    for col, v in zip(cc, ["with_context", "no_context"]):
        with col.container(border=True):
            d = final[final["variant"] == v]
            cm = pd.crosstab(d["true_verdict"], d["llm_verdict"]).reindex(index=order, columns=order, fill_value=0)
            cm.index.name, cm.columns.name = "Analyst", None
            ui(f'<div class="card-title">Confusion matrix, {VARIANT_NAME[v].lower()}'
               f'<span class="hint">rows = analyst, columns = model</span></div>')
            st.dataframe(cm, width="stretch")
    ui('<div class="note">With context, 13 of 19 attacks were labelled "suspicious" rather than "malicious". They were '
       'still raised as alerts, so this is a severity calibration issue, not a missed detection.</div>')

    with st.container(border=True):
        ui('<div class="card-title">By incident category</div>')
        cat = final.groupby(["label_category", "variant"]).agg(exact=("exact", "mean"), rc=("rc_score", "mean")).reset_index()
        cat["Variant"] = cat["variant"].map(VARIANT_NAME)
        t1, t2 = st.tabs(["Exact verdict accuracy", "Root-cause score (0-2)"])
        with t1:
            st.altair_chart(grouped_bars(cat, "label_category", "exact", None), width="stretch")
        with t2:
            st.altair_chart(grouped_bars(cat, "label_category", "rc", None, percent=False), width="stretch")

    with st.container(border=True):
        ui('<div class="card-title">Response time (seconds)<span class="hint">total = context building + LLM call</span></div>')
        lat = final.groupby("variant")[["context_seconds", "llm_seconds", "total_seconds"]].agg(["mean", "median", "max"]).round(2)
        lat.index = lat.index.map(VARIANT_NAME)
        lat.columns = [f"{a.replace('_seconds', '').capitalize()} · {b}" for a, b in lat.columns]
        st.dataframe(lat, width="stretch")
    st.download_button("Download final_evaluation.csv", final.to_csv(index=False), "final_evaluation.csv", "text/csv")


# ======================================================================== page 3: about
def page_about():
    page_header("Project", "About this prototype",
                "Real-Time Root Cause Analysis (RCA): Using LLMs to Interpret Cloud Security Log Anomalies")
    steps = [("Logs", "AWS CloudTrail (flaws.cloud dataset), cleaned to 1.23M events with 17 features.", "list"),
             ("Detect", "Isolation Forest (200 trees, contamination 0.01) flags 12,256 unusual events.", "alert"),
             ("Group", "Flagged events grouped into 4,652 incidents by identity, IP and hour.", "target"),
             ("Context", "History, access method and related events in a ±1 hour window. Names anonymised.", "shield"),
             ("Explain", f"{core.MODEL} at temperature 0 returns verdict, root cause, severity, confidence, evidence and next action.", "cpu"),
             ("Review", "This dashboard shows the answer with measured latency, labelled AI-suggested.", "check")]
    cards = "".join(f'<div class="card"><div class="top" style="display:flex;justify-content:space-between;align-items:center">'
                    f'<div class="eyebrow">Step {i:02d}</div>{icon(ic, ACCENT)}</div>'
                    f'<div style="font-weight:600;margin-top:.5rem">{esc(t)}</div>'
                    f'<div style="font-size:.85rem;color:var(--ink-2);margin-top:.3rem;line-height:1.5">{esc(d)}</div></div>'
                    for i, (t, d, ic) in enumerate(steps, 1))
    ui(f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:1.2rem">{cards}</div>')
    c1, c2 = st.columns(2, gap="small")
    with c1.container(border=True):
        ui("""<div class="card-title">Scope</div><ol class="ev">
          <li>AWS CloudTrail only.</li>
          <li>"Real-time" means near-instant analysis of a flagged incident, not production streaming.</li>
          <li>No automated remediation. Every answer is labelled AI-suggested, verify before acting.</li></ol>""")
    with c2.container(border=True):
        ui("""<div class="card-title">Method safeguards</div><ol class="ev">
          <li>The true category label is never sent to the model.</li>
          <li>Account ID, user, role and session names are anonymised and leak-checked.</li>
          <li>Gold set labelled blind; prompt frozen on a separate practice set first.</li>
          <li>Every answer is cached with the exact input sent, so results are reproducible.</li></ol>""")


# ======================================================================== main
def main():
    ui(CSS)
    ui('<div class="brand"><div class="logo"><svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="#fff" '
       'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' + _P["shield"] + '</svg></div>'
       '<div><div class="name">RCA Console</div><div class="sub">CloudTrail root cause analysis</div></div></div>', st.sidebar)
    ui('<div class="side-label">Menu</div>', st.sidebar)
    page = st.sidebar.radio("Page", [":material/radar: Incidents", ":material/bar_chart: Evaluation",
                                     ":material/info: About"], label_visibility="collapsed")

    try:
        gold, final, inc = load_light_data()
    except FileNotFoundError as e:
        st.error(f"File not found: {e}. Run `streamlit run app.py` from the project folder.")
        return
    cache = core.load_cache(core.CACHE_FILE)
    live_cache = core.load_cache(core.LIVE_CACHE_FILE)

    if page.endswith("Incidents"):
        page_incidents(gold, final, inc, cache, live_cache)
    elif page.endswith("Evaluation"):
        page_evaluation(final)
    else:
        page_about()

    ui('<div class="side-label">Gemini API key</div>', st.sidebar)
    st.sidebar.text_input("API key", type="password", key="api_key", label_visibility="collapsed",
                          placeholder="Only needed for Analyse now",
                          help="Kept in this browser session only; never saved to a file.")


main()
