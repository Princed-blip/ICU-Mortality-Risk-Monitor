"""
Visual components for the ICU-monitor style interface (pure HTML/SVG/CSS strings, no Streamlit calls).
Everything is rendered with st.markdown(..., unsafe_allow_html=True); animations are CSS-only (no JavaScript).
"""
import html
import math
import re

GREEN, CYAN, YELLOW, RED, AMBER = "#19f58a", "#29d3ff", "#ffd23f", "#ff4d5e", "#ffb020"
MUTED, TEXT, PANEL, LINE = "#7f98a8", "#d6e6f0", "#0a131b", "#17303f"


def _compact(s):
    """Remove newlines/indentation so Streamlit's markdown parser keeps the HTML as one raw block."""
    s = re.sub(r"\s*\n\s*", " ", s)
    return re.sub(r">\s+<", "><", s).strip()


# --------------------------------------------------------------------------- CSS
def global_css():
    return _compact("""<style>
.stApp{background:radial-gradient(1100px 520px at 8% -8%,#0c2433 0%,#04080c 58%) fixed;}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:1.1rem;max-width:1320px;}
.mono{font-family:Consolas,"SF Mono",Menlo,"Roboto Mono",monospace;}
.icu-panel{background:linear-gradient(180deg,#0b151d,#070e14);border:1px solid #17303f;border-radius:14px;
  padding:14px 16px;box-shadow:0 10px 30px rgba(0,0,0,.45),inset 0 0 0 1px rgba(25,245,138,.04);}
.hero{display:flex;flex-wrap:wrap;gap:14px;align-items:center;justify-content:space-between;}
.hero h1{margin:0;font-family:Consolas,"SF Mono",Menlo,monospace;font-size:1.75rem;letter-spacing:.14em;color:#e9f6ff;font-weight:700;}
.hero .sub{color:#7f98a8;font-size:.9rem;margin-top:4px;letter-spacing:.02em;}
.chips{display:flex;flex-wrap:wrap;gap:8px;}
.chip{border:1px solid #1d3b4d;background:#0a1620;border-radius:999px;padding:5px 12px;font-size:.78rem;color:#bcd2df;}
.chip b{color:#19f58a;font-weight:600;}
.notice{border:1px solid #6b5313;background:linear-gradient(90deg,rgba(255,176,32,.10),rgba(255,176,32,.03));
  border-radius:10px;padding:9px 14px;color:#f1d9a0;font-size:.85rem;margin:10px 0 14px 0;}
.sec{display:flex;align-items:center;gap:10px;margin:18px 0 8px 0;font-family:Consolas,monospace;letter-spacing:.14em;
  font-size:.82rem;color:#9fc3d6;text-transform:uppercase;}
.sec:before{content:"";width:4px;height:16px;background:#19f58a;border-radius:2px;box-shadow:0 0 8px #19f58a;}
.sec i{flex:1;height:1px;background:linear-gradient(90deg,#17303f,transparent);}
/* monitor */
.mon{display:grid;grid-template-columns:minmax(0,1fr) 250px;gap:14px;}
@media (max-width:820px){.mon{grid-template-columns:1fr;}}
.monhead{display:flex;justify-content:space-between;align-items:center;font-family:Consolas,monospace;font-size:.72rem;
  letter-spacing:.12em;color:#7f98a8;margin-bottom:8px;gap:10px;flex-wrap:wrap;}
.live{display:inline-flex;align-items:center;gap:6px;color:#9fc3d6;}
.live:before{content:"";width:8px;height:8px;border-radius:50%;background:#19f58a;box-shadow:0 0 8px #19f58a;animation:blink 1.6s infinite;}
.trace{position:relative;height:86px;overflow:hidden;border-bottom:1px solid #112331;
  background-image:linear-gradient(rgba(70,130,160,.10) 1px,transparent 1px),linear-gradient(90deg,rgba(70,130,160,.10) 1px,transparent 1px);
  background-size:24px 24px;}
.trace:after{content:"";position:absolute;right:0;top:0;bottom:0;width:46px;background:linear-gradient(90deg,transparent,#070e14);}
.tlabel{position:absolute;left:8px;top:4px;z-index:2;font:700 11px Consolas,monospace;letter-spacing:.12em;}
.scroll{width:200%;height:100%;animation:icuscroll var(--d,6s) linear infinite;}
.scroll svg{width:100%;height:100%;display:block;}
.scroll path{fill:none;stroke-width:2;vector-effect:non-scaling-stroke;stroke-linejoin:round;}
.nums{display:flex;flex-direction:column;gap:2px;}
.num{border-bottom:1px solid #112331;padding:5px 4px 6px 4px;display:flex;justify-content:space-between;align-items:flex-end;}
.num .lab{font:700 11px Consolas,monospace;letter-spacing:.12em;}
.num .val{font:700 34px/1 Consolas,"SF Mono",monospace;text-shadow:0 0 12px currentColor;}
.num .unit{font:11px Consolas,monospace;color:#7f98a8;margin-left:5px;}
.num .val.off{color:#34495a;text-shadow:none;}
.alarm{border-color:#ff4d5e !important;animation:alarmglow 1.3s ease-in-out infinite;}
.tag{display:inline-block;border-radius:6px;padding:2px 9px;font:700 11px Consolas,monospace;letter-spacing:.12em;}
.fine{color:#5f7686;font-size:.72rem;margin-top:8px;}
/* result */
.resgrid{display:grid;grid-template-columns:300px minmax(0,1fr);gap:16px;align-items:center;}
@media (max-width:820px){.resgrid{grid-template-columns:1fr;}}
.bandbox{border-radius:10px;padding:8px 12px;font-weight:600;text-align:center;margin-top:4px;}
.brow{display:grid;grid-template-columns:minmax(0,40%) minmax(0,1fr) 52px;gap:8px;align-items:center;margin:6px 0;font-size:.84rem;}
.brow .lbl{color:#c5d9e6;text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.track{position:relative;height:14px;background:#0d1a24;border-radius:7px;}
.track:before{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:#2b4a5e;}
.fillr,.filll{position:absolute;top:2px;bottom:2px;border-radius:5px;}
.fillr{left:50%;background:linear-gradient(90deg,#b8323f,#ff4d5e);box-shadow:0 0 8px rgba(255,77,94,.5);}
.filll{right:50%;background:linear-gradient(270deg,#1b8aa8,#29d3ff);box-shadow:0 0 8px rgba(41,211,255,.45);}
.brow .v{font:12px Consolas,monospace;color:#8fb0c3;}
.legend{display:flex;gap:16px;font-size:.74rem;color:#7f98a8;margin-top:6px;}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;}
.kpi .k{border:1px solid #17303f;border-radius:12px;padding:12px 14px;background:#0a141c;}
.kpi .k .n{font:700 30px Consolas,monospace;color:#19f58a;text-shadow:0 0 10px rgba(25,245,138,.4);}
.kpi .k .t{font-size:.76rem;color:#7f98a8;letter-spacing:.08em;text-transform:uppercase;}
.cbar{display:grid;grid-template-columns:200px minmax(0,1fr) 60px;gap:10px;align-items:center;margin:9px 0;font-size:.88rem;}
@media (max-width:700px){.cbar{grid-template-columns:110px minmax(0,1fr) 54px;}}
.cbar .tr{height:16px;background:#0d1a24;border-radius:8px;overflow:hidden;}
.cbar .fl{height:100%;border-radius:8px;}
.credit{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;}
.credit .c{border:1px solid #17303f;border-radius:12px;padding:12px 16px;background:#0a141c;}
.credit .r{font-size:.72rem;letter-spacing:.14em;color:#19f58a;text-transform:uppercase;font-family:Consolas,monospace;}
.credit .nm{font-size:1.12rem;font-weight:600;color:#eaf6ff;margin-top:3px;}
.credit .dp{font-size:.82rem;color:#7f98a8;margin-top:2px;}
.foot{margin-top:26px;border-top:1px solid #12222d;padding-top:14px;text-align:center;color:#6d8797;font-size:.8rem;line-height:1.6;}
.foot b{color:#bcd2df;}
@keyframes icuscroll{from{transform:translateX(0)}to{transform:translateX(-50%)}}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
@keyframes alarmglow{0%,100%{box-shadow:0 0 0 rgba(255,77,94,0)}50%{box-shadow:0 0 22px rgba(255,77,94,.55)}}
</style>""")


# --------------------------------------------------------------------------- waveforms
def _g(x, mu, sig):
    return math.exp(-((x - mu) ** 2) / (2 * sig * sig))


def _ecg(ph):      # P, Q, R, S, T waves as a sum of gaussians (one beat, phase 0..1)
    return (0.11 * _g(ph, .14, .028) - 0.13 * _g(ph, .285, .010) + 1.0 * _g(ph, .305, .011)
            - 0.26 * _g(ph, .330, .012) + 0.25 * _g(ph, .56, .050))


def _pleth(ph):    # pulse-oximeter waveform: systolic upstroke + dicrotic notch
    ph = (ph + .12) % 1.0
    return 0.95 * _g(ph, .20, .075) + 0.32 * _g(ph, .47, .06)


def _resp(ph):
    return 0.5 + 0.5 * math.sin(2 * math.pi * ph - math.pi / 2)


def _path(fn, n, base, amp, width=1200, step=2.0):
    pts, x = [], 0.0
    while x <= 2 * width + 1e-6:
        y = base - amp * fn(((x / width) * n) % 1.0)
        pts.append(f"{x:.1f},{y:.1f}")
        x += step
    return "M" + " L".join(pts)


def _trace(label, color, fn, n, dur, base, amp, step):
    d = _path(fn, n, base, amp, step=step)
    return (f'<div class="trace"><div class="tlabel" style="color:{color}">{label}</div>'
            f'<div class="scroll" style="--d:{dur:.2f}s"><svg viewBox="0 0 2400 86" preserveAspectRatio="none">'
            f'<path d="{d}" stroke="{color}" style="filter:drop-shadow(0 0 3px {color})"/></svg></div></div>')


def _fmt(v, decimals=0):
    if v is None:
        return "--"
    return f"{v:.{decimals}f}"


def _num(label, color, value, unit, decimals=0):
    off = ' off' if value is None else ''
    return (f'<div class="num"><div class="lab" style="color:{color}">{label}</div>'
            f'<div><span class="val{off}" style="color:{color}">{_fmt(value, decimals)}</span>'
            f'<span class="unit">{unit}</span></div></div>')


def monitor_html(values, model_name="", alarm=False, band=None):
    """Bedside-monitor panel. Waveform speed follows the entered mean heart rate (default 75 bpm)."""
    hr = values.get("hr_mean")
    rate = float(hr) if hr else 75.0
    rate = min(max(rate, 35.0), 200.0)
    n = int(min(20, max(2, round(rate * 6 / 60))))
    dur = n * 60.0 / rate                              # exactly n beats per scroll period -> heart rate = rate
    traces = (_trace("ECG II", GREEN, _ecg, n, dur, 66, 54, 1.5)
              + _trace("PLETH", CYAN, _pleth, n, dur, 74, 52, 4.0)
              + _trace("RESP", YELLOW, _resp, 2, 6.0, 66, 44, 8.0))
    nums = (_num("HR", GREEN, hr, "bpm") + _num("MAP", RED, values.get("map_mean"), "mmHg")
            + _num("TEMP", YELLOW, values.get("temp_mean"), "°C", 1) + _num("GCS", "#e9f6ff", values.get("gcs_min"), "/15")
            + _num("LACT", AMBER, values.get("lactate"), "mmol/L", 1)
            + _num("URINE", CYAN, values.get("urine_output_24h"), "mL/24h"))
    tag = ""
    if alarm and band:
        tag = f'<span class="tag" style="background:rgba(255,77,94,.18);color:{RED};border:1px solid {RED}">▲ {html.escape(band.upper())} RISK</span>'
    cls = "icu-panel alarm" if alarm else "icu-panel"
    return _compact(f"""<div class="{cls}"><div class="monhead"><span class="live">BEDSIDE MONITOR · ADULT ICU · FIRST 24 H</span>
        {tag}<span>{html.escape(model_name)}</span></div>
        <div class="mon"><div>{traces}</div><div class="nums">{nums}</div></div>
        <div class="fine">Waveforms are an illustrative animation (rate follows the heart rate you enter), not patient recordings.</div></div>""")


# --------------------------------------------------------------------------- header, sections, footer
def header_html(title, subtitle, developer, supervisor):
    ekg = ('<svg viewBox="0 0 160 36" width="150" height="34" style="vertical-align:middle">'
           f'<path d="M0,20 L30,20 L36,20 L40,10 L45,32 L50,3 L55,26 L60,20 L100,20 L106,20 L110,13 L115,28 L120,8 L125,22 L130,20 L160,20" '
           f'fill="none" stroke="{GREEN}" stroke-width="2" style="filter:drop-shadow(0 0 3px {GREEN})"/></svg>')
    return _compact(f"""<div class="icu-panel"><div class="hero"><div><h1>{ekg} {html.escape(title)}</h1>
        <div class="sub">{html.escape(subtitle)}</div></div>
        <div class="chips"><span class="chip">Developer · <b>{html.escape(developer)}</b></span>
        <span class="chip">Supervisor · <b>{html.escape(supervisor)}</b></span></div></div></div>""")


def notice_html(text):
    return f'<div class="notice"><b>RESEARCH PROTOTYPE.</b> {html.escape(text)}</div>'


def section(title):
    return f'<div class="sec">{html.escape(title)}<i></i></div>'


def credits_html(developer, supervisor, department, university, year, title):
    return _compact(f"""<div class="credit">
        <div class="c"><div class="r">Developed by</div><div class="nm">{html.escape(developer)}</div>
          <div class="dp">{html.escape(department)}<br>{html.escape(university)}</div></div>
        <div class="c"><div class="r">Supervisor</div><div class="nm">{html.escape(supervisor)}</div>
          <div class="dp">{html.escape(department)}<br>{html.escape(university)}</div></div>
        <div class="c"><div class="r">Project</div><div class="nm" style="font-size:.95rem">{html.escape(title)}</div>
          <div class="dp">B.Sc. final-year project · {year}</div></div></div>""")


def footer_html(developer, supervisor, university, year):
    return _compact(f"""<div class="foot">Developed by <b>{html.escape(developer)}</b> · Supervised by <b>{html.escape(supervisor)}</b><br>
        {html.escape(university)} · {year}<br>Research prototype. Not a medical device. Not for clinical use.</div>""")


# --------------------------------------------------------------------------- result widgets
def band_color(band):
    return {"Below average": GREEN, "Around average": "#8df0b8", "Elevated": AMBER, "High": RED}.get(band, GREEN)


def gauge_html(p, band, base_rate, n_measured):
    """Semicircle risk gauge on a square-root scale (so the low-risk range is readable)."""
    cx, cy, r = 110, 110, 90
    f = math.sqrt(min(max(p, 0.0), 1.0))
    col = band_color(band)

    def pt(frac, rad):
        th = math.pi * (1 - frac)
        return cx + rad * math.cos(th), cy - rad * math.sin(th)

    ticks = ""
    for tv in (0, .05, .10, .25, .50, 1.0):
        x1, y1 = pt(math.sqrt(tv), r + 9)
        x2, y2 = pt(math.sqrt(tv), r + 16)
        xt, yt = pt(math.sqrt(tv), r + 26)
        ticks += (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#3b5a6e" stroke-width="1.5"/>'
                  f'<text x="{xt:.1f}" y="{yt + 3:.1f}" fill="{MUTED}" font-size="9" text-anchor="middle" font-family="Consolas,monospace">{tv * 100:.0f}%</text>')
    ax1, ay1 = pt(math.sqrt(base_rate), r - 12)
    ax2, ay2 = pt(math.sqrt(base_rate), r + 9)
    avg = f'<line x1="{ax1:.1f}" y1="{ay1:.1f}" x2="{ax2:.1f}" y2="{ay2:.1f}" stroke="#ffffff" stroke-width="2.5" stroke-dasharray="3 2"/>'
    ang = 180 * f
    kf = f"@keyframes ndl{{from{{transform:rotate(0deg)}}to{{transform:rotate({ang:.1f}deg)}}}}"
    return _compact(f"""<div style="text-align:center"><style>{kf}</style>
        <svg viewBox="-28 -26 276 180" width="100%" style="max-width:330px">
        <defs><linearGradient id="gg" x1="20" y1="0" x2="200" y2="0" gradientUnits="userSpaceOnUse">
          <stop offset="0" stop-color="{GREEN}"/><stop offset=".55" stop-color="{YELLOW}"/><stop offset="1" stop-color="{RED}"/></linearGradient></defs>
        <path d="M 20 110 A 90 90 0 0 1 200 110" fill="none" stroke="#10222e" stroke-width="18" stroke-linecap="round"/>
        <path d="M 20 110 A 90 90 0 0 1 200 110" fill="none" stroke="url(#gg)" stroke-width="10" stroke-linecap="round" opacity=".95"/>
        {ticks}{avg}
        <g style="transform-origin:110px 110px;transform:rotate({ang:.1f}deg);animation:ndl 1s cubic-bezier(.2,.8,.2,1)">
          <line x1="110" y1="110" x2="34" y2="110" stroke="#ffffff" stroke-width="3" stroke-linecap="round" style="filter:drop-shadow(0 0 4px {col})"/></g>
        <circle cx="110" cy="110" r="8" fill="#0b151d" stroke="{col}" stroke-width="3"/>
        <text x="110" y="143" text-anchor="middle" fill="{col}" font-size="30" font-weight="700" font-family="Consolas,monospace" style="filter:drop-shadow(0 0 6px {col})">{p * 100:.0f}%</text></svg>
        <div class="bandbox" style="background:{col}22;border:1px solid {col};color:{col}">{html.escape(band)}</div>
        <div class="fine">dashed mark = training average ({base_rate * 100:.0f}%) · based on {n_measured} entered measurements</div></div>""")


def contrib_html(contrib):
    """Diverging bars: left (blue) lowers risk, right (red) raises risk."""
    if contrib is None or len(contrib) == 0:
        return '<div class="fine">Per-patient explanation is available for XGBoost models only.</div>'
    mx = max(float(contrib.abs().max()), 1e-9)
    rows = ""
    for name, v in contrib.items():
        w = abs(float(v)) / mx * 50
        bar = f'<div class="fillr" style="width:{w:.1f}%"></div>' if v >= 0 else f'<div class="filll" style="width:{w:.1f}%"></div>'
        rows += (f'<div class="brow"><div class="lbl" title="{html.escape(str(name))}">{html.escape(str(name))}</div>'
                 f'<div class="track">{bar}</div><div class="v">{v:+.2f}</div></div>')
    return _compact(f"""<div>{rows}<div class="legend"><span><b style="color:{CYAN}">◀</b> lowers risk</span>
        <span><b style="color:{RED}">▶</b> raises risk</span><span>(log-odds units)</span></div></div>""")


def compare_bars_html(rows, lo=0.5, hi=1.0):
    """rows: list of (label, value). First row is highlighted."""
    out = ""
    for i, (label, v) in enumerate(rows):
        w = min(max((v - lo) / (hi - lo), 0), 1) * 100
        color = f"linear-gradient(90deg,#0e8f56,{GREEN})" if i == 0 else "linear-gradient(90deg,#27465a,#4f7f9c)"
        out += (f'<div class="cbar"><div style="color:#c5d9e6">{html.escape(label)}</div>'
                f'<div class="tr"><div class="fl" style="width:{w:.1f}%;background:{color}"></div></div>'
                f'<div class="mono" style="color:#cfe3ef">{v:.3f}</div></div>')
    return f'<div>{out}<div class="fine">AUROC on the held-out test set (axis starts at 0.5 = chance).</div></div>'


def kpi_html(items):
    cells = "".join(f'<div class="k"><div class="n">{html.escape(v)}</div><div class="t">{html.escape(t)}</div></div>' for t, v in items)
    return f'<div class="kpi">{cells}</div>'
