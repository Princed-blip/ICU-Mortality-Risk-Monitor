"""
ICU Mortality Risk Monitor: Streamlit web app (research prototype).
Run locally:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

import predictor as pr
import ui

# ---------------------------------------------------------------- project details (from the report cover page)
DEVELOPER = "Okpoko Chidubem Innocent"
SUPERVISOR = "Dr. T.O. Araoye"
DEPARTMENT = "Department of Biomedical Engineering"
UNIVERSITY = "University of Nigeria, Nsukka"
YEAR = 2026
PROJECT_TITLE = ("Development of a Predictive Model for In-Hospital Mortality in ICU Patients "
                 "Using Classical Machine Learning Techniques")
favicon = Image.open("favicon.png")
st.set_page_config(page_title="ICU Mortality Risk Monitor", page_icon="favicon", layout="wide")
st.markdown(ui.global_css(), unsafe_allow_html=True)
st.markdown(ui.header_html("ICU MORTALITY RISK MONITOR",
                           "Machine-learning estimate of in-hospital mortality from the first 24 hours of ICU data",
                           DEVELOPER, SUPERVISOR), unsafe_allow_html=True)



@st.cache_resource
def load():
    return pr.load_artifacts()


try:
    model, meta = load()
except Exception as e:                                   # missing model files, library version mismatch ...
    st.error("The model could not be loaded. Check that the `model/` folder is present and that the library "
             "versions in requirements.txt match the ones used for training.")
    st.exception(e)
    st.stop()


def _set_values(which):
    for s in pr.INPUTS:
        st.session_state[s["key"]] = s["ex_" + which]
    st.session_state.pop("res", None)


def _clear():
    for s in pr.INPUTS:
        st.session_state[s["key"]] = None if s["kind"] == "num" else s["options"][0]
    st.session_state.pop("res", None)


def widget(s, container):
    label = f"{s['label']} ({s['unit']})" if s.get("unit") else s["label"]
    if s["kind"] == "num":
        container.number_input(label, min_value=s["lo"], max_value=s["hi"], value=None, step=s["step"],
                               format=s["fmt"], key=s["key"], placeholder="not measured", help=s.get("help") or None)
    else:
        container.selectbox(label, s["options"], key=s["key"], format_func=lambda o, s=s: s["labels"].get(o, o))


tab_mon, tab_perf, tab_about = st.tabs(["🩺  Bedside monitor", "📊  Model performance", "ℹ️  About & credits"])

# ================================================================ TAB 1: monitor + inputs + result
with tab_mon:
    monitor_slot = st.empty()                            # filled at the end, once the result is known

    b1, b2, b3, _ = st.columns([1.3, 1.3, 1, 3])
    b1.button("Load example: stable", on_click=_set_values, args=("low",))
    b2.button("Load example: critically ill", on_click=_set_values, args=("high",))
    b3.button("Clear all", on_click=_clear)
    st.caption("Examples are fictional values, only to show how the tool behaves. "
               "Leave a box empty if it was not measured.")

    for group in [pr.G_PAT, pr.G_VIT, pr.G_LAB]:
        st.markdown(ui.section(group), unsafe_allow_html=True)
        cols = st.columns(4)
        for i, s in enumerate([x for x in pr.INPUTS if x["group"] == group]):
            widget(s, cols[i % 4])

    values = {s["key"]: st.session_state.get(s["key"]) for s in pr.INPUTS}
    n = pr.n_measured(values)

    st.markdown(ui.section("Risk estimate"), unsafe_allow_html=True)
    if st.button("Estimate risk", type="primary"):
        if n == 0:
            st.info("Enter at least a few measurements first.")
        else:
            p, X = pr.predict_risk(model, meta, values)
            band, ratio = pr.risk_band(p, meta["base_rate"])
            st.session_state["res"] = dict(p=p, band=band, ratio=ratio, n=n, snap=dict(values),
                                           contrib=pr.explain(model, X))

    res = st.session_state.get("res")
    alarm, alarm_band = False, None
    if res:
        stale = res["snap"] != values
        if stale:
            st.info("Inputs have changed since this estimate. Press **Estimate risk** to update it.")
        if res["n"] < 8:
            st.warning(f"Only {res['n']} measurements were entered. The estimate leans heavily on typical values, "
                       "is unreliable, and tends to UNDERSTATE risk.")
        panel = ('<div class="icu-panel"><div class="resgrid"><div>'
                 + ui.gauge_html(res["p"], res["band"], meta["base_rate"], res["n"])
                 + '</div><div><div class="fine" style="margin:0 0 6px 0;letter-spacing:.12em">WHAT PUSHED THIS ESTIMATE UP OR DOWN</div>'
                 + ui.contrib_html(res["contrib"])
                 + '</div></div></div>')
        st.markdown(panel, unsafe_allow_html=True)
        st.caption(f"{res['ratio']:.1f}× the training-data average of {meta['base_rate'] * 100:.0f}%. "
                   "A single entered value feeds both its 'first' and 'last' columns, so a factor may appear twice.")
        alarm = (not stale) and res["band"] in ("Elevated", "High")
        alarm_band = res["band"]
    else:
        st.markdown('<div class="icu-panel fine" style="text-align:center;padding:22px">'
                    'Enter the measurements above (or load an example) and press <b>Estimate risk</b>.</div>',
                    unsafe_allow_html=True)

    monitor_slot.markdown(ui.monitor_html(values, meta["model_name"], alarm, alarm_band), unsafe_allow_html=True)

# ================================================================ TAB 2: performance
with tab_perf:
    st.markdown(ui.section("Held-out test results"), unsafe_allow_html=True)
    lo, hi = meta["test_auroc_ci"]
    ref_brier = meta["base_rate"] * (1 - meta["base_rate"])
    st.markdown(ui.kpi_html([("AUROC", f"{meta['test_auroc']:.3f}"), ("AUPRC", f"{meta['test_auprc']:.3f}"),
                             ("Brier score", f"{meta['test_brier']:.3f}"),
                             ("Test patients", f"{meta['n_test']:,}"), ("Deaths in test set", f"{meta['n_deaths_test']}")]),
                unsafe_allow_html=True)
    st.markdown(ui.section("Compared with traditional ICU scores"), unsafe_allow_html=True)
    rows = [(f"{meta['model_name']} (this app)", meta["test_auroc"])] + list(meta["baselines"].items())
    st.markdown('<div class="icu-panel">' + ui.compare_bars_html(rows) + '</div>', unsafe_allow_html=True)
    st.write(f"AUROC 95% confidence interval for this model: **{lo:.3f} to {hi:.3f}**. "
             f"A Brier score of {meta['test_brier']:.3f} compares with {ref_brier:.3f} for a model that always "
             "predicts the average risk; lower is better.")
    st.caption("SAPS-I and SOFA are the official scores supplied with the dataset, evaluated on the same test patients.")

# ================================================================ TAB 3: about + credits
with tab_about:
    st.markdown(ui.section("Credits"), unsafe_allow_html=True)
    st.markdown(ui.credits_html(DEVELOPER, SUPERVISOR, DEPARTMENT, UNIVERSITY, YEAR, PROJECT_TITLE),
                unsafe_allow_html=True)
    st.markdown(ui.section("How it was built"), unsafe_allow_html=True)
    st.write(f"- **Model:** {meta['model_name']} inside a leakage-safe pipeline (missing-value filter, imputation, "
             "two-stage feature selection). Trained **without** SMOTE, because SMOTE worsened the accuracy of the "
             "predicted probabilities in the experiments.\n"
             f"- **Data:** {meta['dataset']}. {meta['n_fit']:,} ICU stays used for fitting; "
             f"{meta['n_test']:,} held-out stays ({meta['n_deaths_test']} deaths) used for testing.\n"
             f"- **Outcome:** death before hospital discharge (average rate {meta['base_rate'] * 100:.1f}%).")
    st.markdown(ui.section("Limitations"), unsafe_allow_html=True)
    st.write("- Trained on one dataset from US hospitals; performance elsewhere (including Nigeria) is unknown.\n"
             "- The model was trained on summaries (mean, lowest, first, last) of the first 24 hours. "
             "This form takes one value per item and uses it for related columns, which is an approximation.\n"
             "- Unmeasured items are replaced by typical (median) training values. The typical patient is healthier "
             "than the average patient, so the more items are left blank, the more the estimate tends to understate risk.\n"
             "- It estimates population-level risk and says nothing about what caused a patient's condition or what treatment to give.\n"
             "- The waveforms on the monitor are an illustrative animation, not patient recordings.\n"
             "- Not clinically validated and not approved for any medical use.")
    st.caption("Data: Silva I, Moody G, Scott DJ, Celi LA, Mark RG. Predicting in-hospital mortality of ICU patients: "
               "the PhysioNet/Computing in Cardiology Challenge 2012. Comput Cardiol 2012;39:245-248.")
st.markdown(ui.notice_html("Not a medical device and not for clinical decisions. Trained on one public dataset "
                           "(US ICUs, collected before 2012) and not validated on any other patients."),
            unsafe_allow_html=True)
st.markdown(ui.footer_html(DEVELOPER, SUPERVISOR, UNIVERSITY, YEAR), unsafe_allow_html=True)
