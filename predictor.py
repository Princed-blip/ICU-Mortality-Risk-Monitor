"""
Prediction logic for the web app (no Streamlit code here, so it can be tested on its own).

Flow:  form values -> one full-width row (unknown = NaN) -> same cleaning + feature
engineering as training -> saved pipeline (imputes the NaNs) -> probability.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

import pipeline as P          # must be importable so joblib can unpickle the custom classes

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "icu_model.joblib")
META_PATH = os.path.join(HERE, "model", "app_meta.json")

G_PAT, G_VIT, G_LAB = "Patient", "Vitals (first 24 hours)", "Labs (first 24 hours)"


def N(key, label, group, unit, lo, hi, step, fmt, cols, low, high, help=""):
    return dict(key=key, label=label, group=group, kind="num", unit=unit, lo=float(lo), hi=float(hi),
                step=float(step), fmt=fmt, cols=cols, ex_low=low, ex_high=high, help=help)


def S(key, label, group, options, cols, low, high, mapping=None, labels=None, help=""):
    return dict(key=key, label=label, group=group, kind="select", options=options, cols=cols,
                map=mapping or {}, labels=labels or {}, ex_low=low, ex_high=high, help=help)


def both(name):                     # a single lab value feeds both the "first" and "last" columns
    return [f"{name}_first", f"{name}_last"]


INPUTS = [
    N("age", "Age", G_PAT, "years", 18, 100, 1, "%.0f", ["age"], 52, 79),
    S("gender", "Sex", G_PAT, ["M", "F"], ["gender"], "M", "M", labels={"M": "Male", "F": "Female"}),
    S("icu_type", "ICU type", G_PAT, ["CCU", "CSRU", "MICU", "SICU"], ["icu_type"], "CSRU", "MICU",
      labels={"CCU": "Coronary care", "CSRU": "Cardiac surgery recovery", "MICU": "Medical ICU", "SICU": "Surgical ICU"}),
    S("mech_vent_24h", "Mechanical ventilation in first 24 h", G_PAT, ["No", "Yes"], ["mech_vent_24h"], "No", "Yes",
      mapping={"No": 0.0, "Yes": 1.0}),

    N("hr_mean", "Heart rate, mean", G_VIT, "bpm", 20, 250, 1, "%.0f", ["hr_mean"], 78, 112),
    N("hr_max", "Heart rate, highest", G_VIT, "bpm", 20, 300, 1, "%.0f", ["hr_max"], 95, 145),
    N("map_mean", "Mean arterial pressure, mean", G_VIT, "mmHg", 20, 200, 1, "%.0f", ["map_mean"], 82, 62),
    N("map_min", "Mean arterial pressure, lowest", G_VIT, "mmHg", 20, 200, 1, "%.0f", ["map_min"], 68, 45),
    N("sbp_mean", "Systolic BP, mean", G_VIT, "mmHg", 30, 300, 1, "%.0f", ["sbp_mean"], 125, 98),
    N("sbp_min", "Systolic BP, lowest", G_VIT, "mmHg", 30, 300, 1, "%.0f", ["sbp_min"], 105, 70),
    N("temp_mean", "Temperature, mean", G_VIT, "°C", 30, 43, 0.1, "%.1f", ["temp_mean"], 36.8, 38.4),
    N("gcs_min", "Glasgow Coma Scale, lowest", G_VIT, "3-15", 3, 15, 1, "%.0f",
      ["gcs_total_mean", "gcs_total_min", "gcs_total_first", "gcs_total_last"], 15, 5),
    N("urine_output_24h", "Urine output, total 24 h", G_VIT, "mL", 0, 10000, 50, "%.0f", ["urine_output_24h"], 2400, 400),

    N("bun", "BUN (urea nitrogen)", G_LAB, "mg/dL", 1, 250, 1, "%.0f", both("bun"), 14, 62),
    N("creatinine", "Creatinine", G_LAB, "mg/dL", 0.1, 20, 0.1, "%.1f", both("creatinine"), 0.9, 2.8),
    N("glucose", "Glucose", G_LAB, "mg/dL", 10, 2000, 5, "%.0f", both("glucose"), 110, 190),
    N("bicarbonate", "Bicarbonate (HCO3)", G_LAB, "mmol/L", 2, 60, 1, "%.0f", both("bicarbonate"), 25, 16),
    N("hematocrit", "Hematocrit", G_LAB, "%", 5, 70, 0.5, "%.1f", both("hematocrit"), 36, 27),
    N("potassium", "Potassium", G_LAB, "mmol/L", 1.5, 10, 0.1, "%.1f", both("potassium"), 4.1, 5.0),
    N("sodium", "Sodium", G_LAB, "mmol/L", 100, 180, 1, "%.0f", both("sodium"), 139, 146),
    N("platelets", "Platelets", G_LAB, "x10³/µL", 5, 1500, 5, "%.0f", both("platelets"), 220, 70),
    N("wbc", "White blood cells", G_LAB, "x10³/µL", 0.1, 200, 0.5, "%.1f", both("wbc"), 9, 21),
    N("lactate", "Lactate", G_LAB, "mmol/L", 0.1, 40, 0.1, "%.1f", both("lactate"), 1.2, 6.5),
    N("ph", "Arterial pH", G_LAB, "", 6.6, 7.8, 0.01, "%.2f", both("ph"), 7.40, 7.20),
    N("pao2", "PaO2", G_LAB, "mmHg", 20, 700, 5, "%.0f", both("pao2"), 120, 70),
    N("fio2", "FiO2 (fraction, 0.21-1.0)", G_LAB, "", 0.21, 1.0, 0.05, "%.2f", both("fio2"), 0.4, 0.8),
    N("paco2", "PaCO2", G_LAB, "mmHg", 10, 150, 1, "%.0f", both("paco2"), 40, 52),
    N("bilirubin", "Bilirubin", G_LAB, "mg/dL", 0.05, 60, 0.1, "%.1f", both("bilirubin"), 0.7, 3.5),
    N("magnesium", "Magnesium", G_LAB, "mg/dL", 0.2, 10, 0.1, "%.1f", both("magnesium"), 2.0, 2.4),
]

NICE = {"bun": "BUN", "hr": "Heart rate", "map": "MAP", "sbp": "Systolic BP", "gcs_total": "GCS", "wbc": "WBC",
        "ph": "pH", "pao2": "PaO2", "paco2": "PaCO2", "fio2": "FiO2", "pf_ratio": "PaO2/FiO2 ratio",
        "shock_index": "Shock index", "urine_output": "Urine output", "mech_vent": "Mech. ventilation",
        "icu_type": "ICU type", "hematocrit": "Hematocrit"}
SUFFIX = {"_first": " (first)", "_last": " (last)", "_mean": " (mean)", "_min": " (lowest)",
          "_max": " (highest)", "_std": " (variability)"}


def pretty(name):
    suffix = ""
    for s, txt in SUFFIX.items():
        if name.endswith(s):
            name, suffix = name[: -len(s)], txt
            break
    for k, v in NICE.items():
        if name == k or name.startswith(k + "_"):
            name = v + name[len(k):].replace("_", " ")
            break
    else:
        name = name.replace("_", " ").capitalize()
    return name + suffix


def load_artifacts():
    return joblib.load(MODEL_PATH), json.load(open(META_PATH))


def build_row(values, feature_columns):
    """One-row DataFrame with every model column; anything the user left blank stays NaN."""
    row = {c: np.nan for c in feature_columns}
    for c in feature_columns:
        if c in P.CAT_COLS:
            row[c] = "UNKNOWN"
    for spec in INPUTS:
        v = values.get(spec["key"])
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        v = spec["map"].get(v, v) if spec["kind"] == "select" else float(v)
        for col in spec["cols"]:
            if col in row:
                row[col] = v
    return pd.DataFrame([row], columns=feature_columns)


def n_measured(values):
    return sum(1 for s in INPUTS if s["kind"] == "num" and values.get(s["key"]) is not None)


def predict_risk(model, meta, values):
    cols = meta["feature_columns"]
    df = build_row(values, cols)
    df, _ = P.clean_outliers(df)               # same plausibility rules as training
    df = P.engineer_features(df)               # PaO2/FiO2, shock index ...
    X = df[cols]
    return float(model.predict_proba(X)[0, 1]), X


def explain(model, X, top=8):
    """Per-patient contributions (log-odds) via XGBoost's built-in SHAP values. None if unavailable."""
    try:
        import xgboost as xgb
        inner = model.named_steps["clf"].estimator_
        if not hasattr(inner, "get_booster"):
            return None
        pre = model[:-1]
        names = list(pre.get_feature_names_out())
        contrib = inner.get_booster().predict(xgb.DMatrix(np.asarray(pre.transform(X), float)), pred_contribs=True)[0]
        s = pd.Series(contrib[:-1], index=[pretty(n) for n in names])
        s = s.groupby(level=0).sum()
        return s.reindex(s.abs().sort_values(ascending=False).index).head(top)
    except Exception:
        return None


def risk_band(p, base_rate):
    r = p / base_rate if base_rate else np.nan
    if r < 0.5: band = "Below average"
    elif r < 2: band = "Around average"
    elif r < 4: band = "Elevated"
    else: band = "High"
    return band, r
