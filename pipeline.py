"""
Building blocks for the ICU mortality pipeline.

Design rule: EVERYTHING that learns from data (missingness filter, imputation,
scaling, univariable filter, RF feature selection, SMOTE) lives INSIDE the sklearn
Pipeline, so it is re-fitted on training folds only. No information from
validation/test rows leaks into preprocessing.
"""
import warnings
from functools import partial
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.metaestimators import available_if
from sklearn.tree import DecisionTreeClassifier

try:                                    # preferred, as stated in the thesis
    from imblearn.over_sampling import SMOTE as _ImbSMOTE
except Exception:
    _ImbSMOTE = None
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

ID_COLS = ["subject_id", "hadm_id", "stay_id", "source_set"]
LEAKAGE_COLS = ["icu_los_hours", "hospital_los_days", "length_of_stay_days", "survival_days"]  # known only AFTER the outcome window
BASELINE_COLS = ["sofa_score", "saps1_score"]          # official severity scores: comparison baselines, never inputs
TARGET = "hospital_expire_flag"
CAT_COLS = ["gender", "ethnicity", "insurance", "admission_type", "first_careunit", "icu_type"]

# Clinically plausible ranges (value outside -> treated as missing, then imputed)
PLAUSIBLE = {
    "hr": (20, 300), "map": (20, 200), "sbp": (30, 300), "rr": (4, 70), "temp": (30, 43),
    "spo2": (50, 100), "gcs_total": (3, 15), "creatinine": (0.1, 20), "bun": (1, 250),
    "lactate": (0.1, 40), "hemoglobin": (2, 25), "wbc": (0.1, 200), "platelets": (5, 1500),
    "sodium": (100, 180), "potassium": (1.5, 10), "bicarbonate": (2, 60), "bilirubin": (0.05, 60),
    "albumin": (0.5, 6), "ph": (6.6, 7.8), "pao2": (20, 700), "paco2": (10, 150), "fio2": (0.21, 1.0),
    "urine_output": (0, 10000),
    # added for the PhysioNet 2012 variable set
    "dbp": (10, 200), "sao2": (50, 100), "hematocrit": (5, 70), "glucose": (10, 2000),
    "magnesium": (0.2, 10), "alp": (5, 5000), "alt": (1, 20000), "ast": (1, 40000),
    "cholesterol": (20, 600), "troponin_i": (0, 100), "troponin_t": (0, 50),
    "weight": (20, 300), "height": (100, 230),
}


# ----------------------------------------------------------------------------
# Cohort + stateless feature engineering (safe to run before splitting)
# ----------------------------------------------------------------------------
def select_cohort(df, min_icu_hours=24):
    n0 = len(df)
    df = df[df["age"] >= 18]
    df = df.dropna(subset=[TARGET])
    if min_icu_hours and "icu_los_hours" in df:
        df = df[df["icu_los_hours"] >= min_icu_hours]
    df = df.sort_values("stay_id").drop_duplicates("subject_id", keep="first")   # first ICU stay only
    los_note = f", ICU stay >= {min_icu_hours}h" if (min_icu_hours and "icu_los_hours" in df) else ""
    print(f"[cohort] {n0} -> {len(df)} stays after inclusion criteria (age>=18, first stay{los_note})")
    return df.reset_index(drop=True)


def clean_outliers(df):
    """Set physiologically implausible values to NaN. Returns cleaned frame + report."""
    df = df.copy()
    report = {}
    for col in df.columns:
        for pref, (lo, hi) in PLAUSIBLE.items():
            if (col.startswith(pref + "_") or col == pref) and not col.endswith("_std"):
                bad = (df[col] < lo) | (df[col] > hi)
                if bad.any():
                    report[col] = int(bad.sum())
                    df.loc[bad, col] = np.nan
                break
    print(f"[outliers] {sum(report.values())} implausible values set to NaN across {len(report)} columns")
    return df, report


def _points(values, cuts, ascending=True):
    """SOFA organ points: number of thresholds crossed. NaN -> 0 (assumed normal)."""
    v = np.asarray(values, float)
    pts = np.zeros(len(v))
    for c in cuts:
        pts += (v >= c) if ascending else (v < c)
    return np.where(np.isnan(v), 0, pts)


def engineer_features(df):
    """Derived composites: PaO2/FiO2, shock index, and a simplified SOFA-like score."""
    df = df.copy()
    df["pf_ratio_first"] = df["pao2_first"] / df["fio2_first"]
    df["shock_index_mean"] = df["hr_mean"] / df["sbp_mean"]
    df["shock_index_max"] = df["hr_max"] / df["sbp_min"]
    resp = _points(df["pf_ratio_first"], [400, 300, 200, 100], ascending=False)
    coag = _points(df[["platelets_first", "platelets_last"]].min(axis=1), [150, 100, 50, 20], ascending=False)
    liver = _points(df[["bilirubin_first", "bilirubin_last"]].max(axis=1), [1.2, 2.0, 6.0, 12.0])
    cardio = _points(df["map_min"], [70], ascending=False)                 # no vasopressor data in this extract
    cns = _points(df["gcs_total_min"], [15, 13, 10, 6], ascending=False)
    renal = _points(df[["creatinine_first", "creatinine_last"]].max(axis=1), [1.2, 2.0, 3.5, 5.0])
    df["sofa_like"] = resp + coag + liver + cardio + cns + renal
    return df


# ----------------------------------------------------------------------------
# Leakage-safe transformers
# ----------------------------------------------------------------------------
class MissingnessDropper(BaseEstimator, TransformerMixin):
    """Drop columns whose missing fraction in the TRAINING data exceeds `threshold`."""
    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        frac = X.isna().mean()
        self.keep_ = list(frac.index[frac <= self.threshold])
        self.dropped_ = list(frac.index[frac > self.threshold])
        return self

    def transform(self, X):
        return X[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.keep_)


class UnivariateFilter(BaseEstimator, TransformerMixin):
    """Stage-1 selection: keep features that differ between survivors/non-survivors (Mann-Whitney, p<alpha)."""
    def __init__(self, alpha=0.05, min_keep=10):
        self.alpha = alpha
        self.min_keep = min_keep

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y).astype(int)
        p = np.ones(X.shape[1])
        for j in range(X.shape[1]):
            a, b = X[y == 1, j], X[y == 0, j]
            if np.ptp(X[:, j]) > 0:
                p[j] = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        keep = p < self.alpha
        if keep.sum() < self.min_keep:
            keep = np.zeros_like(keep); keep[np.argsort(p)[: self.min_keep]] = True
        self.pvalues_, self.support_ = p, keep
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)[self.support_]


def _smote(X, y, k, seed):
    """Minimal SMOTE (used only if imbalanced-learn is not installed)."""
    rng = np.random.default_rng(seed)
    minority = X[y == 1]
    n_new = int((y == 0).sum() - len(minority))
    if n_new <= 0 or len(minority) < 3:
        return X, y
    k = min(k, len(minority) - 1)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(minority).kneighbors(minority, return_distance=False)[:, 1:]
    base = rng.integers(0, len(minority), n_new)
    neigh = idx[base, rng.integers(0, k, n_new)]
    new = minority[base] + rng.random((n_new, 1)) * (minority[neigh] - minority[base])
    return np.vstack([X, new]), np.concatenate([y, np.ones(n_new, dtype=int)])


class SmoteClassifier(ClassifierMixin, BaseEstimator):
    """Classifier wrapper that oversamples the minority class at FIT time only (never at predict time)."""
    def __init__(self, estimator=None, use_smote=True, k_neighbors=5, random_state=42):
        self.estimator = estimator
        self.use_smote = use_smote
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y).astype(int)
        if self.use_smote:
            if _ImbSMOTE is not None:
                X, y = _ImbSMOTE(k_neighbors=self.k_neighbors, random_state=self.random_state).fit_resample(X, y)
            else:
                X, y = _smote(X, y, self.k_neighbors, self.random_state)
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X, float))

    # exposed only if the wrapped model really has them (sklearn scorers pick whichever exists)
    @available_if(lambda self: hasattr(self.estimator, "predict_proba"))
    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X, float))

    @available_if(lambda self: hasattr(self.estimator, "decision_function"))
    def decision_function(self, X):
        return self.estimator_.decision_function(np.asarray(X, float))


# ----------------------------------------------------------------------------
# Pipeline + model zoo
# ----------------------------------------------------------------------------
def _num_cols(X):
    return [c for c in X.columns if c not in CAT_COLS]


def _cat_cols(X):
    return [c for c in X.columns if c in CAT_COLS]


def _cap(n_features, X):
    return min(n_features, X.shape[1])


def build_pipeline(estimator, scale, imputer="median", use_smote=True, n_features=30, seed=42):
    if imputer == "mice":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(max_iter=5, random_state=seed, skip_complete=True)
    else:
        imp = SimpleImputer(strategy="median")
    num_steps = [("impute", imp)] + ([("scale", StandardScaler())] if scale else [])
    pre = ColumnTransformer(
        [("num", Pipeline(num_steps), _num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _cat_cols)],
        verbose_feature_names_out=False)
    rf_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced_subsample",
                               n_jobs=-1, random_state=seed),
        max_features=partial(_cap, n_features), threshold=-np.inf)
    return Pipeline([
        ("drop_missing", MissingnessDropper(0.6)),
        ("prep", pre),
        ("univar", UnivariateFilter(0.05)),
        ("select", rf_selector),
        ("clf", SmoteClassifier(estimator, use_smote=use_smote, random_state=seed)),
    ])


def model_zoo(seed=42):
    """name -> (estimator, needs_scaling, param_distributions). Params are prefixed for the pipeline."""
    P = "clf__estimator__"
    zoo = {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, random_state=seed), True,
            {P + "C": loguniform(1e-3, 1e2)}),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=seed), False,
            {P + "max_depth": [3, 4, 5, 6, 8, 10], P + "min_samples_split": [2, 10, 30, 60],
             P + "min_samples_leaf": [1, 5, 10, 20, 40]}),
        "Random Forest": (
            RandomForestClassifier(n_jobs=-1, random_state=seed), False,
            {P + "n_estimators": [100, 200, 300], P + "max_depth": [6, 8, 12, None],
             P + "max_features": ["sqrt", "log2", 0.3], P + "min_samples_leaf": [1, 3, 5]}),
    }
    if XGBClassifier is not None:
        zoo["Gradient Boosting (XGBoost)"] = (
            XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=seed, tree_method="hist"), False,
            {P + "learning_rate": [0.03, 0.05, 0.1], P + "n_estimators": [100, 200, 300],
             P + "max_depth": [3, 4, 5], P + "subsample": [0.7, 0.85, 1.0]})
    else:
        warnings.warn("xgboost not installed -> using sklearn HistGradientBoosting as a stand-in. "
                      "Install xgboost to match the thesis methodology.")
        zoo["Gradient Boosting (HistGB stand-in)"] = (
            HistGradientBoostingClassifier(random_state=seed), False,
            {P + "learning_rate": [0.03, 0.05, 0.1], P + "max_iter": [100, 200, 300],
             P + "max_depth": [3, 4, 5, None]})
    zoo["Support Vector Machine"] = (
        SVC(kernel="rbf", probability=False, cache_size=1000, random_state=seed), True,
        {P + "C": loguniform(0.1, 50), P + "gamma": ["scale", 0.01, 0.03, 0.1]})
    return zoo
