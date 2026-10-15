#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-analysis of the EOIC case-control study (revision after rejection).

Google Colab: upload this file and your data, set DATA_FILE / FLOW_FILE / OUT_DIR
in the CONFIGURATION block below, then run   %run analyse_EOIC_definitive.py
(or paste the whole file into one cell and run it).

The project database (columns Group = G1_Young_Cancer / G2_Young_Healthy /
G3_Old_Cancer, Age, Sex, Income_below_3000, ..._per_week, Rectal_bleeding, ...)
is recognised automatically. flow.csv (optional) has the same layout as the
project flow file (rows "Approached / screened", "Not eligible: ...",
"Excluded: ...", "Analysed"; columns CEOIC / CoY / CLOIC) and produces Figure 1.

The CSV must follow `data_template.csv` (one row per participant; column
dictionary in DATA_DICTIONARY below). If your column names differ, edit the
RENAME dictionary — nothing else needs to change.

Outputs (in output_folder, default ./results):
    results.xlsx      all tables, one sheet per table
    fig_dca.png       decision curves (M1 vs M3) at an assumed prevalence
    fig_calib.png     calibration plot of the primary model on the test set
    model_equation.csv full equation of the primary model (TRIPOD+AI)
    log.txt           run log (n per step, warnings)

What is computed
----------------
 0. Case definition: C21 and non-adenocarcinoma excluded (n reported).
 1. Table 2   : characteristics by group (median [IQR], mean ± SD, %), p-values
                (Mann-Whitney / chi²), BH correction for CEOIC vs CoY.
 2. Table 3   : each dietary exposure modelled SEPARATELY, multiple imputation
                (m = 10, Rubin's rules):
                  Model 1 = matching variables
                  Model 2 = + DAG confounders (main)
                  Model 3 = + usual BMI (potential mediator)
                OR per SD, OR by quartile (Q1 ref, cut-offs from controls),
                p-trend (quartile medians), BH q-values across foods.
                MDS per point and by category (0–3 / 4–5 / 6–8).
 3. Table 4   : sensitivity analyses for primary exposures (C18–C20 only,
                MSI-high excluded, Amsterdam II excluded, cases interviewed
                ≤ 6 months, adjustment for interview year, energy residual,
                complete case) + estimates by control source with
                heterogeneity test + tertiles of diagnosis-to-interview delay.
 4. E-values for primary exposures (Model 2, per SD).
 5. Prediction: nested elastic-net models M0–M3 (+ M1/M3 without Hb), tuned
                by stratified 5-fold CV on the development set, tested once on
                the temporal test set: AUC (bootstrap CI), ΔAUC vs previous
                model (bootstrap CI), Brier, scaled Brier, calibration
                intercept & slope; cut-offs fixed on out-of-fold development
                predictions (rule-out / Youden / rule-in) with Se, Sp (Wilson),
                LR+/LR−, PPV/NPV at 0.1–2% prevalence; decision curves at an
                assumed prevalence; symptomatic-control subgroup (out-of-fold);
                development vs test characteristics; full model equation.
"""

# ##########################################################################
#  CONFIGURATION — the only lines to edit
# ##########################################################################
DATA_FILE = "IC_Morocco_analysis.csv"   # study database (see README for the required columns)
FLOW_FILE = "IC_Morocco_flow.csv"       # participant-flow counts (optional; None if absent)
OUT_DIR   = "results"                   # output folder
# ##########################################################################
import sys, os, json, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")
SEED = 2026
M_IMP = 10          # number of imputations
N_BOOT = 1000       # bootstrap resamples
TEST_CUTOFF = "2026-05-06"   # interviewed <= this date -> development set
PREVALENCES = [0.001, 0.002, 0.005, 0.01, 0.02]
DCA_PREVALENCE = 0.01
# Haemoglobin was measured differently in cases and controls -> excluded from the
# primary model. Set to True only to report an optional sensitivity model with Hb.
USE_HB_SENSITIVITY = False

# --------------------------------------------------------------------------
# Column names. Edit the left-hand side if your file uses other names.
# --------------------------------------------------------------------------
RENAME = {
    # "your_column_name": "expected_name",
}

DATA_DICTIONARY = {
    "id": "participant identifier",
    "group": "CEOIC / CoY / CLOIC",
    "age": "years (age at diagnosis for cases, at inclusion for controls)",
    "sex": "1 = woman, 0 = man",
    "income_cat": "income category used for matching (0/1)",
    "educ": "0 none, 1 primary, 2 secondary, 3 university",
    "rural": "1 = rural",
    "insurance": "none / AMO / Tadamon",
    "marital": "married / single / widowed_divorced",
    "household_size": "number of persons in the household (optional)",
    "site": "recruitment site (cases) — text",
    "control_source": "outpatient / campaign / volunteer (controls only)",
    "interview_date": "YYYY-MM-DD",
    "diagnosis_date": "YYYY-MM-DD (cases only)",
    "icd10": "C17 / C18 / C19 / C20 / C21 (cases only)",
    "adenocarcinoma": "1 = adenocarcinoma, 0 = other histology (cases only)",
    "stage34": "1 = AJCC stage III–IV (cases)",
    "msi_high": "1 = MSI-high or dMMR (cases)",
    "amsterdam2": "1 = meets Amsterdam II criteria",
    # diet, times or servings per week
    "red_meat": "", "processed_meat": "", "grilled_meat": "", "preserved_salted": "",
    "upf": "", "sugary_drinks": "", "traditional_dishes": "", "fish": "",
    "legumes": "", "vegetables": "", "fruits": "", "whole_grains": "",
    "nuts": "", "dairy": "", "olive_oil": "", "mint_tea": "",
    "energy": "kcal/day (approximate, from FFQ)",
    "mds": "Mediterranean Diet Score 0–8 (computed by the script if absent)",
    "smoking_cat": "cigarettes/day category 0–4",
    "alcohol_cat": "0–2", "phys_act": "days/week >= 30 min",
    "bmi_usual": "kg/m², usual weight 2 years before", "bmi_measured": "kg/m²",
    "famhx": "1st-degree family history of IC (0/1)",
    "ibd": "IBD diagnosed >= 24 months before (0/1)",
    "diabetes": "0/1", "hypertension": "0/1", "aspirin": "0/1",
    "bleeding": "rectal bleeding, 12 months (0/1)", "bowel_change": "0/1",
    "abd_pain": "0/1", "weight_loss": "0/1", "hb": "haemoglobin g/dL",
}

FOODS = ["red_meat", "processed_meat", "grilled_meat", "preserved_salted", "upf",
         "sugary_drinks", "traditional_dishes", "fish", "legumes", "vegetables",
         "fruits", "whole_grains", "nuts", "dairy", "olive_oil", "mint_tea"]
PRIMARY = ["mds", "upf", "preserved_salted", "red_meat"]       # a priori (WCRF)
MATCH = ["age", "sex", "income_cat"]
CONF = ["educ", "rural", "smoking_cat", "phys_act", "famhx", "ibd"]   # from the DAG
MEDIATOR = ["bmi_usual"]
SYMPT = ["bleeding", "bowel_change", "abd_pain", "weight_loss"]
RISK = ["famhx", "ibd", "smoking_cat"]
LIFE = ["alcohol_cat", "phys_act", "bmi_usual", "educ", "rural"]
# Mediterranean score: (component, items, direction)  -- CHECK against what was really done
MDS_COMPONENTS = [("vegetables", ["vegetables"], +1), ("legumes", ["legumes"], +1),
                  ("fruits_nuts", ["fruits", "nuts"], +1), ("cereals", ["whole_grains"], +1),
                  ("fish", ["fish"], +1), ("olive_oil", ["olive_oil"], +1),
                  ("meat", ["red_meat", "processed_meat"], -1), ("dairy", ["dairy"], -1)]

LOG = []
CONTROL_SOURCES = []
PRETTY = {"red_meat": "Red meat", "processed_meat": "Processed meat", "grilled_meat": "Charcoal-grilled meat",
          "preserved_salted": "Preserved/salted food", "upf": "Ultra-processed food", "sugary_drinks": "Sugary drinks",
          "traditional_dishes": "Traditional dishes", "fish": "Fish", "legumes": "Legumes", "vegetables": "Vegetables",
          "fruits": "Fruits", "whole_grains": "Whole grains", "nuts": "Nuts & dried fruits", "dairy": "Dairy",
          "olive_oil": "Olive oil", "mint_tea": "Mint tea", "energy": "Energy intake", "mds": "Mediterranean Diet Score",
          "alcohol_cat": "Alcohol use", "phys_act": "Physical activity", "bmi_usual": "Usual BMI", "educ": "Education",
          "rural": "Rural residence", "famhx": "Family history of IC", "ibd": "IBD ≥ 24 months", "smoking_cat": "Cigarettes/day",
          "bleeding": "Rectal bleeding", "bowel_change": "Change in bowel habit", "abd_pain": "Abdominal pain",
          "weight_loss": "Unexplained weight loss", "hb": "Haemoglobin"}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.append(s)


# ==========================================================================
# helpers
# ==========================================================================
def fmt_or(o, l, u):
    if not np.all(np.isfinite([o, l, u])):
        return "not estimable"
    return f"{o:.2f} ({l:.2f}–{u:.2f})"


def fmt_p(p):
    if p is None or np.isnan(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def rubin(est, var):
    """Pool m estimates (log-OR) and variances with Rubin's rules."""
    est, var = np.asarray(est), np.asarray(var)
    m = len(est)
    qbar = est.mean(0); ubar = var.mean(0); b = est.var(0, ddof=1) if m > 1 else 0 * qbar
    t = ubar + (1 + 1 / m) * b
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (1 + 1 / m) * b / ubar
        df = (m - 1) * (1 + 1 / r) ** 2 if m > 1 else np.inf * qbar
    df = np.where(np.isfinite(df), df, 1e6)
    se = np.sqrt(t)
    p = 2 * stats.t.sf(np.abs(qbar / se), df)
    tq = stats.t.ppf(0.975, df)
    return qbar, se, qbar - tq * se, qbar + tq * se, p


def impute(df, cols, m=M_IMP, seed=SEED):
    """Multiple imputation by chained equations (outcome included as predictor)."""
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    out = []
    for i in range(m):
        imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True,
                               max_iter=10, random_state=seed + i)
        arr = imp.fit_transform(num)
        d = df.copy()
        d[cols] = arr
        # keep binary / categorical variables in range
        for c in cols:
            orig = num[c].dropna().unique()
            if len(orig) and np.all(np.isin(orig, np.arange(0, 11))) and len(orig) <= 11:
                d[c] = np.clip(np.round(d[c]), orig.min(), orig.max())
            else:
                d[c] = np.clip(d[c], num[c].min(), num[c].max())
        out.append(d)
    return out


def logit_fit(d, y, X):
    """Logistic regression robust to constant / redundant columns (dropped automatically)."""
    Xd = d[X].astype(float)
    keep = [c for c in X if Xd[c].nunique(dropna=True) > 1]          # constant in this subset
    Xd = Xd[keep].loc[:, ~Xd[keep].T.duplicated()]                    # exact duplicates
    Xc = sm.add_constant(Xd, has_constant="add")
    # IRLS + pseudo-inverse: no 'Singular matrix' error when columns are collinear
    return sm.GLM(d[y].astype(float), Xc, family=sm.families.Binomial()).fit(maxiter=200)


def pooled_or(imps, y, X, terms):
    ests, vars_ = [], []
    for d in imps:
        r = logit_fit(d, y, X)
        ests.append(r.params.reindex(terms).values); vars_.append(r.bse.reindex(terms).values ** 2)
    q, se, lo, hi, p = rubin(ests, vars_)
    return pd.DataFrame({"term": terms, "OR": np.exp(q), "lo": np.exp(lo), "hi": np.exp(hi), "p": p})


def age_band(a):
    return ((a - 18) // 5).clip(lower=0)


def compute_mds(df):
    ctrl = df[df.group == "CoY"]
    score = pd.Series(0.0, index=df.index)
    for name, items, sign in MDS_COMPONENTS:
        v = df[items].sum(axis=1, min_count=len(items))
        for s in [0, 1]:
            med = ctrl.loc[ctrl.sex == s, items].sum(axis=1).median()
            idx = df.sex == s
            pts = (v[idx] >= med) if sign > 0 else (v[idx] < med)
            score[idx] += pts.astype(float).where(v[idx].notna())
    score[df[[i for _, it, _ in MDS_COMPONENTS for i in it]].isna().any(axis=1)] = np.nan
    return score


def evalue(or_, lo, hi):
    """E-value for an OR approximating a RR (rare outcome)."""
    def ev(rr):
        rr = 1 / rr if rr < 1 else rr
        return rr + np.sqrt(rr * (rr - 1))
    ci = hi if or_ < 1 else lo
    e_ci = 1.0 if (lo <= 1 <= hi) else ev(ci)
    return ev(or_), e_ci


def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson")
    return k / n, lo, hi


def calib(y, p):
    try:
        y = np.asarray(y, float); lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
        inter = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
        return inter, slope
    except Exception:
        return np.nan, np.nan


def boot_ci(fun, y, *ps, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed); vals = []
    y = np.asarray(y); ps = [np.asarray(p) for p in ps]
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        try:
            vals.append(fun(y[i], *[p[i] for p in ps]))
        except Exception:
            pass
    return np.percentile(vals, [2.5, 97.5])



# --------------------------------------------------------------------------
# Adapter for the column format of the project database (Group = G1_Young_Cancer …)
# --------------------------------------------------------------------------
USER_MAP = {
    "Age": "age", "Interview_date": "interview_date", "Diagnosis_date": "diagnosis_date",
    "Income_below_3000": "income_cat", "Recruitment_site": "site", "Control_source": "control_source",
    "Red_meat_per_week": "red_meat", "Processed_meat_per_week": "processed_meat",
    "Charcoal_grilled_meat_per_week": "grilled_meat", "Preserved_salted_food_per_week": "preserved_salted",
    "Ultra_processed_food_per_week": "upf", "Sugary_drinks_per_week": "sugary_drinks",
    "Traditional_dishes_per_week": "traditional_dishes", "Fish_per_week": "fish", "Legumes_per_week": "legumes",
    "Vegetables_per_week": "vegetables", "Fruits_per_week": "fruits", "Whole_grains_per_week": "whole_grains",
    "Nuts_dried_fruits_per_week": "nuts", "Dairy_per_week": "dairy", "Olive_oil_per_week": "olive_oil",
    "Mint_tea_cups_per_week": "mint_tea", "Energy_kcal_per_day": "energy", "Mediterranean_Diet_Score": "mds",
    "Physical_activity_days_per_week": "phys_act", "BMI_usual": "bmi_usual", "BMI_measured": "bmi_measured",
    "Family_history_IC_first_degree": "famhx", "IBD_history_ge24m": "ibd", "Diabetes": "diabetes",
    "Hypertension": "hypertension", "Aspirin_regular_use": "aspirin", "Rectal_bleeding": "bleeding",
    "Bowel_habit_change": "bowel_change", "Abdominal_pain": "abd_pain", "Unexplained_weight_loss": "weight_loss",
    "Hemoglobin_g_dL": "hb", "Interview_months_after_dx": "months_dx_interview_raw",
    "Polyp_history_ge24m": "polyp_ge24m", "Polyp_history_reported": "polyp_reported",
    "Symptom_onset_months_before_dx": "symptom_to_dx",
}


def adapt_user_format(raw):
    """Convert the project database (cell-3 format) to the internal names used below."""
    d = raw.rename(columns=USER_MAP).copy()
    d["group"] = raw["Group"].map({"G1_Young_Cancer": "CEOIC", "G2_Young_Healthy": "CoY",
                                   "G3_Old_Cancer": "CLOIC", "G4_Old_Control": "CoO"})
    d = d[d.group.isin(["CEOIC", "CoY", "CLOIC"])].copy()
    d["sex"] = (raw.loc[d.index, "Sex"] != "Male").astype(int)                 # 1 = woman
    d["educ"] = raw.loc[d.index, "Education"].fillna("None").map(
        {"None": 0, "Primary": 1, "Secondary": 2, "University": 3})
    d["rural"] = (raw.loc[d.index, "Residence"] == "Rural").astype(int)
    d["smoking_cat"] = raw.loc[d.index, "Cigarettes_per_day"].fillna("None").map(
        {"None": 0, "1-10": 1, "11-20": 2, "21-30": 3, "31+": 4})
    d["alcohol_cat"] = raw.loc[d.index, "Alcohol_use"].map({"Never": 0, "Occasional": 1, "Regular": 2})
    d["insurance"] = raw.loc[d.index, "Insurance"].fillna("None")
    d["marital"] = raw.loc[d.index, "Marital_status"]
    # tumour characteristics (cases)
    if "Stage" in raw:
        d["stage34"] = raw.loc[d.index, "Stage"].isin(["III", "IV"]).astype(float).where(raw.loc[d.index, "Stage"].notna())
    if "MSI_status" in raw:
        d["msi_high"] = (raw.loc[d.index, "MSI_status"] == "MSI-high").astype(float).where(raw.loc[d.index, "MSI_status"].notna())
    loc = None
    for c in ["Tumor_site", "Tumor_location"]:
        if c in raw:
            loc = raw.loc[d.index, c].astype(str).str.lower() if loc is None else \
                loc.where(loc != "nan", raw.loc[d.index, c].astype(str).str.lower())
    if loc is not None:
        icd = pd.Series(np.nan, index=d.index, dtype=object)
        icd[loc.str.contains("small|duoden|jejun|ile|grêle|c17", regex=True)] = "C17"
        icd[loc.str.contains("colon|caec|cec|sigm|c18|c19", regex=True)] = "C18"
        icd[loc.str.contains("rect|c20", regex=True)] = "C20"
        icd[loc.str.contains("anal|anus|c21", regex=True)] = "C21"
        d["icd10"] = icd.where(d.group != "CoY")
    if "Histological_type" in raw:
        h = raw.loc[d.index, "Histological_type"].astype(str).str.lower()
        d["adenocarcinoma"] = np.where(d.group == "CoY", np.nan,
                                       h.str.contains("adeno|signet|mucin", regex=True).astype(float))
    return d

# ==========================================================================
# 0. data
# ==========================================================================
def load(path):
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    if "Group" in df.columns and "group" not in df.columns:
        df = adapt_user_format(df)
        log("Project database format detected (Group = G1_Young_Cancer …): columns adapted.")
    df = df.rename(columns=RENAME)
    missing = [c for c in ["group", "age", "sex", "income_cat", "interview_date"] + FOODS + SYMPT
               if c not in df.columns]
    if missing:
        sys.exit(f"Missing required columns: {missing}")
    for c in ["interview_date", "diagnosis_date"]:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    log(f"Loaded {len(df)} rows: " + ", ".join(f"{k}={v}" for k, v in df.group.value_counts().items()))
    cases = df.group.isin(["CEOIC", "CLOIC"])
    flow = {}
    for g in ["CEOIC", "CLOIC"]:
        gi = df.group == g
        flow[g] = {"included": int(gi.sum()),
                   "Anal cancer (C21)": int((gi & df.get("icd10", pd.Series("", index=df.index)).astype(str).str.startswith("C21")).sum()),
                   "Non-adenocarcinoma histology": int((gi & ~df.get("icd10", pd.Series("", index=df.index)).astype(str).str.startswith("C21")
                                                       & (df.get("adenocarcinoma", 1) == 0)).sum())}
    keep = ~cases | (~df.get("icd10", pd.Series("", index=df.index)).astype(str).str.startswith("C21")
                     & (df.get("adenocarcinoma", pd.Series(1, index=df.index)) != 0))
    df = df[keep].copy()
    for g in flow:
        flow[g]["analysed"] = int((df.group == g).sum())
    log("After excluding C21 and non-adenocarcinoma:", json.dumps(flow))
    if "mds" not in df.columns or df["mds"].isna().all():
        df["mds"] = compute_mds(df); log("MDS computed from FFQ items (MDS_COMPONENTS).")
    df["age_band"] = age_band(df["age"])
    if "months_dx_interview_raw" in df:
        df["months_dx_interview"] = df["months_dx_interview_raw"]
    elif "diagnosis_date" in df:
        df["months_dx_interview"] = (df["interview_date"] - df["diagnosis_date"]).dt.days / 30.44
    else:
        df["months_dx_interview"] = np.nan
    global CONTROL_SOURCES
    if "control_source" in df:
        CONTROL_SOURCES = sorted(df.loc[df.group == "CoY", "control_source"].dropna().unique())
        log("Control sources:", CONTROL_SOURCES)
    df["interview_year"] = df["interview_date"].dt.year + (df["interview_date"].dt.month - 1) / 12
    df["any_symptom"] = df[SYMPT].max(axis=1)
    return df, flow


# ==========================================================================
# 1. Table 2
# ==========================================================================
def table2(df):
    rows = []
    cont = ["age", "bmi_usual", "phys_act", "mds", "energy", "months_dx_interview", "symptom_to_dx"]
    binv = ["sex", "income_cat", "rural", "famhx", "ibd", "diabetes", "hypertension", "aspirin",
            "bleeding", "bowel_change", "abd_pain", "weight_loss", "stage34", "msi_high"]
    G = {g: df[df.group == g] for g in ["CEOIC", "CoY", "CLOIC"]}
    for c in cont + FOODS:
        if c not in df:
            continue
        r = {"variable": c}
        for g, d in G.items():
            x = d[c].dropna()
            r[g] = f"{x.median():.1f} [{x.quantile(.25):.1f}–{x.quantile(.75):.1f}]; {x.mean():.2f} ± {x.std():.2f}" if len(x) else "—"
        for a, b in [("CEOIC", "CoY"), ("CEOIC", "CLOIC")]:
            x, y = G[a][c].dropna(), G[b][c].dropna()
            r[f"p {a} vs {b}"] = stats.mannwhitneyu(x, y).pvalue if len(x) and len(y) else np.nan
        rows.append(r)
    for c in binv:
        if c not in df:
            continue
        r = {"variable": c + " (%)"}
        for g, d in G.items():
            x = d[c].dropna()
            r[g] = f"{100 * x.mean():.1f}" if len(x) else "—"
        for a, b in [("CEOIC", "CoY"), ("CEOIC", "CLOIC")]:
            x, y = G[a][c].dropna(), G[b][c].dropna()
            if len(x) and len(y) and (x.sum() + y.sum()) > 0:
                t = [[x.sum(), len(x) - x.sum()], [y.sum(), len(y) - y.sum()]]
                r[f"p {a} vs {b}"] = stats.chi2_contingency(t)[1]
            else:
                r[f"p {a} vs {b}"] = np.nan
        rows.append(r)
    for c in ["educ", "insurance", "marital", "smoking_cat"]:
        if c not in df:
            continue
        r = {"variable": c + " (categories)"}
        for g, d in G.items():
            r[g] = "; ".join(f"{k}: {100 * v:.1f}%" for k, v in d[c].value_counts(normalize=True).sort_index().items())
        for a, b in [("CEOIC", "CoY"), ("CEOIC", "CLOIC")]:
            t = pd.crosstab(pd.concat([G[a], G[b]])[c], pd.concat([G[a], G[b]]).group)
            r[f"p {a} vs {b}"] = stats.chi2_contingency(t)[1] if t.shape[0] > 1 else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    ok = t["p CEOIC vs CoY"].notna()
    t.loc[ok, "q (BH) CEOIC vs CoY"] = multipletests(t.loc[ok, "p CEOIC vs CoY"], method="fdr_bh")[1]
    for c in [c for c in t if c.startswith("p ") or c.startswith("q ")]:
        t[c] = t[c].map(fmt_p)
    n = df.group.value_counts()
    t.columns = [f"{c} (n={n.get(c, 0)})" if c in n.index else c for c in t.columns]
    return t


# ==========================================================================
# 2. Table 3  (aetiological, CEOIC vs CoY)
# ==========================================================================
def cc_data(df):
    d = df[df.group.isin(["CEOIC", "CoY"])].copy()
    d["y"] = (d.group == "CEOIC").astype(int)
    return d


IMP_COLS = ["y", "age_band", "sex", "income_cat"] + CONF + MEDIATOR + FOODS + ["mds", "energy", "bmi_measured",
                                                                               "interview_year"]


def adj_sets():
    base = ["age_band", "sex", "income_cat"]
    return {"Model 1": base, "Model 2": base + CONF, "Model 3": base + CONF + MEDIATOR}


def exposure_rows(imps, expo, label, adjust):
    """OR per SD, by quartile and p-trend for one exposure, one adjustment set."""
    ctrl0 = imps[0][imps[0].y == 0]
    sd = ctrl0[expo].std()
    res = {}
    # per SD
    for d in imps:
        d[expo + "_sd"] = d[expo] / sd
    r = pooled_or(imps, "y", adjust + [expo + "_sd"], [expo + "_sd"]).iloc[0]
    res["OR per SD"] = fmt_or(r.OR, r.lo, r.hi); res["p (per SD)"] = r.p
    res["OR per unit (point or time/week)"] = fmt_or(r.OR ** (1 / sd), r.lo ** (1 / sd), r.hi ** (1 / sd))
    # quartiles (cut-offs from controls); fall back to tertiles / any vs none if ties
    q = np.unique(np.quantile(ctrl0[expo], [.25, .5, .75]))
    ncat = len(q) + 1
    qnames = []
    for d in imps:
        cat = np.digitize(d[expo], q, right=True)
        for k in range(1, ncat):
            d[f"{expo}_q{k}"] = (cat == k).astype(float)
        meds = pd.Series(d[expo].values).groupby(cat).median()
        d[expo + "_trend"] = pd.Series(cat).map(meds).values
    qnames = [f"{expo}_q{k}" for k in range(1, ncat)]
    rq = pooled_or(imps, "y", adjust + qnames, qnames)
    for k, (_, rr) in enumerate(rq.iterrows(), start=2):
        res[f"Q{k} vs Q1"] = fmt_or(rr.OR, rr.lo, rr.hi)
    rt = pooled_or(imps, "y", adjust + [expo + "_trend"], [expo + "_trend"]).iloc[0]
    res["p-trend"] = rt.p
    res["SD (controls)"] = round(sd, 2)
    res["cut-offs (controls)"] = ", ".join(f"{v:g}" for v in q)
    res["_or_sd"] = (r.OR, r.lo, r.hi)
    return res


def table3(df):
    d = cc_data(df)
    cols = [c for c in IMP_COLS if c in d]
    imps = impute(d, cols)
    log(f"Case-control set: {int(d.y.sum())} CEOIC, {int((1 - d.y).sum())} CoY; MI m={M_IMP}")
    rows, evals = [], []
    for expo in ["mds"] + FOODS:
        for mname, adj in adj_sets().items():
            r = exposure_rows(imps, expo, expo, adj)
            row = {"exposure": expo, "primary": "yes" if expo in PRIMARY else "exploratory", "model": mname}
            row.update({k: v for k, v in r.items() if not k.startswith("_")})
            rows.append(row)
            if mname == "Model 2" and expo in PRIMARY:
                ev, evci = evalue(*r["_or_sd"])
                evals.append({"exposure": expo, "OR per SD (Model 2)": fmt_or(*r["_or_sd"]),
                              "E-value (point)": round(ev, 2), "E-value (CI limit)": round(evci, 2)})
    t = pd.DataFrame(rows)
    # BH across the 16 foods + MDS within Model 2
    for mname in adj_sets():
        idx = t.model == mname
        t.loc[idx, "q (BH, per SD)"] = multipletests(t.loc[idx, "p (per SD)"], method="fdr_bh")[1]
    # MDS by category
    for dd in imps:
        dd["mds_mid"] = ((dd.mds >= 4) & (dd.mds <= 5)).astype(float)
        dd["mds_high"] = (dd.mds >= 6).astype(float)
    mds_cat = []
    for mname, adj in adj_sets().items():
        r = pooled_or(imps, "y", adj + ["mds_mid", "mds_high"], ["mds_mid", "mds_high"])
        mds_cat.append({"model": mname, "MDS 4–5 vs 0–3": fmt_or(*r.iloc[0][["OR", "lo", "hi"]]),
                        "MDS 6–8 vs 0–3": fmt_or(*r.iloc[1][["OR", "lo", "hi"]])})
    for c in ["p (per SD)", "p-trend", "q (BH, per SD)"]:
        t[c] = t[c].map(fmt_p)
    # other covariates (Model 2 + all foods not needed): clinical / socio-demographic, mutually adjusted
    other = ["famhx", "ibd", "smoking_cat", "phys_act", "educ", "rural"]
    r = pooled_or(imps, "y", ["age_band", "sex", "income_cat"] + other, other)
    r["OR (95% CI)"] = [fmt_or(a, b, c) for a, b, c in zip(r.OR, r.lo, r.hi)]
    r["p"] = r.p.map(fmt_p)
    return t, pd.DataFrame(mds_cat), pd.DataFrame(evals), r[["term", "OR (95% CI)", "p"]], imps


# ==========================================================================
# 3. Table 4  sensitivity analyses
# ==========================================================================
def per_sd_m2(imps, idx, expo, extra=(), complete_case=False, raw=None):
    """Model-2 OR per SD on the rows `idx`, reusing the global imputations."""
    adj = adj_sets()["Model 2"] + list(extra)
    if complete_case:
        sub = [raw.loc[idx].dropna(subset=adj + [expo, "y"]).copy()]
    else:
        sub = [d.loc[idx].copy() for d in imps]
    sd = imps[0].loc[imps[0].y == 0, expo].std()
    for x in sub:
        x[expo + "_sd"] = x[expo] / sd
    r = pooled_or(sub, "y", adj + [expo + "_sd"], [expo + "_sd"]).iloc[0]
    return fmt_or(r.OR, r.lo, r.hi), r.p, len(sub[0]), int(sub[0].y.sum()), (r.OR, r.lo, r.hi)


def table4(df, imps):
    d0 = cc_data(df)
    ctrl = d0.y == 0
    scen = {"Main (Model 2)": d0.index}
    if "icd10" in d0:
        scen["C18–C20 only (small intestine excluded)"] = d0.index[ctrl | ~d0.icd10.astype(str).str.startswith("C17")]
    if "msi_high" in d0:
        scen["MSI-high/dMMR cases excluded"] = d0.index[ctrl | (d0.msi_high != 1)]
    if "amsterdam2" in d0:
        scen["Amsterdam II excluded"] = d0.index[d0.amsterdam2 != 1]
    if d0.months_dx_interview.notna().any():
        scen["Cases interviewed ≤ 6 months after diagnosis"] = d0.index[ctrl | (d0.months_dx_interview <= 6)]
    if "control_source" in d0:
        for src in CONTROL_SOURCES:
            scen[f"Controls: {src} only"] = d0.index[~ctrl | (d0.control_source == src)]
    rows = []
    for name, idx in scen.items():
        row = {"analysis": name}
        for expo in PRIMARY:
            o, p, n, nc, _ = per_sd_m2(imps, idx, expo)
            row[expo] = o
        row["n (cases)"] = f"{n} ({nc})"
        rows.append(row)
    # extra adjustments / complete case
    for name in ["+ calendar time of interview", "+ energy (residual method)", "Complete-case analysis"]:
        row = {"analysis": name}
        for expo in PRIMARY:
            if name.startswith("+ calendar"):
                o, p, n, nc, _ = per_sd_m2(imps, d0.index, expo, extra=["interview_year"])
            elif name.startswith("+ energy"):
                loc = [x.copy() for x in imps]
                if expo != "mds":
                    for x in loc:
                        res = sm.OLS(x[expo], sm.add_constant(x["energy"])).fit().resid
                        x[expo] = res + x[expo].mean()
                o, p, n, nc, _ = per_sd_m2(loc, d0.index, expo, extra=["energy"])
            else:
                o, p, n, nc, _ = per_sd_m2(imps, d0.index, expo, complete_case=True, raw=d0)
            row[expo] = o
        row["n (cases)"] = f"{n} ({nc})"
        rows.append(row)
    t = pd.DataFrame(rows)

    # heterogeneity across control sources (approximate Cochran Q; cases shared)
    het = []
    if "control_source" in d0:
        for expo in PRIMARY:
            b, v = [], []
            for src in CONTROL_SOURCES:
                o, p, n, nc, (orr, lo, hi) = per_sd_m2(imps, scen[f"Controls: {src} only"], expo)
                b.append(np.log(orr)); v.append(((np.log(hi) - np.log(lo)) / (2 * 1.96)) ** 2)
            b, v = np.array(b), np.array(v); okh = np.isfinite(b) & np.isfinite(v) & (v > 0)
            b, v = b[okh], v[okh]
            if len(b) < 2:
                het.append({"exposure": expo, "Cochran Q": np.nan, "p-heterogeneity": "not estimable", "note": ""}); continue
            w = 1 / v
            Q = np.sum(w * (b - np.sum(w * b) / w.sum()) ** 2)
            het.append({"exposure": expo, "Cochran Q": round(Q, 2), "p-heterogeneity": fmt_p(stats.chi2.sf(Q, len(b) - 1)),
                        "note": "approximate: the three comparisons share the same cases"})
    # tertiles of diagnosis-to-interview delay (each tertile of cases vs all controls)
    delay = []
    if d0.months_dx_interview.notna().any():
        cases = d0[d0.y == 1]
        cuts = cases.months_dx_interview.quantile([1 / 3, 2 / 3]).values
        tert = np.digitize(cases.months_dx_interview, cuts)
        for expo in PRIMARY:
            row = {"exposure": expo}; bs = []
            for k in range(3):
                ids = cases.index[tert == k]
                o, p, n, nc, (orr, lo, hi) = per_sd_m2(imps, d0.index[ctrl | d0.index.isin(ids)], expo)
                rng_ = cases.months_dx_interview[ids]
                row[f"T{k + 1} ({rng_.min():.0f}–{rng_.max():.0f} months)"] = o
                bs.append(np.log(orr) if np.isfinite(orr) and orr > 0 else np.nan)
            ok_ = np.isfinite(bs)
            row["slope of log-OR per tertile"] = round(np.polyfit(np.arange(3)[ok_], np.array(bs)[ok_], 1)[0], 3) if ok_.sum() >= 2 else np.nan
            delay.append(row)
    return t, pd.DataFrame(het), pd.DataFrame(delay)


def polyp_bias(df):
    """Illustration of detection bias: polyps diagnosed >= 24 months before vs any report."""
    d = cc_data(df); rows = []
    for c, lab in [("polyp_ge24m", "Polyps diagnosed ≥ 24 months before index date"),
                   ("polyp_reported", "Polyps, any report (incl. diagnostic work-up)")]:
        if c not in d:
            continue
        x = d.dropna(subset=[c] + adj_sets()["Model 2"])
        n_exp = int(x.loc[x.y == 1, c].sum()), int(x.loc[x.y == 0, c].sum())
        row = {"variable": lab, "exposed cases": n_exp[0], "exposed controls": n_exp[1]}
        if min(n_exp) >= 5:
            r = logit_fit(x, "y", adj_sets()["Model 2"] + [c])
            o, (l, h) = np.exp(r.params[c]), np.exp(r.conf_int().loc[c].values)
            row["aOR (Model 2)"] = fmt_or(o, l, h)
        else:
            row["aOR (Model 2)"] = "not estimable (< 5 exposed in a group)"
        rows.append(row)
    return pd.DataFrame(rows)


# ==========================================================================
# 5. prediction
# ==========================================================================
def model_sets(df):
    diet = FOODS + ["energy"]
    M0 = diet + LIFE + RISK
    M1 = list(SYMPT)
    M2 = M1 + RISK
    M3 = M2 + diet + LIFE
    sets = {"M0 diet+lifestyle+risk (no symptoms)": M0,
            "M1 symptoms only": M1,
            "M2 = M1 + risk factors": M2,
            "M3 = M2 + diet/lifestyle (full, primary)": M3}
    if USE_HB_SENSITIVITY and "hb" in df and df["hb"].notna().any():
        sets["M3 + haemoglobin (sensitivity)"] = M3 + ["hb"]
    return sets


def enet():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("lr", LogisticRegressionCV(penalty="elasticnet", solver="saga",
                                                 l1_ratios=[.1, .5, .9, 1.0], Cs=10, tol=1e-3,
                                                 cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                                 scoring="roc_auc", max_iter=2000, random_state=SEED))])


def cutoffs_from_oof(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    rule_out = thr[np.argmax(tpr >= 0.95)]
    youden = thr[np.argmax(tpr - fpr)]
    ok = np.where(1 - fpr >= 0.95)[0]
    rule_in = thr[ok[-1]] if len(ok) else thr[1]
    return {"rule-out (Se ≥ 95%)": rule_out, "Youden": youden, "rule-in (Sp ≥ 95%)": rule_in}


def prediction(df, out):
    d = cc_data(df)
    dev = d[d.interview_date <= pd.Timestamp(TEST_CUTOFF)]
    test = d[d.interview_date > pd.Timestamp(TEST_CUTOFF)]
    log(f"Development: n={len(dev)} ({int(dev.y.sum())} cases); test: n={len(test)} ({int(test.y.sum())} cases)")
    sets = model_sets(df)
    perf, preds, fitted = [], {}, {}
    for name, X in sets.items():
        X = [c for c in X if c in d]
        m = enet().fit(dev[X], dev.y)
        p = m.predict_proba(test[X])[:, 1]
        preds[name], fitted[name] = p, (m, X)
        y = test.y.values
        auc = roc_auc_score(y, p); auc_ci = boot_ci(roc_auc_score, y, p)
        br = brier_score_loss(y, p); br0 = y.mean() * (1 - y.mean())
        ci_, sl = calib(y, p)
        lr = m.named_steps["lr"]
        perf.append({"model": name, "n predictors": len(X),
                     "retained": int((np.abs(lr.coef_[0]) > 1e-8).sum()),
                     "AUC (95% CI)": f"{auc:.3f} ({auc_ci[0]:.3f}–{auc_ci[1]:.3f})",
                     "Brier": round(br, 3), "scaled Brier": round(1 - br / br0, 3),
                     "calibration intercept": round(ci_, 2), "calibration slope": round(sl, 2),
                     "alpha (l1_ratio)": lr.l1_ratio_[0], "C (=1/lambda)": round(float(lr.C_[0]), 4)})
    perf = pd.DataFrame(perf)
    y = test.y.values
    delta = []
    names = list(sets)
    pairs = [(names[1], names[2]), (names[2], names[3]), (names[1], names[3]), (names[0], names[3])]
    if len(names) > 4:
        pairs.append((names[3], names[4]))
    for a, b in pairs:
        da = roc_auc_score(y, preds[b]) - roc_auc_score(y, preds[a])
        ci = boot_ci(lambda yy, pa, pb: roc_auc_score(yy, pb) - roc_auc_score(yy, pa), y, preds[a], preds[b])
        delta.append({"comparison": f"{b}  vs  {a}", "ΔAUC (95% CI)": f"{da:+.3f} ({ci[0]:+.3f} to {ci[1]:+.3f})"})
    delta = pd.DataFrame(delta)

    # primary model = M3 full
    prim = names[3]
    m, X = fitted[prim]
    oof = cross_val_predict(enet(), dev[X], dev.y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    cuts = cutoffs_from_oof(dev.y.values, oof)
    p = preds[prim]
    cls, pv = [], []
    for cname, c in cuts.items():
        tp = int(((p >= c) & (y == 1)).sum()); fn = int(((p < c) & (y == 1)).sum())
        tn = int(((p < c) & (y == 0)).sum()); fp = int(((p >= c) & (y == 0)).sum())
        se, sel, seh = wilson(tp, tp + fn); sp, spl, sph = wilson(tn, tn + fp)
        lrp = se / (1 - sp) if sp < 1 else np.inf; lrn = (1 - se) / sp
        cls.append({"cut-off": f"{cname}: {c:.3f}", "Sensitivity": f"{se:.3f} ({sel:.3f}–{seh:.3f})",
                    "Specificity": f"{sp:.3f} ({spl:.3f}–{sph:.3f})", "LR+": round(lrp, 2), "LR−": round(lrn, 2)})
        for prev in PREVALENCES:
            ppv = se * prev / (se * prev + (1 - sp) * (1 - prev))
            npv = sp * (1 - prev) / (sp * (1 - prev) + (1 - se) * prev)
            pv.append({"cut-off": cname, "prevalence": f"{prev:.1%}", "PPV": f"{ppv:.1%}", "NPV": f"{npv:.2%}"})
    cls, pv = pd.DataFrame(cls), pd.DataFrame(pv)

    # symptomatic-control subgroup, out-of-fold on the whole case-control set
    sub, oof_full = [], {}
    for name in [names[0], names[1], names[3]]:
        Xn = fitted[name][1]
        oo = cross_val_predict(enet(), d[Xn], d.y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                               method="predict_proba")[:, 1]
        oof_full[name] = oo
        allc = roc_auc_score(d.y, oo)
        mask = (d.y == 1) | ((d.y == 0) & (d.any_symptom == 1))
        symp = roc_auc_score(d.y[mask], oo[mask.values])
        ci = boot_ci(roc_auc_score, d.y[mask].values, oo[mask.values])
        sub.append({"model": name, "AUC cases vs all controls (OOF)": round(allc, 3),
                    "AUC cases vs symptomatic controls (OOF)": f"{symp:.3f} ({ci[0]:.3f}–{ci[1]:.3f})",
                    "n symptomatic controls": int(((d.y == 0) & (d.any_symptom == 1)).sum())})
    sub = pd.DataFrame(sub)

    # decision curves at assumed prevalence (recalibrated intercept)
    pi_s = dev.y.mean()
    thr = np.linspace(0.002, 0.10, 99)
    plt.figure(figsize=(6, 4.2))
    for name, col in [(names[1], "#1f77b4"), (prim, "#d62728")]:
        pp = np.clip(preds[name], 1e-6, 1 - 1e-6)
        lp = np.log(pp / (1 - pp)) + np.log(DCA_PREVALENCE / (1 - DCA_PREVALENCE)) - np.log(pi_s / (1 - pi_s))
        pr = 1 / (1 + np.exp(-lp))
        nb = [((pr >= t) & (y == 1)).sum() / (y == 1).sum() * DCA_PREVALENCE
              - ((pr >= t) & (y == 0)).sum() / (y == 0).sum() * (1 - DCA_PREVALENCE) * t / (1 - t) for t in thr]
        plt.plot(thr, nb, label=name, color=col)
    plt.plot(thr, DCA_PREVALENCE - (1 - DCA_PREVALENCE) * thr / (1 - thr), color="grey", lw=0.8, label="Investigate all")
    plt.axhline(0, color="black", lw=0.8, label="Investigate none")
    plt.ylim(-0.002, DCA_PREVALENCE * 1.1); plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
    plt.title(f"Decision curves, assumed prevalence {DCA_PREVALENCE:.0%}", fontsize=10)
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(out, "fig_dca.png"), dpi=300); plt.close()

    # ---- Figure 2: ROC (M1 vs M3) + calibration of the primary model, test set
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.3))
    for name, col in [(names[1], "#1f77b4"), (prim, "#d62728")]:
        f_, t_, _ = roc_curve(y, preds[name])
        a1.plot(f_, t_, color=col, label=f"{name} (AUC {roc_auc_score(y, preds[name]):.3f})")
    a1.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
    a1.set_xlabel("1 − specificity"); a1.set_ylabel("Sensitivity"); a1.legend(fontsize=7, loc="lower right")
    a1.set_title("A  ROC curves, temporal test set", loc="left", fontsize=10)
    qs = pd.qcut(p, 10, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y, "q": qs}).groupby("q", observed=True).mean()
    a2.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
    a2.scatter(g.p, g.y, color="#d62728", s=18, label="Deciles of predicted risk")
    lo = sm.nonparametric.lowess(np.asarray(y, float), p, frac=0.6, it=0)
    a2.plot(lo[:, 0], lo[:, 1], color="#d62728", label="LOESS")
    ci_, sl = calib(y, p)
    a2.text(0.03, 0.92, f"Intercept {ci_:.2f}\nSlope {sl:.2f}", transform=a2.transAxes, fontsize=8, va="top")
    a2.set_xlabel("Predicted probability"); a2.set_ylabel("Observed proportion"); a2.legend(fontsize=7, loc="lower right")
    a2.set_title("B  Calibration of the primary model", loc="left", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(out, "Figure2_ROC_calibration.png"), dpi=300, bbox_inches="tight"); plt.close()

    # ---- cross-validated AUC on the development set (5-fold x 3 repeats)
    cv_aucs = []
    for rep in range(3):
        oo = cross_val_predict(enet(), dev[X], dev.y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED + rep),
                               method="predict_proba")[:, 1]
        cv_aucs.append(roc_auc_score(dev.y, oo))
    perf.loc[perf.model == prim, "CV AUC development (mean ± SD, 5-fold x 3)"] = \
        f"{np.mean(cv_aucs):.3f} ± {np.std(cv_aucs):.3f}"

    # ---- leave-one-site-out validation (primary model)
    loso = []
    if "site" in d:
        for site in sorted(d.site.dropna().unique()):
            te = d.site == site
            if d.loc[te, "y"].sum() < 10 or (1 - d.loc[te, "y"]).sum() < 10:
                loso.append({"site": site, "n": int(te.sum()), "cases": int(d.loc[te, "y"].sum()),
                             "AUC": "not estimable (< 10 cases or controls)", "calibration intercept": ""})
                continue
            mm = enet().fit(d.loc[~te, X], d.loc[~te, "y"])
            pp = mm.predict_proba(d.loc[te, X])[:, 1]
            ci_s, _ = calib(d.loc[te, "y"].values, pp)
            loso.append({"site": site, "n": int(te.sum()), "cases": int(d.loc[te, "y"].sum()),
                         "AUC": round(roc_auc_score(d.loc[te, "y"], pp), 3), "calibration intercept": round(ci_s, 2)})
    loso = pd.DataFrame(loso)

    # ---- Figure 3: standardised coefficients, 200 bootstrap refits with the tuned hyper-parameters
    from sklearn.linear_model import LogisticRegression
    lr_t = m.named_steps["lr"]
    rng = np.random.default_rng(SEED); B = []
    for _ in range(200):
        i = rng.integers(0, len(dev), len(dev))
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("lr", LogisticRegression(penalty="elasticnet", solver="saga", C=float(lr_t.C_[0]),
                                                   l1_ratio=float(lr_t.l1_ratio_[0]), max_iter=2000, tol=1e-3))])
        pipe.fit(dev[X].iloc[i], dev.y.iloc[i]); B.append(pipe.named_steps["lr"].coef_[0])
    B = np.array(B)
    coef = pd.DataFrame({"predictor": X, "OR per SD (penalised)": np.exp(lr_t.coef_[0]),
                         "lo": np.exp(np.percentile(B, 2.5, 0)), "hi": np.exp(np.percentile(B, 97.5, 0)),
                         "selection frequency (%)": (np.abs(B) > 1e-8).mean(0) * 100})
    coef = coef[np.abs(lr_t.coef_[0]) > 1e-8].sort_values("OR per SD (penalised)")
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(coef) + 1.2))
    yy = np.arange(len(coef))
    ax.errorbar(coef["OR per SD (penalised)"], yy, xerr=[coef["OR per SD (penalised)"] - coef.lo,
                coef.hi - coef["OR per SD (penalised)"]], fmt="s", color="#333333", ms=4, lw=0.9)
    ax.axvline(1, color="grey", lw=0.8); ax.set_xscale("log")
    ax.set_yticks(yy); ax.set_yticklabels([f"{PRETTY.get(r.predictor, r.predictor)}  ({r['selection frequency (%)']:.0f}%)" for _, r in coef.iterrows()], fontsize=7)
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3]); ax.set_xticklabels(["0.5", "0.75", "1", "1.5", "2", "3"])
    ax.set_xlabel("Penalised odds ratio per 1 SD (95% bootstrap CI); selection frequency in brackets")
    plt.tight_layout(); plt.savefig(os.path.join(out, "Figure3_coefficients.png"), dpi=300, bbox_inches="tight"); plt.close()
    coef["predictor"] = coef["predictor"].map(lambda v: PRETTY.get(v, v))
    coef = coef.round(3)

    # ---- Figure 4: observed proportion of cases by quintile of out-of-fold risk (whole case-control set)
    oo = oof_full[prim]
    qd = pd.DataFrame({"y": d.y.values, "q": pd.qcut(oo, 5, labels=False, duplicates="drop")})
    quint = []
    for k, gq in qd.groupby("q"):
        pr_, lo_, hi_ = wilson(int(gq.y.sum()), len(gq))
        quint.append({"quintile": int(k) + 1, "n": len(gq), "cases": int(gq.y.sum()),
                      "observed %": round(100 * pr_, 1), "lo": round(100 * lo_, 1), "hi": round(100 * hi_, 1)})
    quint = pd.DataFrame(quint)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.errorbar(quint.quintile, quint["observed %"], yerr=[quint["observed %"] - quint.lo, quint.hi - quint["observed %"]],
                fmt="o-", color="#d62728", capsize=3)
    ax.axhline(100 * d.y.mean(), ls="--", color="grey", lw=0.8)
    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xlabel("Quintile of out-of-fold predicted risk"); ax.set_ylabel("Observed proportion of CEOIC (%)")
    plt.tight_layout(); plt.savefig(os.path.join(out, "Figure4_risk_quintiles.png"), dpi=300, bbox_inches="tight"); plt.close()

    # full equation (unstandardised)
    lr = m.named_steps["lr"]; sc = m.named_steps["sc"]
    b = lr.coef_[0] / sc.scale_
    b0 = lr.intercept_[0] - np.sum(lr.coef_[0] * sc.mean_ / sc.scale_)
    eq = pd.DataFrame({"term": ["(intercept)"] + X, "coefficient (original units)": np.r_[b0, b],
                       "standardised coefficient": np.r_[np.nan, lr.coef_[0]],
                       "median used for missing values": np.r_[np.nan, m.named_steps["imp"].statistics_]})
    eq = eq[(eq.term == "(intercept)") | (eq["standardised coefficient"].abs() > 1e-8)]
    eq.to_csv(os.path.join(out, "model_equation.csv"), index=False)
    eq_note = pd.DataFrame({"item": ["formula", "recalibration", "development case proportion"],
                            "value": ["P = 1 / (1 + exp(-LP)),  LP = intercept + sum(coef_i * x_i)",
                                      "LP_new = LP + ln(pi/(1-pi)) - ln(pi_dev/(1-pi_dev))", round(pi_s, 4)]})

    # dev vs test characteristics
    comp = []
    for c in SYMPT + ["stage34", "famhx", "upf", "mds"]:
        if c not in d:
            continue
        for grp, lab in [(1, "cases"), (0, "controls")]:
            a, bb = dev.loc[dev.y == grp, c].dropna(), test.loc[test.y == grp, c].dropna()
            if len(a) == 0 or len(bb) == 0:
                continue
            comp.append({"variable": c, "group": lab, "development": round(a.mean(), 3), "test": round(bb.mean(), 3),
                         "p": fmt_p(stats.mannwhitneyu(a, bb).pvalue)})
    if "control_source" in d:
        for s in CONTROL_SOURCES:
            comp.append({"variable": f"control source = {s}", "group": "controls",
                         "development": round((dev.loc[dev.y == 0, "control_source"] == s).mean(), 3),
                         "test": round((test.loc[test.y == 0, "control_source"] == s).mean(), 3), "p": ""})
    counts = {"dev_n": int(len(dev)), "dev_cases": int(dev.y.sum()), "test_n": int(len(test)),
              "test_cases": int(test.y.sum())}
    return perf, delta, cls, pv, sub, eq, eq_note, pd.DataFrame(comp), loso, coef, quint, counts


# ==========================================================================
# ==========================================================================
# Figure 1 — STROBE flow diagram
# ==========================================================================
PH = "[   ]"


def flow_fmt(n):
    return PH if n is None else f"{n:,}"


def flow_total(d):
    vals = list(d.values())
    return None if any(v is None for v in vals) else sum(vals)


def box(ax, x, y, w, lines, bold_first=True, fc="white", ec="#333333", fs=8.2, lh=0.0145):
    h = lh * len(lines) + 0.014
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0.002,rounding_size=0.004",
                                fc=fc, ec=ec, lw=0.9))
    for i, t in enumerate(lines):
        ax.text(x + 0.008, y - 0.009 - i * lh, t, ha="left", va="top", fontsize=fs,
                fontweight="bold" if (i == 0 and bold_first) else "normal", family="DejaVu Sans")
    return y - h  # bottom


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333333", shrinkA=0, shrinkB=0))


def draw_flow(cfg_path, out_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    fig = plt.figure(figsize=(11, 13))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    groups = cfg["groups"]
    keys = list(groups.keys())
    colw, xs = 0.30, [0.025, 0.35, 0.675]

    # --- top box
    top_lines = [f"Individuals screened, {cfg['period']}",
                 f"n = {flow_fmt(cfg['screened_total'])}"]
    inc_total = flow_total({k: groups[k]["included"] for k in keys})
    if cfg["screened_total"] is not None and inc_total is not None:
        top_lines.append(f"Not included: n = {cfg['screened_total'] - inc_total:,} (details by group below)")
    yb = box(ax, 0.30, 0.985, 0.40, top_lines, fc="#eef3f8")
    y_split = yb - 0.02
    ax.plot([0.5, 0.5], [yb, y_split], color="#333333", lw=0.9)
    ax.plot([xs[0] + colw / 2, xs[2] + colw / 2], [y_split, y_split], color="#333333", lw=0.9)

    bottoms = []
    for k, x in zip(keys, xs):
        g = groups[k]
        cx = x + colw / 2
        arrow(ax, cx, y_split, cx, y_split - 0.02)
        y = box(ax, x, y_split - 0.02, colw, [g["label"], f"Assessed for eligibility: n = {flow_fmt(g['assessed'])}"],
                fc="#eef3f8")
        # exclusions
        ex = g["excluded"]
        arrow(ax, cx, y, cx, y - 0.018)
        import textwrap
        lines = [f"Excluded: n = {flow_fmt(flow_total(ex))}"]
        for r_, v_ in ex.items():
            r_ = r_[:1].upper() + r_[1:]
            wr = textwrap.wrap(f"{r_}: n = {flow_fmt(v_)}", 46)
            lines += ["  • " + wr[0]] + ["     " + t for t in wr[1:]]
        y = box(ax, x, y - 0.018, colw, lines, fc="#fbf3ee")
        arrow(ax, cx, y, cx, y - 0.018)
        y = box(ax, x, y - 0.018, colw, ["Included", f"n = {flow_fmt(g['included'])}"])
        if g.get("post_excluded") and flow_total(g["post_excluded"]) != 0:
            pe = g["post_excluded"]
            arrow(ax, cx, y, cx, y - 0.018)
            lines = [f"Excluded after pathology review: n = {flow_fmt(flow_total(pe))}"] + \
                    [f"  • {r}: n = {flow_fmt(v)}" for r, v in pe.items()]
            y = box(ax, x, y - 0.018, colw, lines, fc="#fbf3ee")
        analysed = g.get("analysed")
        if analysed is None and g["included"] is not None:
            pe_t = flow_total(g.get("post_excluded") or {}) if g.get("post_excluded") else 0
            analysed = None if pe_t is None else g["included"] - pe_t
        arrow(ax, cx, y, cx, y - 0.018)
        lines = ["Analysed", f"n = {flow_fmt(analysed)}"]
        if k == "CoY" and cfg.get("control_sources"):
            lines += [f"  • {s}: n = {flow_fmt(v)}" for s, v in cfg["control_sources"].items()]
        y = box(ax, x, y - 0.018, colw, lines, fc="#eaf5ec")
        bottoms.append((cx, y))

    # --- analyses boxes
    ymin = min(b[1] for b in bottoms) - 0.03
    # case-control + prediction (CEOIC + CoY)
    p = cfg["prediction"]
    (c0, b0), (c1, b1), (c2, b2) = bottoms
    ax.plot([c0, c0], [b0, ymin], color="#333333", lw=0.9)
    ax.plot([c1, c1], [b1, ymin], color="#333333", lw=0.9)
    ax.plot([c0, c1], [ymin, ymin], color="#333333", lw=0.9)
    mid = (c0 + c1) / 2
    arrow(ax, mid, ymin, mid, ymin - 0.018)
    lines = ["Aetiological analysis and detection model (early-onset cases vs young controls)",
             f"Development set (interviewed ≤ {p['cutoff_date']}): n = {flow_fmt(p['dev_n'])} ({flow_fmt(p['dev_cases'])} cases)",
             f"Temporal test set (interviewed after {p['cutoff_date']}): n = {flow_fmt(p['test_n'])} ({flow_fmt(p['test_cases'])} cases)"]
    box(ax, 0.025, ymin - 0.018, 0.625, lines, fc="#eef3f8", fs=8.2)
    ax.plot([c2, c2], [b2, ymin - 0.018], color="#333333", lw=0.9)
    box(ax, 0.675, ymin - 0.018, colw, ["Case–case comparison", "(exploratory, Supplementary)"], fc="#eef3f8")
    y_end = ymin - 0.018 - (0.0145 * 3 + 0.014) - 0.01
    from matplotlib.transforms import Bbox
    W, H = fig.get_size_inches()
    bb = Bbox([[0, y_end * H], [W, H]])
    fig.savefig(out_path, dpi=300, bbox_inches=bb, facecolor="white")
    if out_path.endswith(".png"):
        fig.savefig(out_path[:-4] + ".pdf", bbox_inches=bb, facecolor="white")
    print("written", out_path)



def make_flow(flow_csv, flow, df, counts, out):
    """Build Figure 1 from the project flow file (rows = steps, columns = CEOIC/CoY/CLOIC)."""
    fc = pd.read_csv(flow_csv, index_col=0)
    groups = {}
    labels = {"CEOIC": "Early-onset cases (18–45 years)", "CoY": "Young controls (18–45 years)",
              "CLOIC": "Late-onset cases (> 45 years)"}
    screened = 0
    for g in ["CEOIC", "CoY", "CLOIC"]:
        if g not in fc.columns:
            continue
        f = fc[g]
        scr = int(f.get("Approached / screened", np.nan)) if "Approached / screened" in f else None
        screened += scr or 0
        excl = {i.replace("Not eligible: ", "").replace("Excluded: ", ""): int(v) for i, v in f.items()
                if i not in ("Approached / screened", "Analysed") and pd.notna(v) and v > 0}
        inc = int(f["Analysed"]) if "Analysed" in f else None
        pe = None
        if g in flow:
            pe = {"Anal cancer (C21)": flow[g]["Anal cancer (C21)"],
                  "Non-adenocarcinoma histology": flow[g]["Non-adenocarcinoma histology"]}
        groups[g] = {"label": labels[g], "assessed": scr, "excluded": excl, "included": inc,
                     "post_excluded": pe, "analysed": int((df.group == g).sum())}
    cfg = {"period": "September 2024 – September 2026", "screened_total": screened, "groups": groups,
           "control_sources": {s: int(((df.group == "CoY") & (df.control_source == s)).sum()) for s in CONTROL_SOURCES},
           "prediction": {"cutoff_date": f"{pd.Timestamp(TEST_CUTOFF).day} {pd.Timestamp(TEST_CUTOFF).strftime('%B %Y')}", **counts}}
    jp = os.path.join(out, "flow_counts_auto.json")
    json.dump(cfg, open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    draw_flow(jp, os.path.join(out, "Figure1_flow.png"))


def safe(label, fun, *a, n_out=1):
    """Run one section; if it fails, log the error and continue with the others."""
    import traceback
    try:
        return fun(*a)
    except Exception as e:
        log(f"!! Section '{label}' failed: {type(e).__name__}: {e}")
        LOG.append(traceback.format_exc())
        empty = pd.DataFrame({"error": [f"{type(e).__name__}: {e}"]})
        return empty if n_out == 1 else tuple([empty] * n_out)


def main(path, out="results", flow_csv=None):
    os.makedirs(out, exist_ok=True)
    df, flow = load(path)
    t2 = safe("Table 2", table2, df)
    res3 = safe("Table 3", table3, df, n_out=5)
    t3, t3mds, ev, t3other, imps = res3
    if isinstance(imps, pd.DataFrame):          # Table 3 failed -> rebuild imputations for Table 4
        d_ = cc_data(df); imps = impute(d_, [c for c in IMP_COLS if c in d_])
    t4, het, delay = safe("Table 4", table4, df, imps, n_out=3)
    pb = safe("Polyp detection bias", polyp_bias, df)
    res5 = safe("Prediction", prediction, df, out, n_out=12)
    perf, delta, cls, pv, sub, eq, eq_note, comp, loso, coef, quint, counts = res5
    if isinstance(counts, pd.DataFrame):
        counts = {"dev_n": None, "dev_cases": None, "test_n": None, "test_cases": None}
    with pd.ExcelWriter(os.path.join(out, "results.xlsx")) as w:
        pd.DataFrame([{"group": g, **v} for g, v in flow.items()]).to_excel(w, sheet_name="Flow (pathology)", index=False)
        pd.DataFrame([counts]).to_excel(w, sheet_name="Flow (dev-test)", index=False)
        for name, t in [("Table2 characteristics", t2), ("Table3 diet (M1-M3)", t3), ("Table3b MDS categories", t3mds),
                        ("Table3c other factors", t3other), ("E-values", ev), ("Table4 sensitivity", t4),
                        ("Table4b control-source het", het), ("Table4c recall delay", delay),
                        ("Table4d polyp detection bias", pb), ("Table5 nested models", perf),
                        ("Table5b delta AUC", delta), ("Table5c cut-offs", cls), ("Table6 PPV-NPV by prevalence", pv),
                        ("Table5e symptomatic controls", sub), ("Leave-one-site-out", loso),
                        ("Fig3 coefficients", coef), ("Fig4 risk quintiles", quint), ("S model equation", eq),
                        ("S equation notes", eq_note), ("S dev vs test", comp)]:
            t.to_excel(w, sheet_name=name, index=False)
    if flow_csv:
        safe("Flow diagram", make_flow, flow_csv, flow, df, counts, out)
    open(os.path.join(out, "log.txt"), "w", encoding="utf-8").write("\n".join(LOG))
    log("Done ->", out)
    errs = [l for l in LOG if l.startswith("!! Section")]
    if errs:
        print("\n".join(errs)); print("Details in log.txt — send it to me.")


def show_outputs(out):
    """Display the figures inline when run in Colab / Jupyter."""
    try:
        from IPython.display import Image, display
        for f in ["Figure1_flow.png", "Figure2_ROC_calibration.png", "Figure3_coefficients.png",
                  "Figure4_risk_quintiles.png", "fig_dca.png"]:
            p = os.path.join(out, f)
            if os.path.exists(p):
                print(f); display(Image(filename=p, width=700))
    except Exception:
        pass


if __name__ == "__main__":
    if "SIM" in os.path.basename(DATA_FILE).upper():
        print("⚠️  WARNING: the data file name contains 'SIM' (simulated data). "
              "Use the REAL study database for the manuscript.")
    main(DATA_FILE, OUT_DIR, FLOW_FILE if FLOW_FILE and os.path.exists(FLOW_FILE) else None)
    show_outputs(OUT_DIR)
