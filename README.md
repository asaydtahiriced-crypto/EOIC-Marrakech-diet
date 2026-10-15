# Early-onset intestinal cancer in low-income young adults — analysis code

Analysis code for the article:

> Ellouad H, Belbachir A, Assarag B, Doukkali A, Saydtahiri A. *Dietary and lifestyle factors and early-onset intestinal cancer among low-income young adults in Marrakech-Safi, Morocco: a case–control study.* [Journal, year, DOI — to be completed after publication]

The script reproduces all tables and figures of the article:

| Output | Content |
|---|---|
| Table 2 | Characteristics of participants by group |
| Table 3, Table S2 | Dietary exposures and early-onset intestinal cancer (Models 1–3; per SD, per unit, quartiles, p-trend, false-discovery-rate correction) |
| Table 4 | Sensitivity analyses (complete case, energy, calendar time, MSI-high exclusion, recall delay, control source with heterogeneity test, E-values) |
| Tables 5–6, Table S1, Table S3 | Nested elastic-net detection models (M0–M3), temporal validation, symptomatic controls, cut-offs, predictive values by prevalence, leave-one-site-out validation, full model equation |
| Figures 1–4, Figure S2 | Flow diagram, ROC/calibration, coefficients, risk quintiles, decision curves |

## Data

The individual participant data are **not included** in this repository, to protect participant confidentiality. They are available from the corresponding author upon reasonable request, subject to approval by the ethics committee of the Clinical Research Center of the Mohammed VI University Hospital of Marrakech.

The script expects one row per participant with the following main columns:
`Group` (G1_Young_Cancer / G2_Young_Healthy / G3_Old_Cancer), `Age`, `Sex`, `Income_below_3000`, `Education`, `Residence`, `Insurance`, `Marital_status`, `Recruitment_site`, `Control_source`, `Interview_date`, `Diagnosis_date`, `Interview_months_after_dx`, `Tumor_location`, `Histological_type`, `Stage`, `MSI_status`, the 16 food-frequency items (`..._per_week`), `Energy_kcal_per_day`, `Mediterranean_Diet_Score`, `Cigarettes_per_day`, `Alcohol_use`, `Physical_activity_days_per_week`, `BMI_usual`, `Family_history_IC_first_degree`, `IBD_history_ge24m`, `Polyp_history_ge24m`, `Polyp_history_reported`, `Rectal_bleeding`, `Bowel_habit_change`, `Abdominal_pain`, `Unexplained_weight_loss`.

## How to run

1. Install the requirements: `pip install -r requirements.txt`
2. Set `DATA_FILE`, `FLOW_FILE` and `OUT_DIR` in the CONFIGURATION block at the top of `analyse_EOIC_definitive.py`.
3. Run `python analyse_EOIC_definitive.py` (or `%run analyse_EOIC_definitive.py` in Jupyter / Google Colab).

All results are written to `OUT_DIR` (`results.xlsx`, figures, `model_equation.csv`, `log.txt`). Run time: about 2 minutes. Random seed: 2026.

## Using the detection model

The full model equation is given in `model_equation.csv` (and Supplementary Table S1 of the article):
P = 1 / (1 + exp(−LP)), with LP = intercept + Σ coefficient × value.
The intercept reflects the case proportion of the development set (0.17) and must be recalibrated to the prevalence of the target population before any use. The model is not validated for clinical use.

## Licence and citation

Code released under the MIT licence (see `LICENSE`). Please cite the article and this repository (see `CITATION.cff`).
