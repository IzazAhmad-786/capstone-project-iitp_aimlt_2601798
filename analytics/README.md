# /analytics — Zepto Analytics Pipeline (Module 2)

Titanic dataset, loaded once (`sns.load_dataset('titanic')`), split across two notebooks. Everything
downstream builds on the same cleaned data.

## How to run

```bash
pip install -r pandas
pip install -r numpy
pip install -r seaborn
pip install -r matplotlib
pip install -r scikit-learn
pip install -r imbalanced-learn
pip install -r joblib
pip install -r jupyter
pip install -r nbformat

jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb
```

Run `01_eda.ipynb` first — only place `sns.load_dataset` is called. It saves the raw data to
`titanic.csv` right after loading, so `02_modeling.ipynb` just reads that CSV instead of hitting
the network again.

## Files

| File | What it is |
|---|---|
| `01_eda.ipynb` | Load, profile, clean, EDA |
| `02_modeling.ipynb` | Split, preprocessing, 3 classifiers, imbalance handling, tuning, regression, saved model |
| `titanic.csv` | Raw dataset, saved once before any cleaning |
| `titanic_full_pipeline.joblib` | Fitted preprocessing + tuned Random Forest, bundled |
| `charts/` | PNGs referenced below |

(Write-ups live here instead of in notebook markdown — kept the notebooks to code + output only.)

---

## Part A — Profiling, cleaning, data story

### Task 1 — Load & profile

Shape: `(891, 15)`. Saved raw to `titanic.csv` before touching anything.

**Missing values:**

| Column | Missing | % |
|---|---|---|
| `deck` | 688 | 77.22% |
| `age` | 177 | 19.87% |
| `embarked` | 2 | 0.22% |
| `embark_town` | 2 | 0.22% |

### Task 2 — Missing-value handling

Rule: <5% → drop rows, 5–30% → impute, >30% → drop column.

- `embarked` (0.22%) — dropped 2 rows.
- `embark_town` (0.22%, same rows) — cleared automatically, it's just `embarked` spelled out.
- `age` (19.87%) — median-imputed (median holds up better than mean given the right-skew).
- `deck` (77.22%) — dropped the column, imputing 3/4 of it would just be making up data.

Result: 889 rows, 13 columns.

### Task 3 — `age` / `fare` outliers & skew

`charts/01_univariate_age_fare.png`

| Column | IQR outliers | Bounds |
|---|---|---|
| `age` | 65 | [2.50, 54.50] |
| `fare` | 114 | [-26.76, 65.66] |

`fare` mean/median/mode: **32.10 / 14.45 / 8.05** → mean > median > mode = right-skewed (long tail
of expensive tickets).

### Task 4 — Bivariate + correlation

Survival by sex: female 0.740, male 0.189
Survival by class: 1st 0.626, 2nd 0.473, 3rd 0.242
Survival by sex+class: female-1st 0.967, male-1st 0.369, female-2nd 0.921, male-2nd 0.157,
female-3rd 0.500, male-3rd 0.135

`charts/02_correlation_heatmap.png` — 6 numeric columns only (`survived, pclass, age, sibsp,
parch, fare`); `adult_male`/`alone` skipped since they're derived from columns already in the
matrix.

Top 2 correlations: **`pclass`/`fare` (-0.55)** — fare basically sets class, expected. **`sibsp`/
`parch` (+0.41)** — both are proxying for "traveling with family."

### Task 5 — Data story (4 charts)

Story: sex mattered most, class/fare second, age only mattered much for kids.

1. **Survival by class & sex** (`03_survival_by_class_sex.png`) — drops from women→men in every
   class and 1st→3rd in every sex. Women in 1st/2nd: 90%+. Men in 3rd: ~15%.
2. **Age by survival** (`04_age_by_survival.png`) — medians similar, but survivors show a cluster
   of young kids. Age alone is a weak signal.
3. **Fare vs age, by survival** (`05_fare_age_scatter.png`) — survivors cluster at higher fares
   across age groups. Fare mattered more than age.
4. **Pairplot** (`06_pairplot.png`) — `pclass`/`fare` visibly separate the two groups; `age`
   distributions mostly overlap.

### Task 6 — Standardization check

EDA-only, doesn't feed the model (that's fit on train split later).

| | age | fare |
|---|---|---|
| mean before | 29.32 | 32.10 |
| std before | 12.98 | 49.70 |
| mean after | 0.00 | 0.00 |
| std after | 1.00 | 1.00 |

Works as expected (`07_standardization_check.png`). `fare` keeps its skew shape — scaling doesn't
fix skew.

---

## Part B — Modeling

Reads raw `titanic.csv`, not the Part A cleaned version — Part A's median fill used the full
dataset, which would leak into the test set if reused for modeling. So this notebook does its own
imputation/encoding/scaling, fit on train only.

### Task 7 — Split

Features: `pclass, sex, age, sibsp, parch, fare, embarked`. Target: `survived`. Class balance
~62/38, moderate imbalance — stratified on `survived` so both splits keep that ratio (checked:
train 61.7/38.3, test 61.5/38.5). 80/20, `random_state=42`.

### Task 8 — Preprocessing

`age` → median impute. `embarked` → mode impute. `sex`/`embarked` → one-hot (`drop='first'`,
`handle_unknown='ignore'`). Numeric cols → `StandardScaler`. All inside a `ColumnTransformer` +
`Pipeline`, so `.fit()` only touches `X_train`, test only gets `.transform()`.

### Tasks 9–10 — Train & evaluate 3 classifiers

Logistic Regression, Decision Tree (`max_depth=4`), Random Forest (`n_estimators=200`) — same
split. Tree plot: `08_decision_tree.png`. Confusion matrices: `09_confusion_matrices.png`. ROC:
`10_roc_curves.png`.

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8435 |
| Decision Tree | 0.7765 | 0.8085 | 0.5507 | 0.6552 | 0.8144 |
| Random Forest | 0.8101 | 0.7778 | 0.7101 | 0.7424 | 0.8310 |

Random Forest wins F1/accuracy, Logistic Regression wins AUC. Decision tree trails on most
metrics — a single depth-4 tree can't capture interactions like sex×class.

### Task 11 — Imbalance handling

Baseline vs `class_weight='balanced'` vs SMOTE (train-fold only, via `imblearn` pipeline), on
Random Forest:

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| Baseline | 0.7778 | 0.7101 | 0.7424 |
| `class_weight='balanced'` | 0.7966 | 0.6812 | 0.7344 |
| SMOTE | 0.7286 | 0.7391 | 0.7338 |

None clearly beats baseline — they just trade precision for recall in opposite directions. Given
the imbalance isn't severe, pick based on whether false negatives or false positives cost more.

### Task 12 — Tuning

`GridSearchCV` on Random Forest (`n_estimators`, `max_depth`, `max_features`, `cv=5`).
`oob_score=True` set at construction (required for `oob_score_` to populate).

Best params: `max_depth=4, max_features='sqrt', n_estimators=200`. Best CV accuracy: 0.8301. OOB:
0.8287. Test accuracy: 0.7933. All close together — not badly overfit.

### Task 13 — Regression (predicting `fare`)

Linear regression, own 80/20 split. Residuals: `11_residual_plot.png`.

| MAE | RMSE | R² | Adj R² |
|---|---|---|---|
| 20.90 | 30.53 | 0.3975 | 0.3729 |

Residuals fan out wider as predicted fare increases — heteroscedasticity, not constant variance.
Matches `fare`'s right-skew from Task 3.

### Task 14 — Final comparison + recommendation

Classifier and regression metrics kept in separate tables (different scales).

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8435 |
| Decision Tree | 0.7765 | 0.8085 | 0.5507 | 0.6552 | 0.8144 |
| Random Forest | 0.8101 | 0.7778 | 0.7101 | 0.7424 | 0.8310 |

| Model | MAE | RMSE | R² | Adj R² |
|---|---|---|---|---|
| Linear Regression (fare) | 20.90 | 30.53 | 0.3975 | 0.3729 |

**Recommendation: tuned Random Forest.** Best F1/accuracy, close enough on AUC (0.8310 vs 0.8435)
that it's not a deciding factor. CV accuracy and OOB score both close to test accuracy — not
overfit. Logistic Regression is the fallback if interpretability matters — performance is nearly
identical.

### Task 15 — Saved pipeline

Whole pipeline (preprocessing + tuned Random Forest) saved as one object via `joblib.dump`, not
just the bare model. Reloaded with `joblib.load`, ran it on raw untouched rows — predictions match
the original. Works end-to-end on new raw data, no manual preprocessing needed.
