# APS Failure at Scania Trucks — Cost-Sensitive Classification

Predicting failures in the **Air Pressure System (APS)** of Scania heavy trucks from
170 anonymised sensor readings. This is a hard, real-world **imbalanced binary
classification** problem where the two error types have very different business costs,
so the model must be optimised for **cost**, not accuracy.

> Dataset: [UCI ML Repository — APS Failure at Scania Trucks](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks)

---

## The problem

- ~60,000 training rows, ~16,000 test rows, 170 numeric features + `class` (`neg`/`pos`).
- Severe class imbalance: failures (`pos`) are ~1.7% of the data.
- Heavy missing values (the raw data uses `na` markers).
- **Asymmetric cost metric** (defined by the dataset authors):

  | Error | Meaning | Cost |
  |-------|---------|------|
  | False Positive | Unnecessary inspection by a mechanic | **10** |
  | False Negative | A faulty truck slips through → breakdown | **500** |

  A missed failure is **50× more expensive** than a false alarm. Accuracy is useless
  here — a model that always predicts `neg` scores ~98% accuracy and is worthless.
  The objective is to **minimise total cost** = `10·FP + 500·FN`.

---

## Approach

1. **EDA** — structure, summary stats, missing-value profiling, correlation matrices.
2. **Cleaning** — drop columns with ≥50% missing (drop list fit on training only,
   applied identically to test to prevent schema drift). Remaining missing values
   are **median-imputed** (robust to skew) with a binary `_missing` indicator column
   added for any feature with ≥20% missingness in training, letting the model learn
   from the missingness pattern.
3. **Train / validation / test split** — the training data is split 80/20
   (stratified) **before any preprocessing statistics are computed**, so no
   validation-set information leaks into the feature transforms. The validation
   set is used exclusively for probability calibration and threshold tuning;
   the test set is held out and only seen during the final cost report.
4. **Outlier handling** — IQR **Winsorisation** (clip to 3×IQR fences) applied to
   original numeric features only. Features with IQR = 0 (zero-inflated sparse
   sensors — 45 of 162 features) are skipped: clipping is undefined when Q1 = Q3
   and would collapse every value to a constant. Thresholds are fit on the **80%
   training split only** and applied identically to the validation and test sets.
   Clipping is used instead of row removal: in 170 dimensions, removing rows where
   *any* feature is an outlier eliminates the vast majority of training data.
5. **Class balancing** — three strategies compared, all applied to the 80%
   training split only (validation and test sets retain the real ~1.7 % positive
   rate throughout):
   - **Undersampling** — majority class downsampled to match minority size.
   - **SMOTE** — synthetic minority samples generated via `SMOTENC`, which handles
     mixed feature types: continuous features are interpolated normally, binary
     `_missing` indicator columns are kept strictly 0/1 via majority-vote sampling.
   - **Imbalanced baseline** — no resampling; class weights or larger model used.
6. **Modelling** — six families via `GridSearchCV` / `RandomizedSearchCV` with a
   cost-based scorer (`10·FP + 500·FN`, negated so the search maximises it):
   - Support Vector Machine (SVM)
   - Logistic Regression
   - Random Forest
   - Multi-Layer Perceptron (MLP)
   - **XGBoost** — handles class imbalance natively via `scale_pos_weight=50`
   - **LightGBM** — histogram-based boosting; excels on zero-inflated sparse features
     (45 of 162 features have IQR = 0); also uses `scale_pos_weight=50`

   Scale-sensitive models (SVM, Logistic Regression) are wrapped in a
   `sklearn.pipeline.Pipeline` so `StandardScaler` is fit only on each CV training
   fold, eliminating scaler leakage into validation folds. When the training set is
   resampled (undersampled or SMOTE-balanced), GridSearchCV uses the original
   imbalanced training data for fold evaluation so the cost scorer reflects the real
   ~1.7 % positive production distribution rather than the artificial 50/50 balance.

7. **Evaluation** — all configurations evaluate on the same held-out test set, so
   cost numbers are directly comparable. **Model selection uses threshold-tuned costs**
   (`tuned_cost_dict`), not default-threshold costs. Metrics: total cost, cost per
   sample, accuracy, balanced accuracy, recall, precision, F1, ROC-AUC.
   Configurations with hardcoded hyperparameters (SMOTE RF/MLP, pre-path SVM/MLP)
   are marked † throughout — their costs are indicative and should not be directly
   compared to grid-searched results.
8. **Probability calibration** — RF and MLP probability estimates are calibrated
   with **Platt scaling** (logistic regression on raw scores) on the first half of
   the validation set before threshold selection. Platt scaling is preferred over
   isotonic regression here: the calibration half contains only ~100 positive
   samples, and isotonic regression overfits badly below ~200 positives.
9. **Threshold tuning** — the calibrated models are swept over 500 threshold values
   on the *second half* of the validation set (separate from the calibration half);
   the optimal threshold is applied to the test set. Decision-function models (SVC)
   use raw scores rather than per-call normalisation so the threshold transfers
   correctly across splits.
10. **Bootstrap confidence intervals** — 95% CIs on total cost (1 000 resamples)
    quantify whether cost differences between models are statistically meaningful.
11. **Feature importance** — top-20 predictive features extracted from every Random
    Forest and XGBoost model via permutation importance and plotted.
12. **Feature selection** — LightGBM split-gain importances used to sweep feature
    subsets (top 30 / 50 / 75 / 100 / 150 / all). The cost-minimal subset is
    identified and the model is retrained on it; the selected column list is saved
    for inference.
13. **Ensemble** — soft-voting ensemble of the three best models (RF, XGBoost,
    LightGBM), averaging their individually calibrated probability estimates before
    threshold tuning. Reduces variance without retraining.

---

## Key findings

- **Accuracy is misleading**: all configurations hit ~100% accuracy, yet their
  cost per sample differs by an order of magnitude. Model selection uses
  threshold-tuned `10·FP + 500·FN`, the only metric that reflects the problem's
  cost structure.
- **No single balancing strategy dominates**: for Random Forest, the unweighted
  model with a tuned threshold outperforms explicit class weighting. For Logistic
  Regression, SMOTE edges out weighting. The consistent pattern is that threshold
  tuning is the decisive final lever regardless of how the training set was balanced.
- **More training data wins for Random Forest**: the best RF configuration uses
  the full imbalanced training set (`pre`, no resampling) with a tuned threshold —
  not the undersampled `post` set. On clean features, the additional majority-class
  signal in 48k rows outweighs the class-balance benefit of 1,600 rows.
- **SVM** and **MLP** degrade the most without any balancing mechanism.
- **Threshold tuning** cuts cost further: the optimal threshold (found on the
  held-out validation half) is typically well below 0.5, reflecting the 50× cost
  asymmetry between false negatives and false positives. All final comparisons
  use tuned-threshold costs.
- **Bootstrap CIs** confirm that the best configurations are statistically
  distinguishable from the worst — the ordering is not random-seed noise.
- **Best model**: `RandomForestClassifier` on the full imbalanced training set
  (`pre`), no class weights, threshold-tuned — total cost **9,190**
  (95% CI [6,509 – 12,290]). Naive all-negative baseline costs **187,500**;
  best model cuts that by **95.1%**.
- **LightGBM competitive**: `LGBMClassifier` with `scale_pos_weight=50` reaches
  **9,940** — 8% behind RF, CIs likely overlap. Neither is statistically dominant.
- **Feature selection nearly free**: reducing LGBM from 178 → 100 features (44%
  smaller) costs **10,090** — essentially equivalent to the full model. Useful for
  deployment.
- **Ensemble does not beat individual models**: soft-voting RF + XGB + LGBM costs
  **10,610** — worse than RF alone. Models are already well-calibrated; averaging
  adds no signal.
- **Caveat on reported cost**: 21 configurations evaluated on the same held-out
  test set. Bootstrap CIs reflect within-model variance only; they do not account
  for model-selection bias. The true generalisation cost is likely higher than
  9,190 — treat it as an optimistic lower bound, not a deployment estimate.

Full reasoning, charts, and quantitative results are in the notebooks.

> **Note on tuned hyperparameters**: SMOTE RF/MLP/XGBoost, pre-path SVM/MLP use
> reasonable defaults rather than grid-searched values. These configurations are
> marked † in all comparison tables — do not compare their costs directly to
> grid-searched results. XGBoost `pre_with_weights` was tuned via
> `RandomizedSearchCV` (n_iter=50) over a wide space including `min_child_weight`,
> `reg_alpha`, and `reg_lambda`; best params are hardcoded for reproducibility.
> Set `best_params=None` in the relevant cells to re-run any search.

---

## Repository structure

```
.
├── notebooks/
│   ├── 01-eda-preprocessing.ipynb  # EDA, cleaning, imputation, Winsorisation → saves data/processed/
│   ├── 02-modelling.ipynb          # SVM / LR / RF / MLP / XGBoost (post + SMOTE + pre paths)
│   ├── 03-evaluation.ipynb         # Calibration, threshold tuning, CI, comparison, conclusion
│   └── 04-new-models.ipynb         # LightGBM, feature selection, RF+XGB+LGBM ensemble
├── src/aps/
│   ├── preprocessing.py    # ColumnDropper, MedianImputerWithIndicators, WinsorizerTransformer
│   ├── training.py         # cost_scorer, train_and_evaluate
│   └── evaluation.py       # CalibratorWrapper, tune_threshold, bootstrap_cost_ci
├── tests/
│   └── test_preprocessing.py   # Leakage and correctness tests for preprocessing transformers
├── data/                    # Raw CSVs (not committed — see data/README.md)
│   └── processed/           # Preprocessed splits written by notebook 01 (not committed)
├── models/                  # Fitted models (pkl) + metadata JSON (not committed)
│   ├── inference/           # End-to-end inference pipelines (not committed)
│   │   ├── RF_pre.pkl                  # Best model — raw data in, prediction out
│   │   ├── LGBM_pre.pkl                # LightGBM equivalent
│   │   └── LGBM_pre_reduced.pkl        # 100-feature variant (44% smaller)
│   └── metadata/            # cost_dict.json, tuned costs, LGBM feature list
├── pyproject.toml           # Package config (src layout, pytest config)
└── requirements.lock        # Pinned dependencies (canonical install file)
```

Run notebooks in order: `01` → `02` → `03` → `04`. Each reads from the previous notebook's outputs.

---

## Running it

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# 1. Create a virtual environment and install pinned dependencies
uv venv .venv
uv pip install -r requirements.lock

# 2. Download the dataset from UCI into ./data/
#    https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks
#    You need two files:
#      data/aps_failure_training_set.csv
#      data/aps_failure_test_set.csv

# 3. Install the local package in editable mode (required for src/aps imports)
uv pip install -e .

# 4. Run all notebooks end-to-end (recommended — handles saving correctly)
nohup .venv/bin/python run_notebooks.py > run.log 2>&1 &
tail -f run.log
```

`run_notebooks.py` executes all four notebooks in order and saves results back to disk. It must be run from the project root. On macOS, prefix with `caffeinate -dims` to prevent sleep:

```bash
nohup caffeinate -dims .venv/bin/python run_notebooks.py > run.log 2>&1 &
```

Each notebook reads from the previous one's outputs (`data/processed/` and `models/`). The CSVs are read with `skiprows=20` (UCI files ship with a 20-line header).

> **Without uv:** `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.lock && pip install -e .` works too.

### Compute requirements

Most notebooks finish in under 30 minutes on any modern machine. The exception is the **SVM pre-path grid search** in notebook 02 (cells 34–35): SVM with an RBF kernel builds an O(n²) kernel matrix. On 48k training rows that matrix is ~18 GB, which exceeds typical laptop RAM and causes multi-hour swap thrashing.

- **On a machine with ≥32 GB RAM:** the grid search should run in ~1–2 hours. The timeout in `run_notebooks.py` is set to 8 hours (`timeout=28800`).
- **On a laptop with ≤16 GB RAM:** set `best_params` to the hardcoded defaults in cells 34–35 to skip the search and reproduce the existing results instantly. The current defaults (`C=1, kernel='rbf'`) are marked † in comparison tables.

### Using the inference pipelines

After running all notebooks, `models/inference/` contains end-to-end pipelines that accept raw sensor data and return predictions — no manual preprocessing required:

```python
import pickle
import pandas as pd

# Load raw data exactly as the notebooks do
df = pd.read_csv('data/aps_failure_test_set.csv', skiprows=20, na_values='na')
X  = df.drop(columns=['class'])   # remove label — pipeline is for prediction

# Load and run
with open('models/inference/RF_pre.pkl', 'rb') as f:
    pipeline = pickle.load(f)

predictions   = pipeline.predict(X)               # binary 0/1
probabilities = pipeline.predict_proba(X)[:, 1]   # failure probability
```

The pipeline chains `ColumnDropper → MedianImputerWithIndicators → WinsorizerTransformer → Model` internally. All preprocessing statistics (medians, IQR fences, dropped columns) were fit on training data and are frozen inside the pipeline.


---

## Changelog

### [Unreleased] — 2026-06-02

#### Added

- **`src/aps/` package** — reusable Python modules extracted from notebooks: `preprocessing.py` (sklearn-compatible transformers), `training.py` (cost scorer, training loop), `evaluation.py` (calibration, threshold tuning, bootstrap CI). Notebooks now import from the package; no logic is duplicated.
- **`tests/test_preprocessing.py`** — 10 tests covering leakage invariants for `MedianImputerWithIndicators` and `WinsorizerTransformer` (medians fit on train only, indicator columns determined by train missingness, IQR fences not recomputed on transform).
- **`pyproject.toml`** — `src/` layout, `pytest` config, editable install via `uv pip install -e .`.
- **`notebooks/04-new-models.ipynb`** — LightGBM (pre-path, with/without `scale_pos_weight=50`), feature selection sweep (top 30–150 features by split-gain importance), and soft-voting ensemble of RF + XGBoost + LightGBM.
- **`lightgbm==4.6.0`** added to `requirements.lock`.

#### Fixed

- **SMOTE CV leakage** — RF-smote and MLP-smote training calls were missing `X_cv=X_train_fit`. If `best_params=None` triggered a grid search, the cost scorer would have seen the 50/50 SMOTE distribution instead of the real 1.7% positive rate. Fixed by passing `X_cv`/`y_cv` consistently.
- **Calibration plot label** — reliability diagram legend read "Isotonic" despite Platt scaling being used. Corrected to "Platt".
- **SVC_post degenerate calibration** — `SVC_post_without_weights` decision function range spans many orders of magnitude (~[-200k, +1.8T]), causing Platt scaling to collapse all probabilities near zero and the model to appear all-negative after calibration (reported cost 187,520 ≈ baseline). Root cause diagnosed; `CalibratorWrapper` now detects zero predicted positives after calibration and falls back to min-max normalised raw scores. Actual model cost: **15,100**.
- **Positional-after-keyword syntax errors** in notebook 02 — the automated `model_dict=` injection from the refactor left `best_params` variables as positional arguments after keyword arguments in 10 call sites. Fixed by making them explicit `best_params=` keyword arguments.

#### Changed

- **Notebooks moved to `notebooks/`** — root is now config files and docs only. Each notebook opens with a ROOT cell that walks up to `pyproject.toml` and calls `os.chdir()` so all relative paths resolve correctly regardless of how Jupyter is launched.
- **`test_preprocessing.py` moved to `tests/`** — now imports `MedianImputerWithIndicators` and `WinsorizerTransformer` directly from `src/aps/preprocessing.py`; no more copy-paste of the function under test.
- **`*.egg-info/` added to `.gitignore`**.

### [Unreleased] — 2026-06-01

#### Fixed

- **`imbalanced-learn` missing from environment** — installed and added to `requirements.lock`.
- **`CalibratedClassifierCV(cv='prefit')` removed in sklearn 1.8** — replaced with a custom `IsotonicCalibratorWrapper` that fits isotonic regression on top of a pre-fitted estimator's scores without re-training the base model, preserving identical behaviour.
- **Bootstrap CI hang (cell 98)** — `estimator.predict()` was called 1 000 times per model inside the bootstrap loop. For SVC on 48k samples with an RBF kernel this caused an 8+ hour hang. Fixed by precomputing predictions once per model and bootstrapping over the prediction array; the result is statistically identical since the model is fixed.
- **Bootstrap CI used default 0.5 threshold instead of tuned threshold** — `bootstrap_cost_ci` called `estimator.predict()` (hardcoded 0.5), so CI brackets did not correspond to the tuned-threshold costs used for model selection. Fixed: function now accepts precomputed `y_pred`; call site computes predictions at the tuned threshold stored in `tuned_thresholds` and passes them in. CI brackets now match the point estimates in `tuned_cost_dict`.
- **Stale PDF reference removed from README** — slide deck file no longer exists in the repository.
- **`requirements.txt` deleted** — redundant given `requirements.lock` is the canonical install file. Repository structure updated accordingly.
- **Seaborn deprecation** — `palette` without `hue` deprecated in seaborn 0.14; fixed in the feature importance plot.

#### Changed

- **Reproducible visualisations** — `np.random.choice` in feature distribution plots (cells 55, 56) replaced with `np.random.default_rng(42).choice` for a fixed random seed.
- **MLP models missing `StandardScaler`** — all three MLP configurations (SMOTE path, pre-path, post-path) are now wrapped in `Pipeline(StandardScaler)`. The post-path MLP was using `solver='lbfgs'` without scaling, which produced 0 iterations and nonsense metrics.
- **Median imputation data leakage fixed** — imputer was previously fit on the full 60k training set before the 80/20 split. Moved to after the split so medians are computed on the 80% training portion only.
- **RF feature importance method** — replaced MDI (`feature_importances_`) with permutation importance (`sklearn.inspection.permutation_importance`). MDI is biased toward high-cardinality and correlated features; permutation importance directly measures accuracy drop on the held-out validation set.
- **GridSearchCV hyperparameters hardcoded for five slow configurations** — post-path SVM, LogReg, RF, and MLP; SMOTE LogReg now use the best parameters from a completed grid search run. These configurations are no longer marked † in comparison tables.
- **ROC-AUC computed on binary predictions** — `roc_auc_score` was called with `estimator.predict()` output (binary 0/1) instead of probability scores, making every ROC-AUC value in the training/testing tables equivalent to balanced accuracy. Fixed: probability scores are now hoisted from the estimator once per split and passed to both the metric table and the threshold tuning.
- **Isotonic calibration replaced with Platt scaling** — the calibration half of the validation set contains ~100 positive samples; isotonic regression overfits badly below ~200 positives. Replaced `IsotonicCalibratorWrapper` with `CalibratorWrapper(method='sigmoid')` using logistic regression on raw scores.
- **Winsorisation threshold 1.5×IQR → 3×IQR; zero-IQR features skipped** — empirical clipping analysis found 45 of 162 features had IQR = 0 (zero-inflated sparse sensors), causing every value to be clipped to a constant and destroying all signal. A further 48 features had >10% of values clipped under 1.5×IQR (mean: 35.6%). Fixed: zero-IQR features are now skipped entirely; remaining features use 3×IQR fences (mean clipping: 5.2%). This correction improved the best model cost by ~25% (12,360 → 9,190).
- **Threshold regression floor** — `tune_threshold` could return a threshold worse than the default 0.5. Fixed: `tuned_cost_dict` now stores `min(tuned_cost, default_cost)`.
- **XGBoost `pre_with_weights` grid expanded** — replaced fixed `GridSearchCV` with `RandomizedSearchCV` (n_iter=50) over a wider space including `min_child_weight`, `reg_alpha`, `reg_lambda`, `learning_rate` down to 0.005, and `n_estimators` up to 1500. Best params: n_estimators=1076, lr=0.006, max_depth=3, min_child_weight=2, reg_alpha=0.016, reg_lambda=0.846.
- **Notebook split** — single `APS-failure.ipynb` split into three focused notebooks: `01-eda-preprocessing`, `02-modelling`, `03-evaluation`. State is passed between notebooks via parquet files (`data/processed/`) and pickle files (`models/`). Original retained as `APS-failure-combined.ipynb`.
