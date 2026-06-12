import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils import resample


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop columns whose missing rate in training is >= threshold.

    Fit on the full training set before the 80/20 split so EDA and the
    feature pipeline see the same column schema.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None) -> "ColumnDropper":
        missing_pct = X.isna().mean()
        dropped = missing_pct[missing_pct >= self.threshold].index.tolist()
        self.cols_to_drop_: list[str] = dropped
        self.cols_to_keep_: list[str] = [c for c in X.columns if c not in dropped]
        print(
            f"ColumnDropper: dropping {len(dropped)} columns "
            f"(>={self.threshold * 100:.0f}% missing): {dropped}"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.cols_to_keep_].copy()


class MedianImputerWithIndicators(BaseEstimator, TransformerMixin):
    """Median imputation with binary missing-value indicator columns.

    Adds a `<col>_missing` indicator (1 = was missing) for every feature
    whose missing rate in training reaches `missing_threshold`, then fills
    all remaining NaNs with training medians.

    Fit on the 80% training split only — medians and indicator column list
    must never be computed from validation or test data.
    """

    def __init__(self, missing_threshold: float = 0.2):
        self.missing_threshold = missing_threshold

    def fit(self, X: pd.DataFrame, y=None) -> "MedianImputerWithIndicators":
        self.medians_: pd.Series = X.median(skipna=True)
        missing_rates = X.isna().mean()
        self.indicator_cols_: list[str] = (
            missing_rates[missing_rates >= self.missing_threshold].index.tolist()
        )
        print(
            f"MedianImputerWithIndicators: added {len(self.indicator_cols_)} "
            f"missing-indicator columns (>={self.missing_threshold * 100:.0f}% missing in training)."
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        indicators = pd.DataFrame(
            {
                f"{c}_missing": X[c].isna().astype(int)
                for c in self.indicator_cols_
                if c in X.columns
            },
            index=X.index,
        )
        out = X.fillna(self.medians_)
        return pd.concat([out, indicators], axis=1)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        base = list(input_features) if input_features is not None else []
        return np.array(base + [f"{c}_missing" for c in self.indicator_cols_])


class WinsorizerTransformer(BaseEstimator, TransformerMixin):
    """IQR Winsorisation fitted on training data only.

    Uses `iqr_scale` × IQR fences (default 3.0, the standard Winsorisation
    threshold). Skips zero-IQR columns (zero-inflated sparse sensors where
    Q1 == Q3 — clipping would collapse every value to a constant) and
    `_missing` indicator columns (binary, clipping is undefined for them).

    Fit on the 80% training split; thresholds transfer unchanged to val/test.
    """

    def __init__(self, iqr_scale: float = 3.0):
        self.iqr_scale = iqr_scale

    def fit(self, X: pd.DataFrame, y=None) -> "WinsorizerTransformer":
        numeric_cols = [c for c in X.columns if not c.endswith("_missing")]
        Q1 = X[numeric_cols].quantile(0.25)
        Q3 = X[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1

        self.zero_iqr_cols_: list[str] = IQR[IQR == 0].index.tolist()
        cols_to_clip = [c for c in numeric_cols if c not in self.zero_iqr_cols_]

        self.lower_: pd.Series = (Q1 - self.iqr_scale * IQR)[cols_to_clip]
        self.upper_: pd.Series = (Q3 + self.iqr_scale * IQR)[cols_to_clip]
        self.cols_to_clip_: list[str] = cols_to_clip

        print(
            f"WinsorizerTransformer ({self.iqr_scale}×IQR): skipping "
            f"{len(self.zero_iqr_cols_)} zero-IQR columns, "
            f"clipping {len(cols_to_clip)} columns."
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.cols_to_clip_:
            if col in out.columns:
                out[col] = out[col].clip(lower=self.lower_[col], upper=self.upper_[col])
        return out

    def clipping_report(self, X: pd.DataFrame) -> pd.Series:
        """Fraction of values clipped per feature (for validation / EDA)."""
        clipped = pd.DataFrame(index=X.index)
        for col in self.cols_to_clip_:
            if col in X.columns:
                lo = X[col] <= self.lower_[col]
                hi = X[col] >= self.upper_[col]
                clipped[col] = (lo | hi).astype(int)
        return clipped.mean().sort_values(ascending=False)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(list(input_features) if input_features is not None else [])


def build_feature_pipeline(
    missing_threshold: float = 0.2,
    iqr_scale: float = 3.0,
) -> Pipeline:
    """Return the fitted-on-training-only feature pipeline.

    Chain: MedianImputerWithIndicators → WinsorizerTransformer.
    ColumnDropper is intentionally excluded — it is fit on the full training
    set before the 80/20 split so EDA can run on a consistent feature schema.
    StandardScaler is excluded — it is added per-model in the training pipeline
    for scale-sensitive models (LR, SVM, MLP) only.
    """
    return Pipeline(
        [
            ("imputer", MedianImputerWithIndicators(missing_threshold=missing_threshold)),
            ("winsoriser", WinsorizerTransformer(iqr_scale=iqr_scale)),
        ]
    )


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select a fixed list of columns — used to reduce features at inference time."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None) -> "ColumnSelector":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns]


def build_inference_pipeline(
    col_dropper: "ColumnDropper",
    feature_pipeline: Pipeline,
    model,
    feature_cols: list[str] | None = None,
) -> Pipeline:
    """Assemble a single end-to-end inference pipeline.

    Chains: ColumnDropper → MedianImputerWithIndicators → WinsorizerTransformer
            → [ColumnSelector] → Model.

    All steps are pre-fitted — call .predict() or .predict_proba() directly.
    Input must be a raw feature DataFrame (no label column) with NaN for
    missing values, matching the original CSV schema.

    Parameters
    ----------
    col_dropper      : fitted ColumnDropper (fit on full training features)
    feature_pipeline : fitted Pipeline with 'imputer' and 'winsoriser' steps
    model            : fitted sklearn estimator (may itself be a Pipeline for
                       scale-sensitive models)
    feature_cols     : if provided, add a ColumnSelector step to reduce to this
                       feature subset (used for the LGBM reduced model)
    """
    steps: list = [
        ("col_dropper", col_dropper),
        ("imputer",     feature_pipeline.named_steps["imputer"]),
        ("winsoriser",  feature_pipeline.named_steps["winsoriser"]),
    ]
    if feature_cols is not None:
        steps.append(("feature_selector", ColumnSelector(feature_cols)))
    steps.append(("model", model))
    return Pipeline(steps)


def downsample_majority_class(
    df: pd.DataFrame,
    target_column: str = "class",
    random_state: int = 1,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Undersample the majority class to match the minority class size."""
    if target_column not in df.columns:
        raise ValueError(f"Column {target_column!r} not in DataFrame")
    majority = df[df[target_column] == "neg"]
    minority = df[df[target_column] == "pos"]
    majority_down = resample(
        majority, replace=False, n_samples=len(minority), random_state=random_state
    )
    balanced = pd.concat([majority_down, minority]).sample(
        frac=1, random_state=random_state
    )
    labels = (balanced[target_column] == "pos").astype(int).to_numpy()
    return balanced, labels
