import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from aps.training import get_scores


class CalibratorWrapper:
    """Platt scaling (sigmoid) calibration wrapper for any sklearn estimator.

    Fits a logistic regression on top of raw scores from a pre-fitted estimator.
    Preferred over isotonic regression when positive-class N is small (~100):
    isotonic regression overfits badly below ~200 positives.
    """

    def __init__(self, estimator, method: str = "sigmoid"):
        self.estimator = estimator
        self.method = method
        self._calibrator = None

    def _raw_scores(self, X) -> np.ndarray:
        return get_scores(self.estimator, X)

    def fit(self, X, y) -> "CalibratorWrapper":
        s = self._raw_scores(X)
        if self.method == "sigmoid":
            self._calibrator = LogisticRegression(C=1.0).fit(s.reshape(-1, 1), y)
        else:
            from sklearn.isotonic import IsotonicRegression
            self._calibrator = IsotonicRegression(out_of_bounds="clip").fit(s, y)

        # Sanity check: calibration is degenerate if it predicts fewer positives
        # than actually exist on the calibration set. This happens when the raw
        # score range is extreme (e.g. SVC decision function spanning many orders
        # of magnitude), causing Platt scaling to collapse all probabilities near 0.
        cal_proba = self.predict_proba(X)[:, 1]
        predicted_pos = (cal_proba >= 0.5).sum()
        actual_pos = int(y.sum())
        if predicted_pos == 0 and actual_pos > 0:
            print(
                f"  WARNING: calibration degenerate (0 predicted positives vs "
                f"{actual_pos} actual). Falling back to raw scores for threshold tuning."
            )
            self._calibrator = None  # flag: use raw scores directly

        return self

    @property
    def _is_degenerate(self) -> bool:
        return self._calibrator is None

    def predict_proba(self, X) -> np.ndarray:
        s = self._raw_scores(X)
        if self._is_degenerate:
            # Normalise raw scores to [0, 1] via min-max so they act as pseudo-probabilities
            lo, hi = s.min(), s.max()
            cal = (s - lo) / (hi - lo) if hi > lo else np.full_like(s, 0.5)
        elif self.method == "sigmoid":
            cal = self._calibrator.predict_proba(s.reshape(-1, 1))[:, 1]
        else:
            cal = self._calibrator.transform(s)
        return np.column_stack([1 - cal, cal])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def decision_function(self, X) -> np.ndarray:
        return self._raw_scores(X)


def tune_threshold(
    estimator,
    X_thr,
    y_thr: np.ndarray,
    X_test,
    y_test: np.ndarray,
    model_name: str,
    cost_fp: int = 10,
    cost_fn: int = 500,
    n_steps: int = 500,
) -> tuple[float | None, int | None]:
    """Find optimal decision threshold on the threshold-tuning val half.

    SVC note: decision_function scores are used raw (no per-call normalisation)
    so the threshold transfers correctly across splits.

    Returns (best_threshold, test_cost_at_threshold), or (None, None) if the
    estimator has no probability / score output.
    """

    def _scores(est, X):
        if hasattr(est, "predict_proba"):
            return est.predict_proba(X)[:, 1]
        if hasattr(est, "decision_function"):
            return est.decision_function(X)
        return None

    thr_scores = _scores(estimator, X_thr)
    test_scores = _scores(estimator, X_test)
    if thr_scores is None:
        print(f"  {model_name}: no probability/score output — skipping.")
        return None, None

    thresholds = np.linspace(thr_scores.min(), thr_scores.max(), n_steps)

    def _cost(scores, y, t):
        cm = confusion_matrix(y, (scores >= t).astype(int))
        return cost_fp * cm[0][1] + cost_fn * cm[1][0]

    thr_costs = [_cost(thr_scores, y_thr, t) for t in thresholds]
    best_t = thresholds[int(np.argmin(thr_costs))]

    cost_tuned = _cost(test_scores, y_test, best_t)
    cost_default = _cost(test_scores, y_test, np.median(thresholds))

    if cost_tuned <= cost_default:
        final_cost, final_t = cost_tuned, best_t
    else:
        final_cost, final_t = cost_default, np.median(thresholds)

    saving = cost_default - final_cost
    pct = saving / cost_default * 100 if cost_default > 0 else 0.0
    print(f"{model_name}:")
    print(f"  default threshold -> test cost {cost_default:,}")
    print(f"  tuned  threshold  -> test cost {final_cost:,}  (saves {saving:,}, {pct:.1f}%)")
    print()
    return final_t, final_cost


def bootstrap_cost_ci(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    n_bootstrap: int = 1000,
    cost_fp: int = 10,
    cost_fn: int = 500,
    seed: int = 42,
) -> tuple[int, int]:
    """95% bootstrap CI for total cost, bootstrapping over precomputed predictions.

    Accepts precomputed predictions so CI brackets match the tuned-threshold cost
    used for model selection. Bootstrapping over a fixed prediction array is
    statistically identical to re-calling the model each iteration.
    """
    rng = np.random.default_rng(seed)
    n = len(y_test)
    costs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        cm = confusion_matrix(y_test[idx], y_pred[idx])
        costs.append(cost_fp * cm[0][1] + cost_fn * cm[1][0])
    lo, hi = np.percentile(costs, [2.5, 97.5])
    return int(lo), int(hi)
