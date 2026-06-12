import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def cost_scorer(y_true: np.ndarray, y_pred: np.ndarray,
                cost_fp: int = 10, cost_fn: int = 500) -> float:
    """Negated asymmetric cost for use as a GridSearchCV scorer.

    Returns -(10·FP + 500·FN) so GridSearchCV's maximisation equals
    cost minimisation. Module-level so it is picklable for n_jobs=-1.
    """
    cm = confusion_matrix(y_true, y_pred)
    return -(cost_fp * cm[0][1] + cost_fn * cm[1][0])


def get_scores(estimator, X: pd.DataFrame) -> np.ndarray | None:
    """Extract probability scores or raw decision-function values."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return None


def train_and_evaluate(
    model_class,
    param_grid: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cost_dict: dict,
    name_suffix: str,
    model_dict: dict | None = None,
    best_params: dict | None = None,
    cost_fp: int = 10,
    cost_fn: int = 500,
    scale: bool = False,
    X_cv: pd.DataFrame | None = None,
    y_cv: np.ndarray | None = None,
    search: str = "grid",
    n_iter: int = 50,
    gs_n_jobs: int = -1,
) -> dict:
    """Train a model, optionally searching hyperparameters, and record test cost.

    Parameters
    ----------
    model_class : sklearn estimator class
    param_grid  : search space; ignored when best_params is provided
    X_train     : training features (possibly resampled)
    X_test      : held-out test features
    y_train     : training labels
    y_test      : held-out test labels
    cost_dict   : dict to record {model_name: default-threshold cost}
    name_suffix : appended to model class name to form the model key
                  e.g. 'post_without_weights' → 'RandomForestClassifier_post_without_weights'
    best_params : if provided, skip search and use these directly
    cost_fp     : cost of a false positive
    cost_fn     : cost of a false negative
    scale       : wrap model in StandardScaler Pipeline (for LR, SVM, MLP)
    X_cv/y_cv   : separate CV data for the cost scorer — pass the original
                  imbalanced training set when X_train is resampled so the
                  scorer sees the real ~1.7% positive rate
    search      : 'grid' (GridSearchCV) or 'random' (RandomizedSearchCV)
    n_iter      : iterations for RandomizedSearchCV
    gs_n_jobs   : n_jobs for the search

    Returns
    -------
    best_params dict (useful when search was run)
    """
    model_name = f"{model_class.__name__}_{name_suffix}"
    print(f"\n>>> {model_name}  train={X_train.shape}  test={X_test.shape}")

    X_for_cv = X_cv if X_cv is not None else X_train
    y_for_cv = y_cv if y_cv is not None else y_train
    scorer = make_scorer(cost_scorer)

    if best_params is None:
        if scale:
            pipeline_grid = {f"model__{k}": v for k, v in param_grid.items()}
            base = Pipeline([("scaler", StandardScaler()), ("model", model_class())])
            if search == "random":
                gs = RandomizedSearchCV(base, pipeline_grid, n_iter=n_iter, cv=5,
                                        n_jobs=gs_n_jobs, verbose=1, scoring=scorer,
                                        random_state=42)
            else:
                gs = GridSearchCV(base, pipeline_grid, cv=5, n_jobs=gs_n_jobs,
                                  verbose=1, scoring=scorer)
            gs.fit(X_for_cv, y_for_cv)
            best_params = {
                k.replace("model__", ""): v for k, v in gs.best_params_.items()
            }
        else:
            if search == "random":
                gs = RandomizedSearchCV(model_class(), param_grid, n_iter=n_iter, cv=5,
                                        n_jobs=gs_n_jobs, verbose=1, scoring=scorer,
                                        random_state=42)
            else:
                gs = GridSearchCV(model_class(), param_grid, cv=5, n_jobs=gs_n_jobs,
                                  verbose=1, scoring=scorer)
            gs.fit(X_for_cv, y_for_cv)
            best_params = gs.best_params_
        print(f"Best parameters: {best_params}")

    if scale:
        estimator = Pipeline(
            [("scaler", StandardScaler()), ("model", model_class(**best_params))]
        )
    else:
        estimator = model_class(**best_params)

    estimator.fit(X_train, y_train)

    if model_dict is not None:
        model_dict[model_name] = estimator

    y_pred_train = estimator.predict(X_train)
    y_pred_test = estimator.predict(X_test)

    def _metrics(y_true, y_pred, y_scores=None):
        auc = roc_auc_score(y_true, y_scores) if y_scores is not None else float("nan")
        return (
            accuracy_score(y_true, y_pred),
            balanced_accuracy_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            precision_score(y_true, y_pred),
            f1_score(y_true, y_pred),
            auc,
        )

    tr = _metrics(y_train, y_pred_train, get_scores(estimator, X_train))
    te = _metrics(y_test, y_pred_test, get_scores(estimator, X_test))

    print()
    print("           ACC    BA     RECALL   PRECISION  F1       ROC-AUC")
    print(f"training  {tr[0]:.3f}  {tr[1]:.3f}  {tr[2]:.3f}    {tr[3]:.3f}     {tr[4]:.3f}  {tr[5]:.3f}")
    print(f"testing   {te[0]:.3f}  {te[1]:.3f}  {te[2]:.3f}    {te[3]:.3f}     {te[4]:.3f}  {te[5]:.3f}")
    print()

    cm = confusion_matrix(y_test, y_pred_test)
    fp, fn = cm[0][1], cm[1][0]
    cost_dict[model_name] = cost_fp * fp + cost_fn * fn
    print(f"Default-threshold test cost: {cost_dict[model_name]:,}")

    return best_params
