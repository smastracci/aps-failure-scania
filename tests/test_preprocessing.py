"""
Tests for MedianImputerWithIndicators and WinsorizerTransformer.

Core invariant tested throughout: statistics (medians, indicator column list,
IQR fences) must be fit on training data only and applied unchanged to
validation and test splits — no leakage in either direction.
"""
import numpy as np
import pandas as pd
import pytest

from aps.preprocessing import ColumnDropper, MedianImputerWithIndicators, WinsorizerTransformer


def _make_df(feature_values: dict, n_rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame(feature_values)


# ── MedianImputerWithIndicators ───────────────────────────────────────────────

class TestMediansFitOnTrainOnly:
    """Medians must come from training data, never from val/test."""

    def test_val_imputed_with_training_median(self):
        train = _make_df({"a": list(range(1, 11)) * 2})       # median 5.5
        val   = _make_df({"a": [np.nan] * 20})                 # val median would be NaN

        imp = MedianImputerWithIndicators().fit(train)
        val_out = imp.transform(val)

        assert imp.medians_["a"] == pytest.approx(5.5)
        assert (val_out["a"] == 5.5).all(), \
            "Val must be imputed with training median 5.5, not val median"

    def test_medians_not_recomputed_on_transform(self):
        train = _make_df({"b": [100.0] * 20})   # median 100
        val   = _make_df({"b": [1.0] * 20})      # median 1

        imp = MedianImputerWithIndicators().fit(train)
        imp.transform(val)  # must NOT change medians_

        assert imp.medians_["b"] == 100.0, \
            "Medians must not be recomputed when transform is called"


class TestIndicatorColumnsTransfer:
    """Indicator columns are determined by training missingness, not val/test."""

    def test_high_train_missingness_creates_indicator_in_val(self):
        train_vals = [np.nan] * 10 + [1.0] * 10   # 50% missing → indicator
        val_vals   = [np.nan] * 1  + [1.0] * 19   # 5% missing — but indicator must appear

        imp = MedianImputerWithIndicators(missing_threshold=0.2).fit(
            _make_df({"c": train_vals})
        )
        val_out = imp.transform(_make_df({"c": val_vals}))

        assert "c" in imp.indicator_cols_
        assert "c_missing" in val_out.columns

    def test_low_train_missingness_no_indicator_in_val(self):
        train_vals = [np.nan] * 1  + [2.0] * 19   # 5% missing → no indicator
        val_vals   = [np.nan] * 10 + [2.0] * 10   # 50% missing — must NOT appear

        imp = MedianImputerWithIndicators(missing_threshold=0.2).fit(
            _make_df({"d": train_vals})
        )
        val_out = imp.transform(_make_df({"d": val_vals}))

        assert "d" not in imp.indicator_cols_
        assert "d_missing" not in val_out.columns, \
            "Indicator must not appear — missingness threshold is train-only"

    def test_indicator_reflects_actual_missingness_in_split(self):
        train_vals = [np.nan] * 10 + [3.0] * 10
        val_vals   = [np.nan] * 3  + [3.0] * 17   # only 3 missing in val

        imp = MedianImputerWithIndicators(missing_threshold=0.2).fit(
            _make_df({"e": train_vals})
        )
        val_out = imp.transform(_make_df({"e": val_vals}))

        assert val_out["e_missing"].sum() == 3, \
            "Indicator should flag 3 missing rows in val, not 10 (train count)"


# ── WinsorizerTransformer ─────────────────────────────────────────────────────

class TestWinsorizerFitOnTrainOnly:
    """IQR fences must be fit on training data only."""

    def test_val_clipped_with_training_fences(self):
        # Training: feature f in [0, 100], IQR ~ [25, 75], fence = [25 - 3*50, 75 + 3*50] = [-125, 225]
        # Val: value 300 — above training upper fence, must be clipped to 225
        train = _make_df({"f": list(range(0, 101, 5))})   # 21 evenly-spaced values
        val   = _make_df({"f": [300.0] * 20})

        wsr = WinsorizerTransformer(iqr_scale=3.0).fit(train)
        val_out = wsr.transform(val)

        assert (val_out["f"] <= wsr.upper_["f"]).all(), \
            "Val values above training upper fence must be clipped"

    def test_zero_iqr_column_not_clipped(self):
        # All values identical → IQR = 0 → must be skipped entirely
        train = _make_df({"g": [0.0] * 20})
        val   = _make_df({"g": [999.0] * 20})   # extreme values — must pass through unchanged

        wsr = WinsorizerTransformer().fit(train)
        val_out = wsr.transform(val)

        assert "g" in wsr.zero_iqr_cols_, "Zero-IQR column must be in zero_iqr_cols_"
        assert (val_out["g"] == 999.0).all(), \
            "Zero-IQR column must not be clipped (IQR = 0 means fence is undefined)"

    def test_indicator_columns_not_clipped(self):
        # Binary _missing columns must never be clipped (0/1 are valid values)
        train = _make_df({"h": list(range(20)), "h_missing": [0, 1] * 10})
        val   = _make_df({"h": [9999.0] * 20, "h_missing": [1] * 20})

        wsr = WinsorizerTransformer().fit(train)
        val_out = wsr.transform(val)

        assert (val_out["h_missing"] == 1).all(), \
            "_missing indicator columns must not be clipped"

    def test_fences_not_recomputed_on_transform(self):
        train = _make_df({"i": list(range(20))})
        extreme_val = _make_df({"i": [1e9] * 20})

        wsr = WinsorizerTransformer().fit(train)
        upper_before = wsr.upper_["i"]
        wsr.transform(extreme_val)

        assert wsr.upper_["i"] == upper_before, \
            "IQR fences must not change when transform is called"


class TestClippingReport:
    def test_clipping_report_returns_zero_for_in_range_data(self):
        train = _make_df({"j": list(range(20))})
        wsr = WinsorizerTransformer().fit(train)
        report = wsr.clipping_report(train)
        # In-range training data should show near-zero clipping at 3×IQR
        assert report["j"] < 0.20, "Training data should not be heavily clipped at 3×IQR"


# ── Mutation-killing tests (added after mutation test run) ────────────────────

class TestMissingThresholdBoundary:
    def test_indicator_created_at_exact_threshold(self):
        # Feature at exactly 20% missing must be flagged — kills >= -> > mutant
        vals = [np.nan] * 4 + [1.0] * 16   # exactly 20%
        imp = MedianImputerWithIndicators(missing_threshold=0.2).fit(
            _make_df({"a": vals})
        )
        assert "a" in imp.indicator_cols_, \
            "Column at exactly the threshold must get an indicator (>= not >)"


class TestImputationUsesMedianNotMean:
    def test_skewed_distribution_imputed_with_median(self):
        # median=1.0, mean=21.0 — distinguishable on NaN val
        # Kills median() -> mean() mutant
        train = _make_df({"d": [1.0] * 4 + [100.0]})
        val   = _make_df({"d": [np.nan] * 5})
        imp = MedianImputerWithIndicators().fit(train)
        out = imp.transform(val)
        assert (out["d"] == 1.0).all(), \
            f"Must impute with median 1.0, not mean 21.0 (got {out['d'].iloc[0]})"


class TestColumnDropperBoundary:
    def test_column_at_exact_threshold_is_dropped(self):
        # Column at exactly 50% missing must be dropped — kills >= -> > mutant
        df = pd.DataFrame({"a": [np.nan, np.nan, 1.0, 1.0]})   # exactly 50%
        dropper = ColumnDropper(threshold=0.5).fit(df)
        assert "a" in dropper.cols_to_drop_, \
            "Column at exactly the threshold must be dropped (>= not >)"


class TestIQRScaleRespected:
    def test_value_within_3x_but_beyond_1x_iqr_not_clipped(self):
        # list(range(20)): Q1~4.75, Q3~14.25, IQR~9.5
        # 1×IQR upper fence ~23.75 — value 25 would be clipped at 1×IQR
        # 3×IQR upper fence ~42.75 — value 25 must pass through at 3×IQR
        # Kills iqr_scale-ignored mutant (always 1×)
        train = _make_df({"c": list(range(20))})
        val   = _make_df({"c": [25.0] * 20})
        wsr = WinsorizerTransformer(iqr_scale=3.0).fit(train)
        out = wsr.transform(val)
        assert (out["c"] == 25.0).all(), \
            f"Value within 3×IQR must not be clipped (iqr_scale=3.0, got {out['c'].iloc[0]})"


class TestTransformColumnGuard:
    def test_transform_skips_columns_absent_from_input(self):
        # Winsoriser fitted on two columns; transform called with only one.
        # Guard `if col in out.columns` must prevent KeyError.
        # Kills guard-removal mutant.
        train = _make_df({"e": list(range(20)), "f": list(range(20))})
        val   = _make_df({"e": list(range(20))})   # 'f' absent
        wsr = WinsorizerTransformer().fit(train)
        out = wsr.transform(val)
        assert "e" in out.columns, "Present column must survive transform"
