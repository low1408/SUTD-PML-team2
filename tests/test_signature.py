import unittest

import numpy as np
import pandas as pd

from sepsis_warning.features import apply_training_fallback
from sepsis_warning.signature import (
    _clean,
    feature_names,
    fit_signature_scaler,
    log_signature_depth2,
    signature_features,
)

def random_stay(rng, hours):
    frame = pd.DataFrame({
        "HR": rng.normal(85, 12, hours), "MAP": rng.normal(78, 10, hours),
        "Resp": rng.normal(18, 3, hours), "O2Sat": rng.normal(97, 1.5, hours),
    })
    frame = frame.mask(rng.random(frame.shape) < 0.15)        # about 15% missing
    return frame


class HandExamples(unittest.TestCase):
    def test_patient_heart_rate_first(self):
        result = log_signature_depth2(np.array([[80, 16], [90, 18], [100, 24]]))
        np.testing.assert_allclose(result, [20, 8, 20])         # increments 20 and 8, area +20

    def test_patient_breathing_first(self):
        result = log_signature_depth2(np.array([[80, 16], [90, 22], [100, 24]]))
        np.testing.assert_allclose(result, [20, 8, -20])        # same increments, area -20

    def test_one_point_gives_zeros(self):
        np.testing.assert_array_equal(log_signature_depth2(np.array([[80, 16]])), [0, 0, 0])


class SignatureFeatures(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.train = [random_stay(rng, 40) for _ in range(30)]
        self.scaler = fit_signature_scaler(self.train)
        self.stay = random_stay(rng, 30)

    def test_fifteen_named_columns_in_fixed_order(self):
        features = signature_features(self.stay, self.scaler)
        self.assertEqual(features.shape, (30, 15))
        self.assertEqual(list(features.columns), feature_names())
        self.assertEqual(features.columns[0], "sig_inc_time")
        self.assertEqual(features.columns[-1], "sig_area_Resp_O2Sat")

    def test_fast_version_matches_window_by_window(self):
        features = signature_features(self.stay, self.scaler, window=6).to_numpy()
        filled = apply_training_fallback(_clean(self.stay, self.scaler.columns), self.scaler.fallback).to_numpy()
        vitals = (filled - self.scaler.means) / self.scaler.stds
        path = np.column_stack([np.arange(30) / 5, vitals])
        for t in range(30):
            expected = log_signature_depth2(path[max(0, t - 5): t + 1])
            np.testing.assert_allclose(features[t], expected, atol=1e-9)

    def test_later_hours_do_not_change_earlier_ones(self):
        changed = self.stay.copy()
        changed.loc[12:, "HR"] = 190.0
        changed.loc[15:, "Resp"] = np.nan
        before = signature_features(self.stay, self.scaler).iloc[:12]
        after = signature_features(changed, self.scaler).iloc[:12]
        pd.testing.assert_frame_equal(before, after)

    def test_first_hour_is_zero_and_short_history_works(self):
        features = signature_features(self.stay.iloc[:3], self.scaler)
        np.testing.assert_array_equal(features.iloc[0].to_numpy(), np.zeros(15))
        self.assertEqual(len(features), 3)
        self.assertFalse(features.isna().any().any())

    def test_leading_missing_uses_training_median_not_a_later_value(self):
        stay = self.stay.iloc[:4].copy()
        stay.loc[0:1, "HR"] = np.nan
        stay.loc[2, "HR"] = 150.0
        early = signature_features(stay, self.scaler).iloc[:2]
        stay.loc[2, "HR"] = 60.0
        pd.testing.assert_frame_equal(early, signature_features(stay, self.scaler).iloc[:2])

    def test_scaler_uses_only_the_tables_it_is_given(self):
        other = fit_signature_scaler(self.train + [self.stay * 2])
        self.assertFalse(np.allclose(other.means, self.scaler.means))
        again = fit_signature_scaler(self.train)
        np.testing.assert_array_equal(again.means, self.scaler.means)


if __name__ == "__main__":
    unittest.main()
