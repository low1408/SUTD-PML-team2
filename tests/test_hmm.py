import unittest

import numpy as np

from sepsis_warning.hmm import forward_filter, gaussian_emissions

START = np.array([0.8, 0.2])
TRANSITION = np.array([[0.9, 0.1], [0.3, 0.7]])
HIGH_HEART_RATE = np.array([0.2, 0.7])
MEANS = np.array([[80.0], [105.0]])
STDS = np.array([[10.0], [15.0]])


class ForwardFilter(unittest.TestCase):
    def test_matches_hand_calculation(self):
        emission = np.array([HIGH_HEART_RATE, HIGH_HEART_RATE])
        filtered = forward_filter(START, TRANSITION, emission)
        np.testing.assert_allclose(filtered[0], [0.16 / 0.30, 0.14 / 0.30])
        np.testing.assert_allclose(filtered[1], [0.124 / 0.39, 0.266 / 0.39])

    def test_each_hour_sums_to_one(self):
        emission = np.random.default_rng(0).uniform(0.01, 1, size=(20, 2))
        filtered = forward_filter(START, TRANSITION, emission)
        np.testing.assert_allclose(filtered.sum(axis=1), np.ones(20))

    def test_later_hours_do_not_change_earlier_ones(self):
        emission = np.random.default_rng(1).uniform(0.01, 1, size=(10, 2))
        changed = emission.copy()
        changed[6:] = np.random.default_rng(2).uniform(0.01, 1, size=(4, 2))
        before = forward_filter(START, TRANSITION, emission)
        after = forward_filter(START, TRANSITION, changed)
        np.testing.assert_array_equal(before[:6], after[:6])


class GaussianEmissions(unittest.TestCase):
    def test_matches_hand_calculation(self):
        emission = gaussian_emissions(np.array([[110.0]]), MEANS, STDS)
        self.assertAlmostEqual(emission[0, 1] / emission[0, 0], 0.025159 / 0.000443, delta=0.1)

    def test_hour_with_nothing_measured_gives_equal_values(self):
        emission = gaussian_emissions(np.array([[np.nan]]), MEANS, STDS)
        np.testing.assert_array_equal(emission, [[1.0, 1.0]])

    def test_missing_measurement_is_skipped(self):
        means = np.array([[80.0, 16.0], [105.0, 24.0]])
        stds = np.array([[10.0, 3.0], [15.0, 5.0]])
        with_gap = gaussian_emissions(np.array([[110.0, np.nan]]), means, stds)
        heart_only = gaussian_emissions(np.array([[110.0]]), MEANS, STDS)
        np.testing.assert_allclose(with_gap, heart_only)

    def test_rescaling_does_not_change_forward_filter(self):
        observations = np.array([[110.0], [np.nan], [85.0]])
        emission = gaussian_emissions(observations, MEANS, STDS)
        scaled = emission * np.array([[3.0], [7.0], [0.5]])
        np.testing.assert_allclose(
            forward_filter(START, TRANSITION, emission),
            forward_filter(START, TRANSITION, scaled),
        )


if __name__ == "__main__":
    unittest.main()