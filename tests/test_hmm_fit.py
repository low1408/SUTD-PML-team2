import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis_warning.hmm import fit_hmm, forward_backward, hmm_state_features, load_hmm, save_hmm, vital_observations

START = np.array([0.8, 0.2])
TRANSITION = np.array([[0.9, 0.1], [0.3, 0.7]])
HIGH_HEART_RATE = np.array([0.2, 0.7])


def simulate(n_stays, hours, seed):
    """Two states: heart rate about 80 (sd 8) or about 110 (sd 10)."""
    rng = np.random.default_rng(seed)
    means, stds = np.array([80.0, 110.0]), np.array([8.0, 10.0])
    stays = []
    for _ in range(n_stays):
        state = rng.choice(2, p=START)
        rows = []
        for _ in range(hours):
            rows.append([rng.normal(means[state], stds[state])])
            state = rng.choice(2, p=TRANSITION[state])
        stays.append(np.array(rows))
    return stays


class BaumWelch(unittest.TestCase):
    def test_e_step_matches_hand_calculation(self):
        log_emission = np.log(np.array([HIGH_HEART_RATE, HIGH_HEART_RATE]))
        gamma, xi_sum, log_likelihood = forward_backward(START, TRANSITION, log_emission)
        np.testing.assert_allclose(gamma[0], [0.04 / 0.117, 0.077 / 0.117])
        np.testing.assert_allclose(xi_sum, np.array([[0.0288, 0.0112], [0.0084, 0.0686]]) / 0.117)
        self.assertAlmostEqual(log_likelihood, np.log(0.117))

    def test_log_likelihood_never_decreases(self):
        _, history = fit_hmm(simulate(30, 40, seed=0), n_states=2, tol=0)
        self.assertTrue(np.all(np.diff(history) >= -1e-6))

    def test_recovers_known_parameters(self):
        model, _ = fit_hmm(simulate(200, 50, seed=1), n_states=2)
        order = np.argsort(model.means[:, 0])
        np.testing.assert_allclose(model.means[order, 0], [80, 110], atol=2)
        np.testing.assert_allclose(model.stds[order, 0], [8, 10], atol=1.5)
        np.testing.assert_allclose(model.transition[np.ix_(order, order)], TRANSITION, atol=0.05)

    def test_missing_hours_are_allowed(self):
        stays = simulate(30, 40, seed=2)
        stays[0][5:10] = np.nan
        model, _ = fit_hmm(stays, n_states=2)
        probabilities = model.filter(stays[0])
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(40))

    def test_same_result_every_run(self):
        stays = simulate(20, 30, seed=3)
        first, _ = fit_hmm(stays, n_states=2)
        second, _ = fit_hmm(stays, n_states=2)
        np.testing.assert_array_equal(first.means, second.means)

    def test_filter_does_not_look_ahead(self):
        stays = simulate(30, 40, seed=4)
        model, _ = fit_hmm(stays, n_states=2)
        changed = stays[0].copy()
        changed[20:] = 200.0
        np.testing.assert_array_equal(model.filter(stays[0])[:20], model.filter(changed)[:20])


class RealDataHelpers(unittest.TestCase):
    def test_implausible_vitals_become_missing(self):
        frame = pd.DataFrame({"HR": [80, 0, 300], "MAP": [70, 70, 70], "Resp": [16, 2, 20], "O2Sat": [97, 97, 120]})
        values = vital_observations(frame)
        self.assertTrue(np.isnan(values[1, 0]) and np.isnan(values[2, 0]))
        self.assertTrue(np.isnan(values[1, 2]) and np.isnan(values[2, 3]))
        self.assertEqual(values[0].tolist(), [80, 70, 16, 97])

    def test_features_have_one_row_per_hour_and_survive_save_load(self):
        rng = np.random.default_rng(5)
        stays = [np.column_stack([rng.normal(80, 8, 30), rng.normal(75, 8, 30), rng.normal(18, 3, 30), rng.normal(97, 1, 30)])
                 for _ in range(20)]
        model, _ = fit_hmm(stays, n_states=3)
        frame = pd.DataFrame(stays[0], columns=["HR", "MAP", "Resp", "O2Sat"])
        frame.loc[3:6, "MAP"] = np.nan
        features = hmm_state_features(frame, model)
        self.assertEqual(list(features.columns), ["hmm_state_0", "hmm_state_1", "hmm_state_2"])
        self.assertEqual(len(features), 30)
        np.testing.assert_allclose(features.sum(axis=1), np.ones(30))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "hmm.json"
            save_hmm(model, path)
            loaded, columns = load_hmm(path)
        self.assertEqual(columns, ["HR", "MAP", "Resp", "O2Sat"])
        pd.testing.assert_frame_equal(hmm_state_features(frame, loaded), features)


if __name__ == "__main__":
    unittest.main()
