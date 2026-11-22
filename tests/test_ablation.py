"""H03 ablation helpers: arm columns, NumPy metrics and the paired bootstrap.

    MPLBACKEND=Agg .venv/bin/python -m unittest tests.test_ablation -v

scikit-learn is used here only as a reference to check the NumPy metrics.
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from sepsis_warning.ablation import (
    ARMS,
    arm_columns,
    auroc,
    average_precision,
    brier,
    paired_bootstrap,
    score_table,
)

BASE = ["HR__last", "MAP__last", "Age"]
GROUPS = {"hmm": ["hmm_state_0", "hmm_state_1", "hmm_state_2"], "signature": ["sig_inc_time", "sig_inc_HR"]}


def validation_table(rng, stays=40, signal=1.0):
    """Hourly rows for several stays; septic stays score higher by `signal`."""
    chunks = []
    for k in range(stays):
        hours = int(rng.integers(5, 30))
        label = np.zeros(hours, dtype=int)
        if k % 3 == 0:
            label[int(rng.integers(0, hours)):] = 1
        chunks.append(pd.DataFrame({
            "patient_id": f"A:p{k:05d}", "hour": np.arange(1, hours + 1), "label": label,
            "noise": rng.normal(size=hours),
        }))
    table = pd.concat(chunks, ignore_index=True)
    table["probability"] = 1 / (1 + np.exp(-(signal * table["label"] + table["noise"] - 2)))
    return table.drop(columns="noise")


class ArmColumns(unittest.TestCase):
    def test_reference_arm_is_the_shared_columns(self):
        self.assertEqual(arm_columns(BASE, GROUPS, "A"), BASE)

    def test_shared_columns_come_first_in_every_arm(self):
        for arm in ARMS:
            columns = arm_columns(BASE, GROUPS, arm)
            self.assertEqual(columns[: len(BASE)], BASE)
            self.assertEqual(len(columns), len(set(columns)))

    def test_each_arm_adds_its_groups_in_order(self):
        self.assertEqual(arm_columns(BASE, GROUPS, "B"), BASE + GROUPS["hmm"])
        self.assertEqual(arm_columns(BASE, GROUPS, "C"), BASE + GROUPS["signature"])
        self.assertEqual(arm_columns(BASE, GROUPS, "D"), BASE + GROUPS["hmm"] + GROUPS["signature"])

    def test_does_not_modify_the_shared_list(self):
        base = list(BASE)
        arm_columns(base, GROUPS, "D")
        self.assertEqual(base, BASE)

    def test_unknown_arm_fails(self):
        with self.assertRaises(KeyError):
            arm_columns(BASE, GROUPS, "E")


class MetricsMatchScikitLearn(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_average_precision(self):
        for _ in range(20):
            labels = self.rng.random(300) < 0.1
            labels[0] = True
            scores = self.rng.random(300) + labels * 0.3
            self.assertAlmostEqual(average_precision(labels, scores), average_precision_score(labels, scores))

    def test_average_precision_with_tied_scores(self):
        labels = self.rng.random(500) < 0.2
        labels[0] = True
        scores = np.round(self.rng.random(500) + labels * 0.2, 1)    # many ties
        self.assertAlmostEqual(average_precision(labels, scores), average_precision_score(labels, scores))

    def test_auroc_with_tied_scores(self):
        labels = self.rng.random(500) < 0.2
        labels[:2] = [True, False]
        scores = np.round(self.rng.random(500) + labels * 0.2, 1)
        self.assertAlmostEqual(auroc(labels, scores), roc_auc_score(labels, scores))

    def test_brier(self):
        labels = self.rng.random(200) < 0.3
        probabilities = self.rng.random(200)
        self.assertAlmostEqual(brier(labels, probabilities), brier_score_loss(labels, probabilities))

    def test_perfect_ranking(self):
        self.assertEqual(average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_one_class_gives_nan_not_an_error(self):
        self.assertTrue(np.isnan(average_precision([0, 0, 0], [0.1, 0.5, 0.9])))
        self.assertTrue(np.isnan(auroc([1, 1, 1], [0.1, 0.5, 0.9])))
        self.assertTrue(np.isnan(auroc([0, 0, 0], [0.1, 0.5, 0.9])))


class ScoreTable(unittest.TestCase):
    def test_first_six_hours_uses_only_early_rows(self):
        table = validation_table(np.random.default_rng(1))
        early = table[table["hour"] <= 6]
        metrics = score_table(table)
        self.assertEqual(metrics["hours"], len(table))
        self.assertAlmostEqual(metrics["auprc_first_6h"], average_precision(early["label"], early["probability"]))
        self.assertAlmostEqual(metrics["auprc"], average_precision(table["label"], table["probability"]))

    def test_late_rows_do_not_change_first_six_hours(self):
        table = validation_table(np.random.default_rng(2))
        changed = table.copy()
        late = changed["hour"] > 6
        changed.loc[late, "probability"] = 1 - changed.loc[late, "probability"]
        self.assertEqual(score_table(table)["auprc_first_6h"], score_table(changed)["auprc_first_6h"])


class PairedBootstrap(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.weak = validation_table(rng, stays=60, signal=0.2)
        self.strong = self.weak.copy()
        self.strong["probability"] = 1 / (1 + np.exp(-(4 * self.weak["label"] - 2)))

    def test_identical_arms_have_zero_difference(self):
        result = paired_bootstrap({"A": self.weak, "B": self.weak.copy()}, resamples=50)
        self.assertEqual(result["B"]["auprc_difference"], 0.0)
        self.assertEqual((result["B"]["ci_low"], result["B"]["ci_high"]), (0.0, 0.0))
        self.assertFalse(result["B"]["helps"])

    def test_clearly_better_arm_helps(self):
        result = paired_bootstrap({"A": self.weak, "B": self.strong}, resamples=100)
        self.assertGreater(result["B"]["ci_low"], 0)
        self.assertTrue(result["B"]["helps"])
        self.assertLessEqual(result["B"]["ci_low"], result["B"]["auprc_difference"])
        self.assertGreaterEqual(result["B"]["ci_high"], result["B"]["auprc_difference"])

    def test_worse_arm_does_not_help(self):
        result = paired_bootstrap({"A": self.strong, "B": self.weak}, resamples=100)
        self.assertLess(result["B"]["auprc_difference"], 0)
        self.assertFalse(result["B"]["helps"])

    def test_reference_arm_is_not_reported(self):
        result = paired_bootstrap({"A": self.weak, "B": self.strong, "C": self.weak}, resamples=20)
        self.assertEqual(set(result), {"B", "C"})

    def test_same_seed_gives_same_interval(self):
        arms = {"A": self.weak, "B": self.strong}
        self.assertEqual(paired_bootstrap(arms, resamples=30, seed=5), paired_bootstrap(arms, resamples=30, seed=5))

    def test_every_arm_sees_the_same_resamples(self):
        # C is B with every score shifted, so its ranking and AUPRC match B on every resample.
        shifted = self.strong.copy()
        shifted["probability"] = shifted["probability"] * 0.5 + 0.1
        result = paired_bootstrap({"A": self.weak, "B": self.strong, "C": shifted}, resamples=50)
        self.assertAlmostEqual(result["B"]["ci_low"], result["C"]["ci_low"])
        self.assertAlmostEqual(result["B"]["ci_high"], result["C"]["ci_high"])

    def test_rows_that_do_not_line_up_fail(self):
        reordered = self.strong.iloc[::-1].reset_index(drop=True)
        with self.assertRaises(ValueError):
            paired_bootstrap({"A": self.weak, "B": reordered}, resamples=5)


if __name__ == "__main__":
    unittest.main()
