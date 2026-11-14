"""Checks of the ICU ward replay page (synthetic fixture only)."""

import json
import re
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from sepsis_warning import ward
from sepsis_warning.cli import run_ward
from sepsis_warning.data import generate_synthetic, load_patients, split_patients
from sepsis_warning.features import META, feature_table
from sepsis_warning.scoring import score_patients
from sepsis_warning.service import ScoringService


def page_payload(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(re.search(r"const data = (\{.*?\});\n", text, re.S).group(1))


class WardReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data = cls.root / "synthetic"
        generate_synthetic(cls.data, source_size=80, external_size=20)
        (cls.data / "SYNTHETIC.json").write_text("{}")
        cls.splits = split_patients(load_patients(cls.data), 42)
        train = feature_table(cls.splits["train"], 6)
        columns = [c for c in train if c not in META]
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              LogisticRegression(max_iter=500)).fit(train[columns], train["label"])
        cls.bundle = {"model": model, "model_name": "test_model", "feature_columns": columns, "window": 6,
                      "threshold": 0.3, "cooldown": 6, "synthetic": True}
        cls.model_path = cls.root / "model.joblib"
        joblib.dump(cls.bundle, cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def stays(self, beds=6, seed=1):
        return ward.choose_stays(self.splits["validation"], beds, 0.5, 72, seed)

    def test_choose_stays_is_seeded_mixed_and_trimmed(self):
        stays = self.stays()
        self.assertEqual([s.patient_id for s in stays], [s.patient_id for s in self.stays()])
        self.assertTrue(all(len(s.frame) <= 72 for s in stays))
        self.assertTrue(any(s.septic for s in stays))
        self.assertTrue(any(not s.septic for s in stays))
        with self.assertRaises(ValueError):
            ward.choose_stays(self.splits["validation"], 0, 0.3, 30, 1)

    def test_test_splits_need_the_freeze_flag(self):
        for split in ward.HELD_OUT_SPLITS:
            with self.assertRaises(ValueError):
                ward.check_split(split, after_freeze=False)
            ward.check_split(split, after_freeze=True)
        ward.check_split("validation", after_freeze=False)

    def test_trimming_does_not_change_scores_on_shown_hours(self):
        stays = self.stays()
        trimmed = ward.score_locally(self.bundle, stays)
        full_cohort = [p for p in self.splits["validation"] if p.patient_id in {s.patient_id for s in stays}]
        full = score_patients(self.bundle, full_cohort)
        for stay, frame in zip(stays, trimmed, strict=True):
            reference = full.loc[full["patient_id"] == stay.patient_id, "probability"].to_numpy()[:len(frame)]
            np.testing.assert_allclose(frame["probability"].to_numpy(), reference)

    def test_api_scoring_matches_in_process_scoring_and_sends_no_label(self):
        stays = self.stays()
        service = ScoringService(self.bundle)

        def fake_post(url, body, timeout):
            self.assertTrue(url.endswith("/score"))
            self.assertTrue(all("SepsisLabel" not in hour for hour in body["hours"]))
            return service.score(body["patient_id"], body["hours"])

        with mock.patch.object(ward, "_post_json", side_effect=fake_post):
            frames, details = ward.score_via_api("http://127.0.0.1:8000", stays)
        local = ward.score_locally(self.bundle, stays)
        for api_frame, local_frame in zip(frames, local, strict=True):
            np.testing.assert_allclose(api_frame["probability"], local_frame["probability"], rtol=1e-6)
            np.testing.assert_array_equal(api_frame["alert_event"], local_frame["alert_event"])
        self.assertEqual(details["model_name"], "test_model")

    def test_page_has_no_record_ids_escapes_markup_and_refuses_overwrite(self):
        stays = self.stays()
        payload = ward.build_payload(
            ward.score_locally(self.bundle, stays), model_name="m</script>", threshold=0.3, cooldown=6,
            synthetic=True, split="validation", scored_by="test", cohort_septic_share=0.2, seed=1)
        self.assertTrue(all(0 <= bed["offset"] <= ward.MAX_OFFSET_HOURS for bed in payload["beds"]))
        path = self.root / "page" / "ward.html"
        ward.write_ward(path, payload)
        text = path.read_text(encoding="utf-8")
        self.assertNotIn("synthetic_", text)
        self.assertNotIn("m</script>", text)
        self.assertEqual(page_payload(path)["model_name"], "m</script>")
        with self.assertRaises(ValueError):
            ward.write_ward(path, payload)

    def test_cli_writes_the_page_and_checks_synthetic_agreement(self):
        args = Namespace(model=self.model_path, api_url=None, data=self.data, splits=None, seed=42,
                         split="validation", after_freeze=False, beds=8, septic_share=0.3, max_hours=48,
                         output=self.root / "cli" / "ward.html")
        run_ward(args)
        payload = page_payload(args.output)
        self.assertEqual(len(payload["beds"]), 8)
        self.assertTrue(payload["synthetic"])
        real_bundle = dict(self.bundle, synthetic=False)
        real_path = self.root / "real.joblib"
        joblib.dump(real_bundle, real_path)
        with self.assertRaises(ValueError):
            run_ward(Namespace(**{**vars(args), "model": real_path, "output": self.root / "cli" / "other.html"}))


if __name__ == "__main__":
    unittest.main()
