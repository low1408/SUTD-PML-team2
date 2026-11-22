"""Phase 2: traceable responses, input ranges, limits, alert ledger, client resilience, telemetry."""

import importlib.util
import io
import itertools
import json
import random
import tempfile
import unittest
import urllib.error
from pathlib import Path

import numpy as np

from sepsis_warning.alert_store import (
    DynamoAlertStore,
    InMemoryAlertStore,
    LedgerConflict,
    ledger_events,
    stay_key,
)
from sepsis_warning.client import (
    ScoringClient,
    ScoringRejected,
    ScoringUnavailable,
    sirs_fallback,
)
from sepsis_warning.data import DYNAMIC, generate_synthetic, load_patients
from sepsis_warning.service import (
    MAX_TOTAL_HOURS,
    InvalidInput,
    ScoringService,
    TooLarge,
    risk_tier,
)
from sepsis_warning.telemetry import Telemetry

HAVE_API = all(importlib.util.find_spec(name) for name in ("fastapi", "httpx"))


def make_service(**options):
    from test_service import make_bundle

    temp = tempfile.TemporaryDirectory()
    generate_synthetic(Path(temp.name), source_size=60, external_size=20)
    patients = load_patients(Path(temp.name))
    return ScoringService(make_bundle(patients), "a" * 64, **options), patients, temp


def rising(hours: int) -> list[dict]:
    """A stay that deteriorates: scores cross any reasonable threshold."""
    return [{"ICULOS": h, "HR": 80 + 6 * h, "Resp": 14 + 2 * h, "Temp": 36.8 + 0.25 * h, "MAP": 85 - 3 * h}
            for h in range(1, hours + 1)]


class FakeDynamo:
    """get_item/update_item with the condition semantics the store relies on."""

    def __init__(self):
        self.items = {}

    def get_item(self, TableName, Key, ConsistentRead):
        item = self.items.get(Key["stayId"]["S"])
        return {"Item": item} if item else {}

    def update_item(self, TableName, Key, UpdateExpression, ConditionExpression, ExpressionAttributeValues):
        key, values = Key["stayId"]["S"], ExpressionAttributeValues
        item = self.items.get(key)
        ok = item is None if ConditionExpression.startswith("attribute_not_exists") else (
            item is not None and item["version"] == values[":expected"])
        if not ok:
            error = RuntimeError("conditional check failed")
            error.response = {"Error": {"Code": "ConditionalCheckFailedException"}}
            raise error
        existing = item["alerts"]["L"] if item else []
        self.items[key] = {"stayId": {"S": key}, "version": values[":next"], "expiresAt": values[":ttl"],
                           "alerts": {"L": existing + values[":new"]["L"]}}


class TraceabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service, cls.patients, cls.temp = make_service(model_version="7")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_responses_name_the_model_and_have_stable_prediction_ids(self):
        first = self.service.score("bed-1", rising(6))
        again = self.service.score("bed-1", rising(6))
        self.assertEqual(first["model_version"], "7")
        self.assertEqual(first["bundle_sha256"], "a" * 64)
        self.assertEqual(first["alert_source"], "stateless_replay")
        ids = [h["prediction_id"] for h in first["hours"]]
        self.assertEqual(ids, [h["prediction_id"] for h in again["hours"]])
        self.assertEqual(len(set(ids)), 6)
        self.assertNotEqual(ids[-1], self.service.score("bed-2", rising(6))["hours"][-1]["prediction_id"])
        batch = self.service.predict([{"patient_id": "bed-1", "hours": rising(6)}])
        self.assertEqual(batch["predictions"][0]["prediction_id"], ids[-1])
        self.assertEqual(batch["model_version"], "7")

    def test_risk_tiers_follow_the_threshold(self):
        self.assertEqual([risk_tier(p, 0.4) for p in (0.1, 0.2, 0.39, 0.4, 0.9)],
                         ["low", "watch", "watch", "alert", "alert"])
        for hour in self.service.score("bed-1", rising(8))["hours"]:
            self.assertEqual(hour["risk_tier"] == "alert", bool(hour["hourly_positive"]))

    def test_impossible_values_are_rejected_and_odd_ones_warned(self):
        for field, value in (("HR", 9999), ("Temp", 101.3), ("Lactate", -1), ("pH", 74), ("Gender", 2)):
            with self.assertRaisesRegex(InvalidInput, "impossible"):
                self.service.score("bed-1", [{"ICULOS": 1, field: value}])
        result = self.service.score("bed-1", [{"ICULOS": 1, "HR": 260, "FiO2": 40}, {"ICULOS": 2, "BaseExcess": -12}])
        self.assertEqual(len(result["input_warnings"]), 2)
        self.assertIn("HR=260", result["input_warnings"][0])
        self.assertEqual(self.service.score("bed-1", rising(3))["input_warnings"], [])

    def test_seeded_fuzz_never_crashes_the_service(self):
        rng = random.Random(7)
        fields = [*DYNAMIC, "Age", "Gender", "HospAdmTime"]
        outcomes = {"ok": 0, "rejected": 0}
        for _ in range(150):
            hours = []
            for h in range(1, rng.randint(1, 6) + 1):
                record = {"ICULOS": h}
                for name in rng.sample(fields, rng.randint(0, 8)):
                    record[name] = rng.choice([None, rng.uniform(-50, 500), rng.uniform(0, 150), 0, 1e9])
                hours.append(record)
            try:
                result = self.service.score("fuzz", hours)
                self.assertTrue(all(0 <= x["probability"] <= 1 for x in result["hours"]))
                outcomes["ok"] += 1
            except InvalidInput:
                outcomes["rejected"] += 1
        self.assertGreater(outcomes["ok"], 0)
        self.assertGreater(outcomes["rejected"], 0)

    def test_total_hours_are_capped(self):
        stays = [{"patient_id": f"bed-{i}", "hours": [{"ICULOS": h} for h in range(1, 1001)]} for i in range(9)]
        self.assertGreater(9 * 1000, MAX_TOTAL_HOURS)
        with self.assertRaisesRegex(TooLarge, "smaller batches"):
            self.service.predict(stays)


class AlertLedgerTests(unittest.TestCase):
    def test_ledger_events_continue_from_delivered_alerts(self):
        hours, p = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0.9] * 10
        events, new = ledger_events(hours, p, 0.5, 6, [])
        self.assertEqual(events.tolist(), [1, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        self.assertEqual(new, [0, 6])
        # Delivered at hour 3 (by an older model): history shows hour 3, and the cooldown runs from it.
        events, new = ledger_events(hours, p, 0.5, 6, [3])
        self.assertEqual(events.tolist(), [0, 0, 1, 0, 0, 0, 0, 0, 1, 0])
        self.assertEqual(new, [8])
        # A delivered alert stays in the history even if the current model would not raise it.
        events, new = ledger_events(hours, [0.0] * 10, 0.5, 6, [3])
        self.assertEqual((events.tolist(), new), ([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], []))

    def run_model_swap(self, store):
        service, _, temp = make_service(alert_store=store, model_version="1")
        self.addCleanup(temp.cleanup)
        first = service.score("bed-9", rising(8))
        alerts_v1 = [h["hour"] for h in first["hours"] if h["alert_event"]]
        self.assertTrue(alerts_v1, "the deteriorating stay should alert")
        self.assertEqual(first["alert_source"], "ledger")
        self.assertEqual([a["hour"] for a in store.get("bed-9")[0]], alerts_v1)
        # Same request again (a retry): nothing new is recorded.
        service.score("bed-9", rising(8))
        self.assertEqual(len(store.get("bed-9")[0]), len(alerts_v1))
        # "New model": a much lower threshold would, statelessly, alert at hour 1 and re-alert early.
        swapped = ScoringService(dict(service.bundle, threshold=0.0), "b" * 64, alert_store=store, model_version="2")
        stateless = ScoringService(dict(service.bundle, threshold=0.0), "b" * 64).score("bed-9", rising(9))
        self.assertEqual(stateless["hours"][0]["alert_event"], 1)
        after = swapped.score("bed-9", rising(9))
        alerts_v2 = [h["hour"] for h in after["hours"] if h["alert_event"]]
        self.assertEqual(alerts_v2[:len(alerts_v1)], alerts_v1)            # history unchanged
        for earlier, later in itertools.pairwise(alerts_v2):
            self.assertGreaterEqual(later - earlier, service.bundle["cooldown"])  # no alert inside the cooldown
        recorded = store.get("bed-9")[0]
        self.assertEqual({a["model_version"] for a in recorded} - {"1"}, {"2"} if len(alerts_v2) > len(alerts_v1) else set())

    def test_model_swap_does_not_realert_in_memory(self):
        self.run_model_swap(InMemoryAlertStore())

    def test_model_swap_does_not_realert_in_dynamodb(self):
        fake = FakeDynamo()
        self.run_model_swap(DynamoAlertStore("table", client=fake))
        item = fake.items[stay_key("bed-9")]
        self.assertNotIn("bed-9", json.dumps(item))            # only the hashed key is stored
        self.assertGreater(int(item["expiresAt"]["N"]), 0)

    def test_conflicting_writers_do_not_record_an_alert_twice(self):
        for store in (InMemoryAlertStore(), DynamoAlertStore("table", client=FakeDynamo())):
            _, version = store.get("bed-1")
            alert = {"hour": 4.0, "prediction_id": "p", "model_version": "1", "at": "t"}
            store.append("bed-1", [alert], version)
            with self.assertRaises(LedgerConflict):
                store.append("bed-1", [alert], version)       # stale version: must re-read
            self.assertEqual(len(store.get("bed-1")[0]), 1)

    def test_service_retries_after_a_conflict(self):
        store = InMemoryAlertStore()
        service, _, temp = make_service(alert_store=store)
        self.addCleanup(temp.cleanup)
        original, raced = store.append, {"done": False}

        def racing_append(patient_id, alerts, expected):
            if not raced["done"]:
                raced["done"] = True
                original(patient_id, alerts, expected)   # the other request wins the race
                raise LedgerConflict(patient_id)
            return original(patient_id, alerts, expected)

        store.append = racing_append
        result = service.score("bed-3", rising(8))
        hours = [a["hour"] for a in store.get("bed-3")[0]]
        self.assertEqual(len(hours), len(set(hours)))
        self.assertEqual([h["hour"] for h in result["hours"] if h["alert_event"]], hours)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def http_error(code):
    return urllib.error.HTTPError("http://x", code, "error", {}, io.BytesIO(b'{"detail":"nope"}'))


class ClientTests(unittest.TestCase):
    def client(self, outcomes, **options):
        calls, sleeps = [], []

        def opener(request, timeout):
            calls.append(request)
            outcome = outcomes[min(len(calls), len(outcomes)) - 1]
            if isinstance(outcome, Exception):
                raise outcome
            return FakeResponse(json.dumps(outcome).encode())

        self.clock = [0.0]
        client = ScoringClient("http://api", opener=opener, sleep=sleeps.append, clock=lambda: self.clock[0],
                               rng=random.Random(1), **options)
        return client, calls, sleeps

    def test_retries_503_with_growing_jittered_backoff_then_succeeds(self):
        client, calls, sleeps = self.client([http_error(503), http_error(503), http_error(503), {"ok": True}])
        self.assertEqual(client.predict([{"patient_id": "b", "hours": [{"ICULOS": 1}]}]), {"ok": True})
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(sleeps), 3)
        for attempt, pause in enumerate(sleeps):
            self.assertLessEqual(pause, 0.5 * 2 ** attempt)
        self.assertIn("X-request-id", {k.capitalize() for k in calls[0].headers})

    def test_a_rejected_request_is_not_retried(self):
        client, calls, _ = self.client([http_error(422)])
        with self.assertRaises(ScoringRejected) as caught:
            client.score("b", [{"ICULOS": 1, "SepsisLabel": 1}])
        self.assertEqual((caught.exception.status, len(calls)), (422, 1))

    def test_gives_up_after_the_retries(self):
        client, calls, _ = self.client([urllib.error.URLError("down")], retries=2)
        with self.assertRaisesRegex(ScoringUnavailable, "3 attempt"):
            client.health()
        self.assertEqual(len(calls), 3)

    def test_circuit_opens_then_probes_again_after_the_pause(self):
        client, calls, _ = self.client([urllib.error.URLError("down")] * 4 + [{"ok": True}],
                                       retries=1, breaker_failures=4, breaker_reset=30)
        for _ in range(2):
            with self.assertRaises(ScoringUnavailable):
                client.health()
        self.assertTrue(client.circuit_open)
        with self.assertRaisesRegex(ScoringUnavailable, "circuit open"):
            client.health()
        self.assertEqual(len(calls), 4)                 # the open circuit made no call
        self.clock[0] = 31.0
        self.assertEqual(client.health(), {"ok": True})
        self.assertFalse(client.circuit_open)

    def test_fallback_returns_marked_sirs_results(self):
        client, _, _ = self.client([urllib.error.URLError("down")], retries=0)
        stays = [{"patient_id": "bed-1", "hours": [{"ICULOS": 1, "HR": 80}, {"ICULOS": 2, "HR": 120, "Resp": 26}]},
                 {"patient_id": "bed-2", "hours": [{"ICULOS": 1, "HR": 70, "Temp": 37.0}]}]
        result = client.predict_or_fallback(stays)
        self.assertTrue(result["fallback"])
        self.assertEqual(result["model_name"], "sirs_rule_fallback")
        first, second = result["predictions"]
        self.assertEqual((first["sirs_criteria_met"], first["above_threshold"], first["new_alert"]), (2, True, True))
        self.assertEqual((second["sirs_criteria_met"], second["above_threshold"]), (0, False))
        self.assertIn("down", result["reason"])
        self.assertEqual(sirs_fallback(stays)["predictions"], result["predictions"])

    def test_no_fallback_when_the_service_answers(self):
        client, _, _ = self.client([{"predictions": [], "model_name": "m"}])
        self.assertFalse(client.predict_or_fallback([{"patient_id": "b", "hours": [{"ICULOS": 1}]}])["fallback"])


@unittest.skipUnless(HAVE_API, "needs `pip install -e .[serve,test]`")
class ApiHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from sepsis_warning.api import create_app

        cls.service, cls.patients, cls.temp = make_service(model_version="3")
        cls.stream = io.StringIO()
        cls.client = TestClient(create_app(cls.service, None, Telemetry(stream=cls.stream, environment="test")))

    @classmethod
    def tearDownClass(cls):
        cls.service.on_prediction = None
        cls.temp.cleanup()

    def lines(self):
        return [json.loads(line) for line in self.stream.getvalue().splitlines()]

    def test_request_id_is_echoed_or_generated(self):
        echoed = self.client.get("/health", headers={"X-Request-Id": "abc-123"})
        self.assertEqual(echoed.headers["x-request-id"], "abc-123")
        self.assertEqual(len(self.client.get("/health").headers["x-request-id"]), 32)

    def test_oversized_bodies_get_413_before_parsing(self):
        response = self.client.post("/predict", content=b"x" * 10, headers={"content-length": "6000000",
                                                                            "content-type": "application/json"})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"], "too_large")
        many = {"stays": [{"patient_id": f"b{i}", "hours": [{"ICULOS": h} for h in range(1, 1001)]} for i in range(9)]}
        self.assertEqual(self.client.post("/predict", json=many).status_code, 413)

    def test_malformed_values_use_the_same_error_shape(self):
        for value in ("92", True, [1]):
            response = self.client.post("/score", json={"patient_id": "b", "hours": [{"ICULOS": 1, "HR": value}]})
            self.assertEqual(response.status_code, 422, value)
            self.assertEqual(response.json()["error"], "invalid_input")
        impossible = self.client.post("/score", json={"patient_id": "b", "hours": [{"ICULOS": 1, "HR": 9999}]})
        self.assertIn("impossible", impossible.json()["detail"])

    def test_telemetry_has_metrics_and_predictions_but_no_measurements(self):
        self.stream.truncate(0)
        self.stream.seek(0)
        body = {"stays": [{"patient_id": "bed-secret-1", "hours": rising(5)}]}
        self.assertEqual(self.client.post("/predict", json=body, headers={"X-Request-Id": "r1"}).status_code, 200)
        self.client.post("/score", json={"patient_id": "b", "hours": [{"ICULOS": 1, "SepsisLabel": 1}]})
        records = self.lines()
        requests = [r for r in records if r["record_type"] == "request"]
        ok = next(r for r in requests if r["request_id"] == "r1")
        self.assertEqual((ok["Path"], ok["status"], ok["HoursScored"], ok["StaysScored"]), ("/predict", 200, 5, 1))
        self.assertEqual(ok["model_version"], "3")
        self.assertIn("LatencyMs", [m["Name"] for m in ok["_aws"]["CloudWatchMetrics"][0]["Metrics"]])
        self.assertEqual(requests[-1]["ClientErrors"], 1)
        prediction = next(r for r in records if r["record_type"] == "prediction")
        self.assertEqual(prediction["hour"], 5.0)
        self.assertIn("Lactate", prediction["missing_now"])
        self.assertNotIn("HR", prediction["missing_now"])
        text = self.stream.getvalue()
        self.assertNotIn("bed-secret-1", text)             # stay IDs are hashed
        for forbidden in ('"HR":', '"Temp":', '"hours":'):
            self.assertNotIn(forbidden, text)              # no measurement values in logs

    def test_openapi_documents_every_input_field(self):
        schema = self.client.get("/openapi.json").json()["components"]["schemas"]
        self.assertEqual(set(schema["HourRecord"]["properties"]), set(DYNAMIC) | {"Age", "Gender", "HospAdmTime", "ICULOS"})
        self.assertIn("prediction_id", schema["StayPrediction"]["properties"])
        self.assertIn("model_version", schema["PredictResponse"]["properties"])


class ThreadingTests(unittest.TestCase):
    def test_nested_estimators_predict_on_one_thread_and_there_is_no_lock(self):
        service, _, temp = make_service()
        self.addCleanup(temp.cleanup)
        self.assertFalse(hasattr(service, "_predict_lock"))
        from sepsis_warning.features import META, feature_table
        from sepsis_warning.modeling import candidate_models, fit_calibrated
        from sepsis_warning.service import _single_thread

        patients = load_patients(Path(temp.name))
        table = feature_table(patients[:40], 6)
        columns = [c for c in table if c not in META]
        forest = fit_calibrated(candidate_models(0)["random_forest"], table, table, columns)
        _single_thread(forest)
        inner = forest.calibrated_classifiers_[0].estimator.estimator[-1]
        self.assertEqual(inner.n_jobs, 1)
        self.assertTrue(np.all(np.isfinite(forest.predict_proba(table[columns]))))


if __name__ == "__main__":
    unittest.main()
