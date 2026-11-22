"""Alert ledger: the alerts that were actually raised for a stay.

Scoring stays stateless: every request carries the full stay history. The one
thing worth remembering is which alerts were already raised. Without it the
cooldown is replayed from the history on every request, so after a model or
threshold change the same stay could alert again inside the cooldown, or an
alert staff already saw could disappear from the history.

With a ledger:

* hours up to the last recorded alert show exactly the recorded alerts;
* later hours continue the cooldown from that last recorded alert;
* each new alert is written once (optimistic locking makes retries and
  concurrent requests safe).

The key is a SHA-256 of the stay ID, so the table holds no identifier as sent.
``InMemoryAlertStore`` is for one process (laptop, tests). ``DynamoAlertStore``
uses the table the CloudFormation stack creates (hash key ``stayId``, TTL
``expiresAt``).
"""

import hashlib
import threading
import time

import numpy as np


class LedgerConflict(RuntimeError):
    """Another request changed this stay's ledger; read it again and retry."""


def stay_key(patient_id: str) -> str:
    return hashlib.sha256(patient_id.encode("utf-8")).hexdigest()


def ledger_events(hours, probabilities, threshold: float, cooldown_hours: float,
                  delivered_hours: list[float]) -> tuple[np.ndarray, list[int]]:
    """Alert flags per hour given the alerts already delivered.

    Returns (events, new_rows): ``events[i]`` is 1 where an alert was or is now
    raised; ``new_rows`` are the row indexes of alerts not yet in the ledger.
    """
    hours = np.asarray(hours, dtype=float)
    events = np.zeros(len(hours), dtype=int)
    delivered = sorted(float(h) for h in delivered_hours)
    last_alert = delivered[-1] if delivered else -np.inf
    recorded = set(delivered)
    new_rows = []
    for i, (hour, probability) in enumerate(zip(hours, probabilities, strict=True)):
        if hour <= last_alert and delivered:
            events[i] = int(hour in recorded)   # history is what was delivered, not a replay
        elif probability >= threshold and hour - last_alert >= cooldown_hours:
            events[i] = 1
            last_alert = hour
            new_rows.append(i)
    return events, new_rows


class InMemoryAlertStore:
    """Process-local ledger with the same optimistic-locking contract as DynamoDB."""

    def __init__(self):
        self._items: dict[str, tuple[int, list[dict]]] = {}
        self._lock = threading.Lock()

    def get(self, patient_id: str) -> tuple[list[dict], int]:
        with self._lock:
            version, alerts = self._items.get(stay_key(patient_id), (0, []))
            return [dict(a) for a in alerts], version

    def append(self, patient_id: str, alerts: list[dict], expected_version: int) -> None:
        with self._lock:
            key = stay_key(patient_id)
            version, existing = self._items.get(key, (0, []))
            if version != expected_version:
                raise LedgerConflict(key)
            self._items[key] = (version + 1, existing + [dict(a) for a in alerts])


class DynamoAlertStore:
    """One item per stay: ``{stayId, version, alerts: [{hour, prediction_id, model_version, at}], expiresAt}``."""

    def __init__(self, table_name: str, client=None, ttl_days: int = 30):
        if client is None:
            import boto3

            client = boto3.client("dynamodb")
        self.table_name, self.client, self.ttl_seconds = table_name, client, ttl_days * 86400

    @staticmethod
    def _decode(item: dict) -> list[dict]:
        return [{"hour": float(a["M"]["hour"]["N"]), "prediction_id": a["M"]["prediction_id"]["S"],
                 "model_version": a["M"]["model_version"]["S"], "at": a["M"]["at"]["S"]}
                for a in item.get("alerts", {}).get("L", [])]

    @staticmethod
    def _encode(alerts: list[dict]) -> dict:
        return {"L": [{"M": {"hour": {"N": repr(float(a["hour"]))}, "prediction_id": {"S": str(a["prediction_id"])},
                             "model_version": {"S": str(a.get("model_version") or "")}, "at": {"S": str(a["at"])}}}
                      for a in alerts]}

    def get(self, patient_id: str) -> tuple[list[dict], int]:
        response = self.client.get_item(TableName=self.table_name, Key={"stayId": {"S": stay_key(patient_id)}},
                                        ConsistentRead=True)
        item = response.get("Item")
        if not item:
            return [], 0
        return self._decode(item), int(item["version"]["N"])

    def append(self, patient_id: str, alerts: list[dict], expected_version: int) -> None:
        condition = "attribute_not_exists(stayId)" if expected_version == 0 else "version = :expected"
        values = {
            ":new": self._encode(alerts), ":empty": {"L": []}, ":next": {"N": str(expected_version + 1)},
            ":ttl": {"N": str(int(time.time()) + self.ttl_seconds)},
        }
        if expected_version:
            values[":expected"] = {"N": str(expected_version)}
        try:
            self.client.update_item(
                TableName=self.table_name, Key={"stayId": {"S": stay_key(patient_id)}},
                UpdateExpression="SET alerts = list_append(if_not_exists(alerts, :empty), :new), "
                                 "version = :next, expiresAt = :ttl",
                ConditionExpression=condition, ExpressionAttributeValues=values,
            )
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if code == "ConditionalCheckFailedException":
                raise LedgerConflict(stay_key(patient_id)) from error
            raise
