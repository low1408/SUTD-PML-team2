"""Write a standalone, offline patient replay page.

``train`` writes it for validation stays; ``evaluate`` writes it for the test
splits of a frozen model. ``summary["splits"]`` holds the metrics per split shown.
"""

import json
from pathlib import Path


def write_report(path, table, summary, threshold, synthetic):
    # Labels appear as retrospective annotations only.
    patients = []
    for patient_id, rows in table.groupby("patient_id", sort=True):
        rows = rows.sort_values("hour")
        patients.append({
            "id": str(patient_id), "split": str(rows["split"].iloc[0]),
            "hour": rows["hour"].tolist(), "probability": rows["probability"].tolist(),
            "label": rows["label"].tolist(), "alert": rows["alert_event"].tolist(),
        })
    payload = json.dumps({
        "patients": patients, "metrics": summary, "threshold": threshold,
        "synthetic": synthetic,
    }, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("report_template.html").read_text()
    Path(path).write_text(template.replace("__PAYLOAD__", payload))
