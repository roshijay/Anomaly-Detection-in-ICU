# recommender.py

import json
from collections import defaultdict

def load_feedback(log_path="feedback_log.json"):
    try:
        with open(log_path, "r") as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        return []

def compute_alert_bias(feedback_entries):
    """
    Count how many times each alert type was rejected or deferred.
    Return a dict like:
    { 'Abnormal Pulse': {'Reject': 3, "️Defer': 2} }
    """
    alert_bias = defaultdict(lambda: {"Reject": 0, "Defer": 0})

    for entry in feedback_entries:
        action = entry["action"]
        notes = entry.get("notes", "")
        alert_type = entry.get("alert_type", "Unknown")

        if action in ["Reject", "Defer"]:
            alert_bias[alert_type][action] += 1

    return alert_bias

def adjust_priority(alert, alert_bias, threshold=3):
    """
    If an alert type has been frequently rejected or deferred,
    downgrade its severity_score or flag as 'downgraded'.
    """
    alert_type = alert.get("alert_type")
    score = alert.get("severity_score", 0)

    bias = alert_bias.get(alert_type, {})
    downgrade = bias.get("Reject", 0) + bias.get("️Defer", 0)

    if downgrade >= threshold:
        alert["severity_score"] = max(score - 2, 1)
        alert["recommendation"] = "Downgraded (based on past feedback)"
    else:
        alert["recommendation"] = "Normal"

    return alert
