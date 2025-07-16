
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kafka import KafkaConsumer
import json

# Local imports (adjusted to relative paths for local run)
from alert_prioritizer import prioritize_alerts
from recommender import load_feedback, compute_alert_bias, adjust_priority

def detect_anomalies(vitals):
    issues = []

    if vitals.get("SysBP") is not None:
        if vitals["SysBP"] < 90 or vitals["SysBP"] > 150:
            issues.append("Abnormal SysBP")

    if vitals.get("Pulse") is not None:
        if vitals["Pulse"] < 50 or vitals["Pulse"] > 120:
            issues.append("Abnormal Pulse")

    if vitals.get("Emergency") == 1:
        issues.append("Emergency Case")

    if issues:
        score = len(issues)  # Simple scoring: 1 point per issue

        return {
            "patient_id": vitals["patient_id"],
            "Emergency": vitals.get("Emergency", 0),
            "severity_score": score,
            "alert_type": ", ".join(issues)
        }

    return None

def run_consumer(topic_name='icu_vitals'):
    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers='localhost:9092',
        auto_offset_reset='latest',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    feedback = load_feedback()
    bias_map = compute_alert_bias(feedback)

    alert_buffer = []
    print(f"📡 Listening to Kafka topic: {topic_name}")

    for message in consumer:
        vitals = message.value
        print(f"Received: {vitals}")
        alert = detect_anomalies(vitals)

        if alert:
            print(f"[ALERT] Patient {alert['patient_id']} | Score: {alert['severity_score']} | Issues: {alert['alert_type']}")
            alert_buffer.append(alert)

        # Prioritize every 5 alerts
        if len(alert_buffer) >= 5:
            ranked = prioritize_alerts(alert_buffer)
            adjusted = [adjust_priority(a, bias_map) for a in ranked]

            print("\n📊 Prioritized Alerts:")
            for a in adjusted:
                print(f"Patient {a['patient_id']} | Score: {a['severity_score']} | Emergency: {a['Emergency']}")

            alert_buffer = []  # Reset for next batch

if __name__ == "__main__":
    run_consumer()
