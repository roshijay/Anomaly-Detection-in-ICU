from kafka import KafkaConsumer
import json
from realtime_alert_system.alert_prioritizer import prioritize_alerts
from realtime_alert_system.recommender import load_feedback, compute_alert_bias, adjust_priority

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
        score = len(issues) # Simple scoring: 1 point per issue 

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
    print(f"Listening to Kafka topic: {topic_name}")

    for message in consumer:
        vitals = message.value
        alert = detect_anomalies(vitals)

        if alert:
            alert_buffer.append(alert)

        if len(alert_buffer) >= 5:
            ranked = prioritize_alerts(alert_buffer)
            adjusted = [adjust_priority(a, bias_map) for a in ranked]

            print("\nPrioritized Alerts:")
            for a in adjusted:
                print(f"Patient {a['patient_id']} | Score: {a['severity_score']} | Emergency: {a['Emergency']}")

            alert_buffer = []

if __name__ == "__main__":
    run_consumer()

