import pandas as pd
import json
import time
from kafka import KafkaProducer

def run_producer(file_path, topic_name='icu_vitals', delay=1.0):
    # Load the mixed_focus dataset
    df = pd.read_csv(file_path)

    # Initialize Kafka producer
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    print(f"Streaming {len(df)} ICU records to topic '{topic_name}'...")

    for _, row in df.iterrows():
        message = {
            "patient_id": int(row["patient_id"]),
            "Pulse": int(row.get("Pulse", 80)),
            "SysBP": int(row.get("SysBP", 110)),
            "Emergency": int(row.get("Emergency", 0)),
            "Infection": int(row.get("Infection", 0))  # Optional, add if present
        }

        # Send message to Kafka
        producer.send(topic_name, value=message)
        print(f"Sent: {message}")
        time.sleep(delay)

    print("Finished streaming.")

if __name__ == "__main__":
    run_producer("/Users/roshinijay/Anomaly-Detection-in-ICU/data/processed/mixed_focus.csv")

