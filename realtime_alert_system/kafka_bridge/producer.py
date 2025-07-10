import pandas as pd 
import math 
import json
import time 
from kafka import KafkaProducer

def run_producer(file_path, topic_name='icu_vitals', delay=1.0):
    # Load data
    df = pd.read_csv(file_path)
    
    # Initialize Kafka Producer
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
        
    print(f"Streaming {len(df)} ICU records to topic '{topic_name}'...")
    
    for _, row in df.iterrows():
        message = {
            "patient_id": int(row["patient_id"]),
            "SysBP": int(row.get("saps-i", 120)),     # Use 'saps-i' as proxy
            "Pulse": int(row["HR"]) if not math.isnan(row.get("HR", float("nan"))) else 80,                                     # HR = heart rate
            "Infection": 0,                           # Not available in Kaggle
            "Emergency": 1 if int(row.get("mortality", 0)) == 1 else 0
        }

        # Send message to Kafka
        producer.send(topic_name, value=message)
        print(f"Sent: {message}")
        time.sleep(delay)
        
    print("Finished streaming all records.")
     
if __name__ == "__main__":
    run_producer("data/processed/processed_kaggle_icu.csv")


