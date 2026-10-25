import pandas as pd
import numpy as np
import os
import random

def generate_social_network_dataset(output_dir="data_samples/dataset_c"):
    """
    Generates a highly realistic synthetic version of Dataset C (SocialNetwork)
    with 14 business services, 7 metrics, traces, and logs.
    Includes a targeted fault (CPU Spike) at a specific timestamp.
    """
    os.makedirs(output_dir, exist_ok=True)

    services = [f"social_service_{i}" for i in range(14)]
    metric_types = ["cpu_usage", "memory_usage", "network_tx", "network_rx", "disk_read", "disk_write", "latency"]

    # Time window: 1 hour (60 minutes), 1 data point per minute
    timestamps = list(range(1696000000, 1696000000 + 3600, 60))

    # 1. Generate Metrics
    metrics_data = []
    fault_timestamp = 1696001800 # Fault occurs 30 mins in
    fault_service = "social_service_3"

    for t in timestamps:
        for srv in services:
            for m_type in metric_types:
                base_value = random.uniform(10, 30)

                # INJECT FAULT: CPU Spike in social_service_3 at fault_timestamp
                if srv == fault_service and m_type == "cpu_usage" and fault_timestamp <= t < fault_timestamp + 600:
                    base_value = random.uniform(85, 99) # Spike!

                metrics_data.append({
                    "timestamp": t,
                    "entity": srv,
                    "metric_type": m_type,
                    "value": base_value
                })

    pd.DataFrame(metrics_data).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    # 2. Generate Traces
    traces_data = []
    span_id_counter = 1
    for t in timestamps:
        # Simulate 5 traces per minute
        for _ in range(5):
            parent_id = None
            for idx, srv in enumerate(services[:5]): # simple chain
                duration = random.uniform(5, 15)
                # INJECT FAULT: Latency cascades down the chain from service 3
                if srv == fault_service and fault_timestamp <= t < fault_timestamp + 600:
                    duration = random.uniform(100, 500) # High latency

                traces_data.append({
                    "span_id": f"span_{span_id_counter}",
                    "parent_span_id": parent_id,
                    "service_name": srv,
                    "start_time": t + random.randint(0, 50),
                    "duration": duration
                })
                parent_id = f"span_{span_id_counter}"
                span_id_counter += 1

    pd.DataFrame(traces_data).to_csv(os.path.join(output_dir, "traces.csv"), index=False)

    # 3. Generate Logs
    logs_data = []
    for t in timestamps:
        for srv in services:
            message = "INFO: Request processed successfully"
            if srv == fault_service and fault_timestamp <= t < fault_timestamp + 600:
                if random.random() > 0.5:
                    message = "ERROR: Connection timeout, CPU thread starvation"

            logs_data.append({
                "timestamp": t,
                "entity": srv,
                "message": message
            })

    pd.DataFrame(logs_data).to_csv(os.path.join(output_dir, "logs.csv"), index=False)

    # 4. Generate Ground-Truth Labels
    generate_labels(output_dir=output_dir)

    print(f"Realistic Synthetic Dataset C successfully generated at {output_dir}")
    print(f"INJECTED FAULT: CPU Spike on {fault_service} starting at timestamp {fault_timestamp}")

def generate_labels(output_dir="data_samples/dataset_c"):
    """
    Generates ground-truth labels for Dataset C (SocialNetwork) aligned with 60-second buckets.

    Schema:
      - timestamp: Bucket start time (epoch seconds)
      - timestamp_end: Bucket end time (epoch seconds, timestamp + 60)
      - entity: Service entity identifier (social_service_0 .. social_service_13)
      - is_fault: Binary label (0 = normal, 1 = faulty)
      - fault_type: Categorical fault type ('normal' or 'cpu_load')
    """
    os.makedirs(output_dir, exist_ok=True)
    services = [f"social_service_{i}" for i in range(14)]
    timestamps = list(range(1696000000, 1696000000 + 3600, 60))
    fault_timestamp = 1696001800
    fault_service = "social_service_3"
    fault_duration = 600

    labels_data = []
    for t in timestamps:
        for srv in services:
            is_fault = 1 if (srv == fault_service and fault_timestamp <= t < fault_timestamp + fault_duration) else 0
            fault_type = "cpu_load" if is_fault else "normal"
            labels_data.append({
                "timestamp": t,
                "timestamp_end": t + 60,
                "entity": srv,
                "is_fault": is_fault,
                "fault_type": fault_type
            })

    labels_df = pd.DataFrame(labels_data)
    labels_path = os.path.join(output_dir, "labels.csv")
    labels_df.to_csv(labels_path, index=False)
    print(f"Ground-truth labels successfully generated at {labels_path}")
    return labels_df

if __name__ == "__main__":
    generate_labels()
