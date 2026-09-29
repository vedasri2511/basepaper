import pandas as pd
import os

class DatasetALoader:
    """
    Loader for Dataset A (Hipstershop large-scale AIOps challenge data).
    According to the paper (Section V.B.1):
    - 56 resource entities total: 10 services, 40 pods, 6 nodes.
    - 7 days of training data, 300 fault injection labels.
    - Metrics for services, pods, and nodes.
    - Traces and logs associated primarily with pods.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        self.services = [f"service_{i}" for i in range(10)]
        self.pods = [f"pod_{i}" for i in range(40)]
        self.nodes = [f"node_{i}" for i in range(6)]
        
        self.entities = self.services + self.pods + self.nodes

    def load_metrics(self):
        """Loads minute-level metrics for all 56 entities."""
        metrics_path = os.path.join(self.data_dir, "metrics.csv")
        if os.path.exists(metrics_path):
            return pd.read_csv(metrics_path)
        else:
            print(f"Warning: {metrics_path} not found. Using placeholder.")
            # Return placeholder for pipeline execution
            return pd.DataFrame(columns=['timestamp', 'entity', 'metric_type', 'value'])

    def load_traces(self):
        """Loads traces (spans) for service invocations."""
        traces_path = os.path.join(self.data_dir, "traces.csv")
        if os.path.exists(traces_path):
            return pd.read_csv(traces_path)
        else:
            print(f"Warning: {traces_path} not found. Using placeholder.")
            return pd.DataFrame(columns=['span_id', 'parent_span_id', 'service_name', 'start_time', 'duration'])

    def load_logs(self):
        """Loads unstructured logs."""
        logs_path = os.path.join(self.data_dir, "logs.csv")
        if os.path.exists(logs_path):
            return pd.read_csv(logs_path)
        else:
            print(f"Warning: {logs_path} not found. Using placeholder.")
            return pd.DataFrame(columns=['timestamp', 'entity', 'message'])

    def get_topology_edges(self):
        """
        Constructs the RERG (Resource Entity Relation Graph) edges for Dataset A.
        Connects Pods to their host Nodes, Services to their backing Pods, and inter-service call graphs.
        """
        # Placeholder static topology mapping for Dataset A structure
        # (2 edges added per relation as per Section IV.D: "with one entity pointing to the other and its reverse edge")
        edges = [[], []]
        
        # Example mapping: 4 pods per service
        for i, srv in enumerate(self.services):
            for j in range(4):
                pod_idx = i * 4 + j
                srv_node_idx = i
                pod_node_idx = 10 + pod_idx
                
                edges[0].extend([srv_node_idx, pod_node_idx])
                edges[1].extend([pod_node_idx, srv_node_idx])
                
        # Example mapping: Pods distributed across 6 nodes
        for pod_idx in range(40):
            node_idx = pod_idx % 6
            pod_node_idx = 10 + pod_idx
            host_node_idx = 50 + node_idx
            
            edges[0].extend([pod_node_idx, host_node_idx])
            edges[1].extend([host_node_idx, pod_node_idx])
            
        return edges
