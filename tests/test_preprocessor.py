import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
import numpy as np
from data.preprocessor import Preprocessor

def test_trace_extraction():
    # Dummy data
    spans = pd.DataFrame([
        {'span_id': 's1', 'parent_span_id': None, 'service_name': 'serviceA', 'start_time': 100, 'duration': 10},
        {'span_id': 's2', 'parent_span_id': 's1', 'service_name': 'serviceB', 'start_time': 105, 'duration': 5},
        {'span_id': 's3', 'parent_span_id': 's2', 'service_name': 'serviceC', 'start_time': 107, 'duration': 2}
    ])
    
    preprocessor = Preprocessor(time_interval=(100, 110), time_granularity=5)
    services = ['serviceA', 'serviceB', 'serviceC']
    features = preprocessor.extract_trace_features(spans, services)
    
    print("Trace Features Shape:", features.shape)
    assert features.shape == (2, 3 * 6), "Incorrect shape for trace features"
    print("Trace features extracted successfully.")

def test_metric_extraction():
    metrics = pd.DataFrame([
        {'timestamp': 101, 'entity': 'serviceA', 'metric_type': 'cpu', 'value': 0.5},
        {'timestamp': 106, 'entity': 'serviceA', 'metric_type': 'cpu', 'value': 0.8},
        {'timestamp': 102, 'entity': 'serviceB', 'metric_type': 'mem', 'value': 200},
    ])
    preprocessor = Preprocessor(time_interval=(100, 110), time_granularity=5)
    entities = ['serviceA', 'serviceB']
    metric_types = ['cpu', 'mem']
    
    features = preprocessor.extract_metric_features(metrics, entities, metric_types)
    print("Metric Features Shape:", features.shape)
    assert features.shape == (2, 2, 2), "Incorrect shape for metric features"
    print("Metric features extracted successfully.")
    
def test_log_extraction():
    logs = pd.DataFrame([
        {'timestamp': 101, 'entity': 'serviceA', 'message': 'Received request for /api/v1/users'},
        {'timestamp': 106, 'entity': 'serviceB', 'message': 'Connection timeout while connecting to db'}
    ])
    preprocessor = Preprocessor(time_interval=(100, 110), time_granularity=5)
    entities = ['serviceA', 'serviceB']
    
    # Just running to ensure it doesn't crash since it's a skeleton for TF-IDF right now
    preprocessor.extract_log_features(logs, entities)
    print("Log features processing completed without errors.")

if __name__ == '__main__':
    test_trace_extraction()
    test_metric_extraction()
    test_log_extraction()
    print("All preprocessing tests passed!")
