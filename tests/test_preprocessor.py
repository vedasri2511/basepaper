import sys
import os
import unittest
import pandas as pd
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.preprocessor import Preprocessor, PreprocessedOutput


class TestPreprocessor(unittest.TestCase):

    def test_trace_extraction(self):
        # Dummy data
        spans = pd.DataFrame([
            {'span_id': 's1', 'parent_span_id': None, 'service_name': 'serviceA', 'start_time': 100, 'duration': 10},
            {'span_id': 's2', 'parent_span_id': 's1', 'service_name': 'serviceB', 'start_time': 105, 'duration': 5},
            {'span_id': 's3', 'parent_span_id': 's2', 'service_name': 'serviceC', 'start_time': 107, 'duration': 2}
        ])

        preprocessor = Preprocessor(time_interval=(100, 110), time_granularity=5)
        services = ['serviceA', 'serviceB', 'serviceC']
        features = preprocessor.extract_trace_features(spans, services)

        self.assertEqual(features.shape, (2, 3 * 6))
        self.assertFalse(np.isnan(features).any(), "NaN found in trace features")
        self.assertFalse(np.isinf(features).any(), "Inf found in trace features")

    def test_metric_extraction(self):
        metrics = pd.DataFrame([
            {'timestamp': 101, 'entity': 'serviceA', 'metric_type': 'cpu', 'value': 0.5},
            {'timestamp': 106, 'entity': 'serviceA', 'metric_type': 'cpu', 'value': 0.8},
            {'timestamp': 102, 'entity': 'serviceB', 'metric_type': 'mem', 'value': 200},
        ])
        preprocessor = Preprocessor(time_interval=(100, 110), time_granularity=5)
        entities = ['serviceA', 'serviceB']
        metric_types = ['cpu', 'mem']

        # 3D unflattened
        features_3d = preprocessor.extract_metric_features(metrics, entities, metric_types, flatten=False)
        self.assertEqual(features_3d.shape, (2, 2, 2))

        # 2D flattened
        features_2d = preprocessor.extract_metric_features(metrics, entities, metric_types, flatten=True)
        self.assertEqual(features_2d.shape, (2, 4))
        self.assertFalse(np.isnan(features_2d).any(), "NaN found in metric features")

    def test_log_extraction(self):
        logs = pd.DataFrame([
            {'timestamp': 101, 'entity': 'serviceA', 'message': 'Received request for /api/v1/users'},
            {'timestamp': 106, 'entity': 'serviceB', 'message': 'Connection timeout while connecting to db'}
        ])
        preprocessor = Preprocessor(time_interval=(100, 110), time_granularity=5)
        entities = ['serviceA', 'serviceB']

        features, vocab = preprocessor.extract_log_features(logs, entities)
        self.assertGreater(len(vocab), 0, "Log vocabulary should not be empty")
        self.assertEqual(features.shape, (2, len(entities) * len(vocab)))
        self.assertFalse(np.isnan(features).any(), "NaN found in log features")

    def test_process_all_dataset_c(self):
        """Validates full end-to-end preprocessing on Dataset C sample data."""
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_samples', 'dataset_c'))
        traces = pd.read_csv(os.path.join(data_dir, 'traces.csv'))
        metrics = pd.read_csv(os.path.join(data_dir, 'metrics.csv'))
        logs = pd.read_csv(os.path.join(data_dir, 'logs.csv'))
        labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))

        preprocessor = Preprocessor(
            time_interval=(1696000000, 1696003600),
            time_granularity=60,
            top_k_log=5,
            tf_idf_threshold=0.3,
            word_threshold=0.04
        )

        raw_data = {'traces': traces, 'metrics': metrics, 'logs': logs}
        output = preprocessor.process_all(raw_data)

        # 1. Output type
        self.assertIsInstance(output, PreprocessedOutput)

        # 2. Bucket count
        num_buckets = 60
        self.assertEqual(output.X_trace.shape[0], num_buckets)
        self.assertEqual(output.X_metric.shape[0], num_buckets)
        self.assertEqual(output.X_log.shape[0], num_buckets)
        self.assertEqual(len(output.timestamps), num_buckets)

        # 3. Timestamp alignment with labels.csv
        label_ts = sorted(labels['timestamp'].unique())
        self.assertEqual(output.timestamps, label_ts, "Timestamps do not align with labels.csv")

        # 4. Entity count & ordering
        self.assertEqual(len(output.entities), 14)
        self.assertEqual(output.entities[0], 'social_service_0')
        self.assertEqual(output.entities[-1], 'social_service_13')

        # 5. Dimensionalities
        n_trace = 14 * 6 # 84
        n_metric = 14 * 7 # 98
        k_log = len(output.log_vocab)
        n_log = 14 * k_log

        self.assertEqual(output.X_trace.shape, (num_buckets, n_trace))
        self.assertEqual(output.X_metric.shape, (num_buckets, n_metric))
        self.assertEqual(output.X_log.shape, (num_buckets, n_log))

        # 6. Entity Feature Map integrity
        total_features = n_trace + n_metric + n_log
        for ent in output.entities:
            m = output.entity_feature_map[ent]
            self.assertEqual(len(m['trace_indices']), 6)
            self.assertEqual(len(m['metric_indices']), 7)
            self.assertEqual(len(m['log_indices']), k_log)
            self.assertEqual(len(m['global_indices']), 6 + 7 + k_log)
            self.assertLess(max(m['global_indices']), total_features)

        # 7. Finite numerical integrity
        for name, arr in [('X_trace', output.X_trace), ('X_metric', output.X_metric), ('X_log', output.X_log)]:
            self.assertFalse(np.isnan(arr).any(), f"NaN detected in {name}")
            self.assertFalse(np.isinf(arr).any(), f"Inf detected in {name}")
            self.assertIn(arr.dtype, [np.float32, np.float64])

    def test_missing_and_edge_cases(self):
        """Validates resilience against empty dataframes and out-of-range timestamps."""
        preprocessor = Preprocessor(time_interval=(1000, 1300), time_granularity=60)

        # All empty
        empty_raw = {'traces': pd.DataFrame(), 'metrics': pd.DataFrame(), 'logs': pd.DataFrame()}
        out = preprocessor.process_all(empty_raw, entities=['srv_a', 'srv_b'])

        self.assertEqual(out.X_trace.shape, (5, 2 * 6))
        self.assertEqual(out.X_metric.shape, (5, 0))
        self.assertFalse(np.isnan(out.X_trace).any())
        self.assertTrue((out.X_trace == 0.0).all())

        # Out of range timestamps
        out_of_range_metrics = pd.DataFrame([
            {'timestamp': 50, 'entity': 'srv_a', 'metric_type': 'cpu', 'value': 99.0},
            {'timestamp': 5000, 'entity': 'srv_a', 'metric_type': 'cpu', 'value': 99.0}
        ])
        feat = preprocessor.extract_metric_features(out_of_range_metrics, ['srv_a'], ['cpu'], flatten=True)
        self.assertTrue((feat == 0.0).all(), "Out-of-range records should not populate buckets")

    def test_deterministic_vocabulary(self):
        """Validates that supplying a fixed vocabulary ensures identical feature columns."""
        preprocessor = Preprocessor(time_interval=(100, 200), time_granularity=50)
        logs = pd.DataFrame([
            {'timestamp': 110, 'entity': 'srv_1', 'message': 'service starting up'},
            {'timestamp': 160, 'entity': 'srv_1', 'message': 'database connection failed'}
        ])
        fixed_vocab = ['tmpl:database connection failed', 'tmpl:service starting up', 'word:connection', 'word:failed']

        f1, v1 = preprocessor.extract_log_features(logs, ['srv_1'], vocab=fixed_vocab)
        f2, v2 = preprocessor.extract_log_features(logs, ['srv_1'], vocab=fixed_vocab)

        self.assertEqual(v1, fixed_vocab)
        self.assertEqual(v2, fixed_vocab)
        self.assertTrue(np.array_equal(f1, f2))


if __name__ == '__main__':
    unittest.main()
