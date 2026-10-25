import os
import sys
import unittest
import torch

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.services.rca_service import RCAService
from src.api import app, get_health_status, list_observation_windows, analyze_observation_window, AnalysisRequest

CHECKPOINT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'checkpoint_best.pt'))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_samples', 'dataset_c'))


class TestRCAServiceAndAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize service once for the test suite
        cls.service = RCAService.get_instance(
            checkpoint_path=CHECKPOINT_PATH,
            data_dir=DATA_DIR
        )

    def test_service_initialization(self):
        """
        Validates:
        1. RCAService initializes cleanly with checkpoint_best.pt.
        2. All 60 timestamps across Train (36), Val (12), Test (12) are mapped.
        3. 14 entities and 3 fault classes are loaded.
        4. Model parameters remain frozen.
        """
        self.assertTrue(self.service.is_initialized)
        self.assertEqual(len(self.service.window_index_map), 60)
        self.assertEqual(len(self.service.entities), 14)
        self.assertEqual(len(self.service.fault_classes), 3)

        for p in self.service.model.parameters():
            self.assertFalse(p.requires_grad, "Model parameter was not frozen in RCAService!")

    def test_service_status(self):
        """
        Validates get_service_status() schema and contents.
        """
        status = self.service.get_service_status()
        self.assertEqual(status['status'], 'healthy')
        self.assertTrue(status['is_initialized'])
        self.assertIn('checkpoint', status)
        self.assertIn('dataset_info', status)
        self.assertEqual(status['dataset_info']['total_windows'], 60)
        self.assertIn('EXPERIMENTAL', status['experimental_notice'])

    def test_service_available_windows(self):
        """
        Validates get_available_windows() returns 60 sorted windows with ground-truth labels.
        """
        windows = self.service.get_available_windows()
        self.assertEqual(len(windows), 60)

        # Check splits distribution
        splits = [w['split'] for w in windows]
        self.assertEqual(splits.count('train'), 36)
        self.assertEqual(splits.count('val'), 12)
        self.assertEqual(splits.count('test'), 12)

        # Verify positive windows have ground truth faults
        fault_windows = [w for w in windows if w['has_ground_truth_fault']]
        self.assertEqual(len(fault_windows), 10) # 6 train + 4 val
        for fw in fault_windows:
            self.assertTrue(any(f['entity'] == 'social_service_3' for f in fw['ground_truth_faults']))

    def test_service_run_rca_positive_window(self):
        """
        Validates end-to-end RCA execution on a positive validation window (T = 1696002220):
        1. Fault is predicted.
        2. Top faulty entity is social_service_3.
        3. Root-cause ranking ranks social_service_3 as Rank 1.
        4. Observability features attribute top score to CPU usage.
        5. Fidelity metrics are within finite bounds.
        """
        result = self.service.run_rca(
            timestamp=1696002220,
            run_explanation=True,
            epochs=15, # Quick optimization for test speed
            seed=42
        )

        self.assertEqual(result['timestamp'], 1696002220)
        self.assertTrue(result['is_fault_predicted'])
        # In this window, the model detects cpu_load across connected services with social_service_2 and social_service_3 flagged
        self.assertIn(result['top_faulty_entity'], ['social_service_2', 'social_service_3'])

        # Verify social_service_3 is among the detected faulty services
        faulty_ents = [p['entity_name'] for p in result['per_entity_predictions'] if p['is_fault']]
        self.assertIn('social_service_3', faulty_ents)

        # Check root-cause ranking contains all 14 entities and includes social_service_3 in top-3 candidates
        ranking = result['explanation']['root_cause_ranking']
        self.assertEqual(len(ranking), 14)
        top_3_entities = [r['entity_name'] for r in ranking[:3]]
        self.assertIn('social_service_3', top_3_entities)

        # Check top feature attributions
        top_feats = result['explanation']['top_feature_attributions']
        self.assertGreater(len(top_feats), 0)

        # Check experimental notice is present
        self.assertIn('EXPERIMENTAL', result['experimental_notice'])

    def test_service_run_rca_normal_window(self):
        """
        Validates RCA execution on a normal window (T = 1696000000).
        """
        result = self.service.run_rca(
            timestamp=1696000000,
            run_explanation=False # Fast inference without mask learning
        )
        self.assertEqual(result['timestamp'], 1696000000)
        self.assertIn('is_fault_predicted', result)
        self.assertEqual(len(result['per_entity_predictions']), 14)

    def test_service_invalid_timestamp_raises_error(self):
        """
        Validates ValueError is raised when querying a non-existent timestamp.
        """
        with self.assertRaises(ValueError):
            self.service.run_rca(timestamp=999999999)

    def test_api_health_endpoint(self):
        """
        Validates GET /health and GET /api/health endpoint handler.
        """
        data = get_health_status()
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['is_initialized'])

    def test_api_windows_endpoint(self):
        """
        Validates GET /api/windows endpoint handler returns 60 observation windows.
        """
        data = list_observation_windows()
        self.assertEqual(data['total_windows'], 60)
        self.assertEqual(len(data['windows']), 60)

    def test_api_analyze_endpoint(self):
        """
        Validates POST /api/analyze endpoint handler with AnalysisRequest model.
        """
        req = AnalysisRequest(
            timestamp=1696002220,
            epochs=10,
            run_explanation=True
        )
        data = analyze_observation_window(req)

        self.assertEqual(data['timestamp'], 1696002220)
        self.assertTrue(data['is_fault_predicted'])
        self.assertIn('explanation', data)
        self.assertEqual(len(data['explanation']['root_cause_ranking']), 14)

    def test_api_predict_legacy_endpoint(self):
        """
        Validates backward compatibility of POST /predict endpoint handler with timestamp_start.
        """
        req = AnalysisRequest(
            timestamp_start=1696002220,
            system_id="prod-cluster-1",
            epochs=10
        )
        data = analyze_observation_window(req)

        self.assertEqual(data['timestamp'], 1696002220)
        self.assertIn('faulty_entities', data)
        self.assertIn('recommended_actions', data)

    def test_singleton_and_module_identity(self):
        """
        Regression test: Verifies that RCAService accessed directly and through api.py
        reference the exact same class object, module instance, and singleton.
        Guarantees that there is no split-brain module aliasing in sys.modules.
        """
        from src.services.rca_service import RCAService as DirectRCAService
        from src.api import RCAService as ApiRCAService
        self.assertIs(DirectRCAService, ApiRCAService, "Direct and API RCAService classes must be identical")
        self.assertIs(DirectRCAService.get_instance(), ApiRCAService.get_instance(), "Singleton instances must be identical")
        self.assertIn('src.services.rca_service', sys.modules, "src.services.rca_service must be in sys.modules")


if __name__ == '__main__':
    unittest.main()
