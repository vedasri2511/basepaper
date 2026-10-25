import sys
import os
import unittest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.holistic_rca import HolisticRCA
from src.models.mask_learning import MaskLearner


class TestMask(unittest.TestCase):

    def test_mask_learning(self):
        batch_size = 1
        num_time_buckets = 5
        num_nodes = 7 # 2 service, 4 pod, 1 node

        # 10 traces, 8 metrics, 15 log features total = 33 total max
        feature_counts_obs = {'trace': 10, 'metric': 8, 'log': 15}
        total_features = 33

        feature_counts = {
            'service': 6,
            'pod': 4,
            'node': 3
        }

        # Create the model
        model = HolisticRCA(
            node_types=['service', 'pod', 'node'],
            feature_counts=feature_counts,
            num_time_buckets=num_time_buckets,
            embed_dim=32,
            entity_out_dim=64,
            fault_classes={'service': ['f1', 'f2'], 'pod': ['f1'], 'node': ['f1']}
        )

        # Create the Mask Learner
        mask_learner = MaskLearner(model, num_nodes=num_nodes, feature_counts=feature_counts_obs)

        # Fake inputs
        traces = torch.rand(batch_size, num_time_buckets, 10)
        metrics = torch.rand(batch_size, num_time_buckets, 8)
        logs = torch.rand(batch_size, num_time_buckets, 15)

        entity_mappings = {
            'service': torch.randint(0, 33, (2, 6)),
            'pod': torch.randint(0, 33, (4, 4)),
            'node': torch.randint(0, 33, (1, 3)),
        }

        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 0, 1]
        ], dtype=torch.long)

        # Get original predictions
        original_preds = model(traces, metrics, logs, entity_mappings, edge_index)

        # Run through mask learner
        masked_preds, edge_mask, feature_mask = mask_learner(traces, metrics, logs, entity_mappings, edge_index)

        # Calc loss
        loss, loss_dict = mask_learner.loss_function(masked_preds, original_preds, edge_mask, feature_mask)

        self.assertGreaterEqual(loss.item(), 0)
        self.assertEqual(edge_mask.shape[0], edge_index.shape[1])


if __name__ == '__main__':
    unittest.main()
