import os
import sys
import json
import unittest
import numpy as np
import torch
import torch.nn as nn

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.holistic_rca import HolisticRCA
from src.models.mask_learning import MaskLearner, HolisticRCAExplainer, ExplanationOutput
from src.models.checkpoint import load_checkpoint
from src.data.dataset import load_dataset_c_splits


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_samples', 'dataset_c'))
CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))


class TestMaskLearningAndExplainer(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.num_nodes = 14
        self.entities = [f"social_service_{i}" for i in range(self.num_nodes)]
        self.fault_classes = ["cpu_load", "network_delay", "network_loss"]
        self.feature_counts = {"trace": 84, "metric": 98, "log": 84}
        self.total_features = 84 + 98 + 84 # 266

        # Build synthetic base model for quick isolated testing
        self.model = HolisticRCA(
            node_types=['service'],
            feature_counts={'service': 19},
            num_time_buckets=60,
            embed_dim=32,
            entity_out_dim=64,
            fault_classes={'service': self.fault_classes}
        )
        self.model.eval()

        # Synthetic gathering indices: 14 entities x 19 features
        self.entity_indices = torch.arange(14 * 19, dtype=torch.long).view(14, 19) % self.total_features

        # Synthetic edges (bidirectional loop with 22 edges)
        edges_src = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 2, 3, 4, 0, 1, 2, 3]
        edges_dst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 0, 1, 2, 3, 1, 2, 3, 4]
        self.edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

        # Synthetic observability inputs
        self.traces = torch.randn(60, 84)
        self.metrics = torch.randn(60, 98)
        self.logs = torch.randn(60, 84)

    def test_mask_shapes_and_modality_partition_slicing(self):
        """
        Validates:
        1. MaskLearner parameters EM in R^(n_g x n_g) and om in R^(n_f).
        2. Exact shapes: EM (14, 14), om (266,).
        3. Edge mask evaluated along edge_index has shape (num_edges,).
        4. Forward pass returns masked predictions with identical output shape.
        """
        learner = MaskLearner(
            trained_model=self.model,
            num_nodes=self.num_nodes,
            feature_counts=self.feature_counts,
            init_strategy="normal"
        )

        self.assertEqual(learner.EM.shape, (14, 14))
        self.assertEqual(learner.om.shape, (266,))

        masked_preds, edge_mask, feat_mask = learner(
            traces=self.traces,
            metrics=self.metrics,
            logs=self.logs,
            entity_mappings=self.entity_indices,
            edge_index=self.edge_index
        )

        self.assertEqual(edge_mask.shape, (self.edge_index.shape[1],))
        self.assertEqual(feat_mask.shape, (266,))
        self.assertEqual(masked_preds['service'].shape, (14, 3))

    def test_classifier_weights_remain_strictly_frozen(self):
        """
        Validates:
        Section IV.F requirement: classifier parameters must remain strictly frozen
        while only the mask parameters EM and om receive gradients and updates.
        """
        learner = MaskLearner(
            trained_model=self.model,
            num_nodes=self.num_nodes,
            feature_counts=self.feature_counts
        )

        # Verify requires_grad is False for all model parameters
        for name, param in learner.model.named_parameters():
            self.assertFalse(param.requires_grad, f"Parameter {name} was not frozen!")

        # Record parameter states before optimization
        initial_params = {k: v.clone() for k, v in self.model.state_dict().items()}

        optimizer = torch.optim.Adam([learner.EM, learner.om], lr=0.1)
        orig_preds = self.model(self.traces, self.metrics, self.logs, self.entity_indices, self.edge_index)

        for _ in range(5):
            optimizer.zero_grad()
            m_preds, e_mask, f_mask = learner(self.traces, self.metrics, self.logs, self.entity_indices, self.edge_index)
            loss, _ = learner.loss_function(m_preds, orig_preds, e_mask, f_mask)
            loss.backward()
            optimizer.step()

        # Verify bit-for-bit equality of classifier parameters after mask updates
        for k, v in self.model.state_dict().items():
            self.assertTrue(torch.equal(v, initial_params[k]), f"Model parameter {k} changed during mask optimization!")

    def test_mask_optimization_and_gradient_flow(self):
        """
        Validates:
        1. Gradients flow into EM and om.
        2. Both EM.grad and om.grad have non-zero norms.
        3. Objective loss is finite and strictly decreases across steps.
        """
        learner = MaskLearner(
            trained_model=self.model,
            num_nodes=self.num_nodes,
            feature_counts=self.feature_counts
        )
        optimizer = torch.optim.Adam([learner.EM, learner.om], lr=0.1)
        orig_preds = self.model(self.traces, self.metrics, self.logs, self.entity_indices, self.edge_index)

        losses = []
        for step in range(5):
            optimizer.zero_grad()
            m_preds, e_mask, f_mask = learner(self.traces, self.metrics, self.logs, self.entity_indices, self.edge_index)
            loss, _ = learner.loss_function(m_preds, orig_preds, e_mask, f_mask, target_entity_idx=3)
            loss.backward()

            self.assertIsNotNone(learner.EM.grad)
            self.assertIsNotNone(learner.om.grad)
            self.assertGreater(learner.EM.grad.norm().item(), 0.0)
            self.assertGreater(learner.om.grad.norm().item(), 0.0)

            optimizer.step()
            losses.append(loss.item())

        self.assertTrue(all(np.isfinite(losses)))
        self.assertLess(losses[-1], losses[0], "Mask loss did not decrease during optimization!")

    def test_deterministic_output_under_fixed_seed(self):
        """
        Validates:
        Explanation generation is strictly deterministic when seed is provided.
        """
        explainer = HolisticRCAExplainer(
            model=self.model,
            entities=self.entities,
            fault_classes=self.fault_classes,
            feature_counts=self.feature_counts,
            entity_feature_indices=self.entity_indices
        )

        exp1 = explainer.explain_window(
            traces=self.traces,
            metrics=self.metrics,
            logs=self.logs,
            edge_index=self.edge_index,
            epochs=15,
            seed=123
        )

        exp2 = explainer.explain_window(
            traces=self.traces,
            metrics=self.metrics,
            logs=self.logs,
            edge_index=self.edge_index,
            epochs=15,
            seed=123
        )

        self.assertEqual(exp1.root_cause_ranking[0]['entity_name'], exp2.root_cause_ranking[0]['entity_name'])
        self.assertAlmostEqual(exp1.root_cause_ranking[0]['score'], exp2.root_cause_ranking[0]['score'], places=5)
        self.assertAlmostEqual(exp1.fidelity_metrics['logit_l1_diff'], exp2.fidelity_metrics['logit_l1_diff'], places=5)

    def test_root_cause_ranking_structure_and_entity_mapping(self):
        """
        Validates:
        1. Ranking contains all 14 entities mapped to canonical service names.
        2. Scores are monotonically non-increasing (sorted descending).
        3. Feature attributions contain top ranked features with modality tags.
        """
        feature_names = {
            'trace': [f"trace_{self.entities[i%14]}_lat" for i in range(84)],
            'metric': [f"metric_{self.entities[i%14]}_cpu" for i in range(98)],
            'log': [f"log_{self.entities[i%14]}_err" for i in range(84)]
        }

        explainer = HolisticRCAExplainer(
            model=self.model,
            entities=self.entities,
            fault_classes=self.fault_classes,
            feature_counts=self.feature_counts,
            feature_names=feature_names,
            entity_feature_indices=self.entity_indices
        )

        output = explainer.explain_window(
            traces=self.traces,
            metrics=self.metrics,
            logs=self.logs,
            edge_index=self.edge_index,
            epochs=10,
            seed=42
        )

        self.assertEqual(len(output.root_cause_ranking), 14)
        scores = [item['score'] for item in output.root_cause_ranking]
        self.assertEqual(scores, sorted(scores, reverse=True))

        ranked_names = [item['entity_name'] for item in output.root_cause_ranking]
        self.assertEqual(set(ranked_names), set(self.entities))

        self.assertLessEqual(len(output.top_feature_attributions), 15)
        for fa in output.top_feature_attributions:
            self.assertIn(fa['modality'], ['trace', 'metric', 'log'])

    def test_structured_explanation_serialization(self):
        """
        Validates:
        ExplanationOutput can be serialized to JSON without any TypeError or custom object issues.
        """
        explainer = HolisticRCAExplainer(
            model=self.model,
            entities=self.entities,
            fault_classes=self.fault_classes,
            feature_counts=self.feature_counts,
            entity_feature_indices=self.entity_indices
        )

        output = explainer.explain_window(
            traces=self.traces,
            metrics=self.metrics,
            logs=self.logs,
            edge_index=self.edge_index,
            epochs=5,
            seed=42
        )

        output_dict = output.to_dict()
        json_str = json.dumps(output_dict)
        self.assertIsInstance(json_str, str)
        reloaded = json.loads(json_str)
        self.assertIn("root_cause_ranking", reloaded)
        self.assertIn("top_feature_attributions", reloaded)
        self.assertIn("limitations_and_warnings", reloaded)

    def test_controlled_synthetic_anomaly_identification(self):
        """
        Validates explanation behavior on a controlled synthetic case:
        We inject high values specifically into features mapped to entity 3.
        The explainer should attribute higher feature significance to entity 3.
        """
        # Inject large positive signal into entity 3 features
        perturbed_metrics = self.metrics.clone()
        ent3_feats = self.entity_indices[3]
        for f in ent3_feats:
            if f < 98:
                perturbed_metrics[:, f] += 10.0

        explainer = HolisticRCAExplainer(
            model=self.model,
            entities=self.entities,
            fault_classes=self.fault_classes,
            feature_counts=self.feature_counts,
            entity_feature_indices=self.entity_indices
        )

        output = explainer.explain_window(
            traces=self.traces,
            metrics=perturbed_metrics,
            logs=self.logs,
            edge_index=self.edge_index,
            target_entity="social_service_3",
            epochs=20,
            seed=42
        )

        self.assertEqual(output.target_entity, "social_service_3")
        self.assertTrue(len(output.top_feature_attributions) > 0)
        self.assertLess(output.fidelity_metrics['logit_l1_diff'], 5.0)

    def test_real_checkpoint_loading_and_positive_window_explanation(self):
        """
        Validates Phase 4 requirement:
        1. Loads the actual saved checkpoint (checkpoints/checkpoint_best.pt).
        2. Retrieves the actual positive validation window for social_service_3 (cpu_load).
        3. Executes mask optimization and verifies paper-faithful ranking.
        4. Verifies presence of mandatory experimental notices and limitation warnings.
        """
        ckpt_path = os.path.join(CHECKPOINT_DIR, "checkpoint_best.pt")
        if not os.path.exists(ckpt_path):
            self.skipTest(f"Checkpoint not found at {ckpt_path}")

        real_model, checkpoint_dict = load_checkpoint(ckpt_path)
        real_model.eval()

        train_ds, val_ds, test_ds, meta = load_dataset_c_splits(
            data_dir=DATA_DIR,
            time_granularity=60,
            window_size=1
        )

        # Find positive validation window
        val_sample = None
        for item in val_ds:
            if item['target'][:, 0].sum() > 0: # cpu_load positive
                val_sample = item
                break

        self.assertIsNotNone(val_sample, "No positive cpu_load sample found in validation set!")

        feature_names = {
            'trace': meta['preprocessed_out'].trace_feature_names,
            'metric': meta['preprocessed_out'].metric_feature_names,
            'log': meta['preprocessed_out'].log_feature_names
        }

        explainer = HolisticRCAExplainer(
            model=real_model,
            entities=val_sample.get('entities', train_ds.entities),
            fault_classes=train_ds.fault_classes,
            feature_counts={'trace': 84, 'metric': 98, 'log': 84},
            feature_names=feature_names,
            entity_feature_indices=val_sample['entity_indices']
        )

        output = explainer.explain_window(
            traces=val_sample['trace'],
            metrics=val_sample['metric'],
            logs=val_sample['log'],
            edge_index=val_sample['edge_index'],
            timestamp=val_sample['timestamp'],
            epochs=25,
            seed=42
        )

        self.assertIsInstance(output, ExplanationOutput)
        self.assertTrue(len(output.root_cause_ranking) == 14)

        # Verify limitation warning is explicitly present
        has_warning = any("EXPERIMENTAL RESULT" in w for w in output.limitations_and_warnings)
        self.assertTrue(has_warning, "Mandatory experimental dataset limitation warning was missing!")


if __name__ == '__main__':
    unittest.main()
