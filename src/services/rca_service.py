import os
import sys
import threading
from typing import Dict, List, Optional, Any, Tuple

import torch
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.checkpoint import load_checkpoint
from src.models.holistic_rca import HolisticRCA
from src.models.mask_learning import HolisticRCAExplainer, ExplanationOutput
from src.data.dataset import load_dataset_c_splits, HolisticRCADataset


class RCAService:
    """
    Singleton Inference & Explanation Service for HolisticRCA.
    Caches model checkpoint, dataset splits, graph topology, and preprocessed metadata
    in memory to ensure low-latency API responses.
    """
    _instance: Optional['RCAService'] = None
    _lock = threading.Lock()

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/checkpoint_best.pt",
        data_dir: str = "data_samples/dataset_c",
        device: Optional[str] = None
    ):
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.data_dir = os.path.abspath(data_dir)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model: Optional[HolisticRCA] = None
        self.checkpoint_metadata: Dict[str, Any] = {}
        self.train_dataset: Optional[HolisticRCADataset] = None
        self.val_dataset: Optional[HolisticRCADataset] = None
        self.test_dataset: Optional[HolisticRCADataset] = None
        self.dataset_metadata: Dict[str, Any] = {}
        self.window_index_map: Dict[int, Tuple[HolisticRCADataset, int]] = {}
        self.entities: List[str] = []
        self.fault_classes: List[str] = []
        self.feature_names: Dict[str, List[str]] = {}
        self.explainer: Optional[HolisticRCAExplainer] = None

        self.is_initialized = False
        self._init_error: Optional[str] = None

    @classmethod
    def get_instance(
        cls,
        checkpoint_path: str = "checkpoints/checkpoint_best.pt",
        data_dir: str = "data_samples/dataset_c"
    ) -> 'RCAService':
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(checkpoint_path=checkpoint_path, data_dir=data_dir)
                cls._instance.initialize()
            return cls._instance

    def initialize(self) -> bool:
        """
        Loads checkpoint and preprocessed dataset into memory once.
        """
        if self.is_initialized:
            return True

        try:
            # 1. Load trained model checkpoint
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"Trained checkpoint not found at: {self.checkpoint_path}")

            model, ckpt = load_checkpoint(self.checkpoint_path, device=str(self.device))
            model.to(self.device)
            model.eval()

            # Freeze all parameters
            for p in model.parameters():
                p.requires_grad = False

            self.model = model
            self.checkpoint_metadata = {
                'epoch': ckpt.get('epoch', 0),
                'val_loss': ckpt.get('val_loss', 0.0),
                'train_loss': ckpt.get('train_loss', 0.0),
                'is_best_validated': ckpt.get('is_best_validated', False)
            }

            # 2. Load dataset splits and metadata
            if not os.path.exists(self.data_dir):
                raise FileNotFoundError(f"Dataset directory not found at: {self.data_dir}")

            train_ds, val_ds, test_ds, meta = load_dataset_c_splits(
                data_dir=self.data_dir,
                time_granularity=60,
                window_size=1
            )
            self.train_dataset = train_ds
            self.val_dataset = val_ds
            self.test_dataset = test_ds
            self.dataset_metadata = meta
            self.entities = train_ds.entities
            self.fault_classes = train_ds.fault_classes

            # 3. Build fast lookup index for all 60 timestamps
            self.window_index_map = {}
            for ds in [train_ds, val_ds, test_ds]:
                for idx, ts in enumerate(ds.timestamps):
                    self.window_index_map[int(ts)] = (ds, idx)

            # 4. Extract feature names
            pre_out = meta.get('preprocessed_out')
            if pre_out is not None:
                self.feature_names = {
                    'trace': pre_out.trace_feature_names,
                    'metric': pre_out.metric_feature_names,
                    'log': pre_out.log_feature_names
                }
            else:
                self.feature_names = {
                    'trace': [f"trace_feat_{i}" for i in range(84)],
                    'metric': [f"metric_feat_{i}" for i in range(98)],
                    'log': [f"log_feat_{i}" for i in range(84)]
                }

            # 5. Initialize reusable explainer
            sample_item = val_ds[0]
            self.explainer = HolisticRCAExplainer(
                model=self.model,
                entities=self.entities,
                fault_classes=self.fault_classes,
                feature_counts={'trace': 84, 'metric': 98, 'log': 84},
                feature_names=self.feature_names,
                entity_feature_indices=sample_item['entity_indices'],
                device=self.device
            )

            self.is_initialized = True
            self._init_error = None
            return True

        except Exception as e:
            self.is_initialized = False
            self._init_error = str(e)
            return False

    def get_service_status(self) -> Dict[str, Any]:
        """
        Returns health status, checkpoint information, and dataset summary.
        """
        return {
            'status': 'healthy' if self.is_initialized else 'degraded',
            'is_initialized': self.is_initialized,
            'init_error': self._init_error,
            'device': str(self.device),
            'checkpoint': {
                'path': self.checkpoint_path,
                'available': os.path.exists(self.checkpoint_path),
                'epoch': self.checkpoint_metadata.get('epoch'),
                'val_loss': self.checkpoint_metadata.get('val_loss'),
                'train_loss': self.checkpoint_metadata.get('train_loss'),
                'is_best_validated': self.checkpoint_metadata.get('is_best_validated', False)
            },
            'dataset_info': {
                'total_windows': len(self.window_index_map),
                'train_windows': len(self.train_dataset) if self.train_dataset else 0,
                'val_windows': len(self.val_dataset) if self.val_dataset else 0,
                'test_windows': len(self.test_dataset) if self.test_dataset else 0,
                'entities': self.entities,
                'fault_classes': self.fault_classes,
                'total_features': 266
            },
            'experimental_notice': (
                "EXPERIMENTAL: Model trained on Dataset C slice containing a single continuous cpu_load "
                "fault episode on social_service_3. Results demonstrate paper-faithful RCA mechanics."
            )
        }

    def get_available_windows(self) -> List[Dict[str, Any]]:
        """
        Lists all 60 observation windows with ground-truth labels for reference.
        """
        if not self.is_initialized:
            raise RuntimeError(f"RCAService not initialized: {self._init_error}")

        windows = []
        for ts, (ds, idx) in sorted(self.window_index_map.items()):
            target_tensor = ds.targets[idx] # (14, 3)
            ground_truth_faults = []

            for ent_i, ent_name in enumerate(self.entities):
                for fc_i, fc_name in enumerate(self.fault_classes):
                    if target_tensor[ent_i, fc_i].item() > 0.5:
                        ground_truth_faults.append({
                            'entity': ent_name,
                            'fault_type': fc_name
                        })

            windows.append({
                'timestamp': ts,
                'split': ds.split_name,
                'has_ground_truth_fault': len(ground_truth_faults) > 0,
                'ground_truth_faults': ground_truth_faults
            })

        return windows

    def run_rca(
        self,
        timestamp: int,
        run_explanation: bool = True,
        epochs: int = 50,
        lr: float = 0.05,
        target_entity: Optional[str] = None,
        detection_threshold: float = 0.5,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Executes end-to-end RCA inference and optional mask learning explanation for a window.
        """
        if not self.is_initialized:
            raise RuntimeError(f"RCAService not initialized: {self._init_error}")

        if timestamp not in self.window_index_map:
            raise ValueError(f"Observation window with timestamp {timestamp} not found in Dataset C.")

        ds, idx = self.window_index_map[timestamp]
        sample = ds[idx]

        trace = sample['trace'].to(self.device)
        metric = sample['metric'].to(self.device)
        log = sample['log'].to(self.device)
        edge_index = sample['edge_index'].to(self.device)
        entity_indices = sample['entity_indices'].to(self.device)
        target = sample['target'] # (14, 3)

        # 1. HolisticRCA Model Inference (Strictly Frozen)
        with torch.no_grad():
            preds = self.model(trace, metric, log, entity_indices, edge_index)
            # Default node_type is 'service'
            node_type = list(preds.keys())[0]
            logits = preds[node_type]
            if logits.dim() == 3:
                logits = logits.squeeze(0)
            probs = torch.sigmoid(logits) # (14, 3)

        probs_np = probs.cpu().numpy()

        # Determine detected faults per entity
        per_entity_predictions = []
        any_fault_detected = False
        max_prob_overall = 0.0
        top_faulty_entity = None

        for ent_i, ent_name in enumerate(self.entities):
            class_probs = {
                fc: float(probs_np[ent_i, c_i])
                for c_i, fc in enumerate(self.fault_classes)
            }
            ent_max_p = max(class_probs.values())
            ent_top_class = max(class_probs, key=class_probs.get)
            is_fault = ent_max_p >= detection_threshold

            if is_fault:
                any_fault_detected = True
            if ent_max_p > max_prob_overall:
                max_prob_overall = ent_max_p
                top_faulty_entity = ent_name

            per_entity_predictions.append({
                'entity_name': ent_name,
                'entity_idx': ent_i,
                'is_fault': is_fault,
                'predicted_fault': ent_top_class if is_fault else 'normal',
                'max_probability': round(ent_max_p, 4),
                'probabilities': {k: round(v, 4) for k, v in class_probs.items()}
            })

        # Ground truth reference (clearly labeled)
        ground_truth_records = []
        for ent_i, ent_name in enumerate(self.entities):
            for c_i, fc in enumerate(self.fault_classes):
                if target[ent_i, c_i].item() > 0.5:
                    ground_truth_records.append({
                        'entity': ent_name,
                        'fault_type': fc
                    })

        # 2. Run Mask-Learning Explanation if requested
        explanation_result: Optional[Dict[str, Any]] = None
        if run_explanation:
            expl_target = target_entity if target_entity else (top_faulty_entity if any_fault_detected else None)

            exp_output = self.explainer.explain_window(
                traces=trace,
                metrics=metric,
                logs=log,
                edge_index=edge_index,
                entity_indices=entity_indices,
                timestamp=timestamp,
                target_entity=expl_target,
                epochs=epochs,
                lr=lr,
                detection_threshold=detection_threshold,
                seed=seed
            )
            explanation_result = exp_output.to_dict()

        return {
            'timestamp': timestamp,
            'split_name': ds.split_name,
            'system_id': 'dataset_c_cluster',
            'is_fault_predicted': any_fault_detected,
            'max_fault_probability': round(max_prob_overall, 4),
            'top_faulty_entity': top_faulty_entity if any_fault_detected else None,
            'per_entity_predictions': per_entity_predictions,
            'ground_truth_benchmark': {
                'has_fault': len(ground_truth_records) > 0,
                'faults': ground_truth_records
            },
            'explanation': explanation_result,
            'metadata': {
                'model_checkpoint': os.path.basename(self.checkpoint_path),
                'trained_epoch': self.checkpoint_metadata.get('epoch'),
                'device': str(self.device),
                'epochs_optimized': epochs if run_explanation else 0
            },
            'experimental_notice': (
                "EXPERIMENTAL NOTE: This RCA explanation is generated by learning dynamic relation and feature masks "
                "on a single-episode dataset slice. Root-cause scores indicate model feature/relation attribution, "
                "not proven real-world causality."
            )
        }
