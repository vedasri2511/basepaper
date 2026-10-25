# HolisticRCA: End-to-End Multimodal Root Cause Analysis Framework

This repository provides an end-to-end implementation of the research paper:

> **"Holistic Root Cause Analysis for Failures in Cloud-Native Systems Through Observability Data"**
> *Yongqi Han, Chaoyang Deng, Yuhang Chen, Chengyang Luo, Zhenjie Wei, Meng Ma, Ping Wang, Dan Pei*
> **IEEE Transactions on Services Computing (IEEE TSC), 2024**

The framework analyzes cloud-native microservices by fusing multimodal observability data (distributed traces, host/service metrics, and log templates) over a dynamic Resource Entity Relation Graph (RERG) using Graph Attention Networks (GATv2), followed by Mutual Information (MI) Mask Learning for explainable root-cause localization.

---

## 1. Project Status & Architecture Overview

### Current Implementation Scope

1. **Multimodal Preprocessing (`src/data/preprocessor.py`)**:
   - **Traces**: Mean duration, current call counts, and previous-window call counts.
   - **Metrics**: Standardized statistical metrics (CPU, memory, network tx/rx, disk read/write, latency).
   - **Logs**: Semantic template mining via **Drain3** combined with TF-IDF vectorization (vocabulary mined strictly from training windows to prevent leakage).
2. **Dynamic Graph Construction (`src/data/rerg_builder.py`)**:
   - Dynamic Resource Entity Relation Graph (RERG) capturing parent-child call relationships and service deployments.
   - Bidirectional edge expansion and self-loop preservation conforming to PyTorch Geometric contracts.
3. **Representation Learning & GNN Fusion (`src/models/holistic_rca.py`)**:
   - Positional Encoding (Vaswani et al.) applied to temporal feature sequences.
   - Modality-specific Transformer Encoders ($\text{TE}_{mo}$).
   - Entity-type Transformer Encoders ($\text{TE}_{et}$) and linear projection to standardized entity space.
   - Graph Attention Network (**GATv2Conv**) message passing across RERG topology.
4. **Fault Classification & Loss (`src/models/losses.py`, `src/data/label_encoder.py`)**:
   - Multi-label classification supporting multiple concurrent fault classes (`cpu_load`, `network_delay`, `network_loss`).
   - Weighted Binary Cross-Entropy Loss (Equations 6 & 7) mitigating extreme class imbalance.
5. **Mask Learning & Root-Cause Explanation (`src/models/mask_learning.py`)**:
   - Optimizes continuous relation mask $EM \in \mathbb{R}^{n_g \times n_g}$ and observability feature mask $om \in \mathbb{R}^{n_f}$ via Equation (8).
   - Maximizes mutual information via conditional entropy minimization ($\mathcal{L}_{MI}$) with sparsity ($\mathcal{L}_{size}$) and discrete decisiveness ($\mathcal{L}_{ent}$) penalties.
   - Model parameters remain **strictly frozen** during explanation.
   - **Two-dimensional root-cause attribution**:
     - *Dimension 1 (Resource Entities)*: Ranks suspect entities by combining incident relation mask scores and entity feature mask scores.
     - *Dimension 2 (Observability Features)*: Ranks individual metrics, traces, and log templates driving the anomaly.
6. **Inference Service & Backend API (`src/services/rca_service.py`, `src/api.py`)**:
   - FastAPI server with cached checkpoint loading, window lookup, and structured JSON output.
7. **Interactive Dashboard (`frontend/`)**:
   - React 19 + Vite 8 dashboard for selecting observation windows, running live RCA, and inspecting rankings, feature bars, and diagnostics.
8. **Automated Test Suite (`tests/`)**:
   - 8 test files covering all components with **60 passing tests**.

### Architecture Flow

```
[Raw Observability Data]
  ├── Traces (spans.csv)    ──> Preprocessor (Duration & Counts) ──┐
  ├── Metrics (metrics.csv)  ──> Preprocessor (Standardized)      ──┼──> [Positional Encoding]
  └── Logs (logs.csv)       ──> Drain3 Template Miner + TF-IDF   ──┘           │
                                                                               ▼
[Topology Spans] ───────────────> Dynamic RERG Builder ────────────> [Modality Encoders]
                                      (Graph Edges E)                          │
                                             │                                 ▼
                                             │                       [Entity Integrators]
                                             │                                 │
                                             ▼                                 ▼
                                     [GATv2 Message Passing] <───────── Node Matrix V
                                             │
                                             ├──> Multi-Label Classifiers ──> Predicted Faults (Dim 3)
                                             │
                                             ▼
                             [Section IV.F Mask Learning (Eq. 8)]
                                 (Model parameters strictly frozen)
                                             ├──> Relation Mask EM  ──> Entity Ranking (Dim 1)
                                             └──> Feature Mask om   ──> Telemetry Attribution (Dim 2)
                                             │
                                             ▼
                                  [FastAPI Service (8000)]
                                             │
                                             ▼
                                 [React Dashboard (5173)]
```

---

## 2. Component Inventory & Execution Modes

| Component | Technology | Role | Supported Execution Mode | Status in Repo |
| :--- | :--- | :--- | :--- | :--- |
| **Backend API** | FastAPI / Uvicorn | RCA inference, window metadata, mask learning | Local Python process (port 8000) | **Verified** |
| **Frontend Dashboard** | React 19 / Vite 8 | Interactive observation window selection and RCA visualizer | Local Node.js process (port 5173) | **Verified** |
| **Model & Checkpoints** | PyTorch / PyG | GATv2 model weights, history metadata | Committed (`checkpoints/checkpoint_best.pt`) | **Verified** |
| **Dataset C** | CSV / Drain3 | 60 one-minute observation windows across 14 services | Committed (`data_samples/dataset_c/`) | **Verified** |
| **Automated Tests** | Python `unittest` | 60 regression and integration tests | Local test runner (`tests/`) | **Verified (60/60)** |
| **Jaeger Tracing** | Kubernetes manifest | All-in-one distributed tracing collector & UI | Kubernetes (`jaeger.yaml`) | Configuration present (unverified runtime) |
| **Prometheus** | Kubernetes manifest | Cluster metric scraping and storage | Kubernetes (`prometheus.yaml`) | Configuration present (unverified runtime) |
| **Docker Compose** | Docker / Compose | Containerized multi-service orchestration | *Not configured* (no Compose files exist) | **Not Configured** |

> [!NOTE]
> **Docker / Containerized Mode**: No `Dockerfile` or `docker-compose.yml` is present in this repository. The primary supported execution mode is **Local / Manual Execution**.
>
> **Kubernetes Mode**: Manifests exist for standalone observability infrastructure (`jaeger.yaml` and `prometheus.yaml`). The HolisticRCA application (API + Dashboard) currently runs locally and does not require a live Kubernetes cluster to analyze Dataset C.

---

## 3. Prerequisites

### Local Development Environment
- **Operating System**: Windows 10/11 (PowerShell commands provided below; bash equivalents also documented), Linux, or macOS.
- **Python**: Version **3.10** or **3.11** (tested on CPython 3.11.8).
- **Node.js**: Version **18+** or **20+** with `npm` (tested on Node v20 / npm v10).
- **Virtual Environment**: Pre-configured at `basepaper/venv` (or recreate using `requirements.txt`).

### Optional Prerequisites (for Kubernetes Manifests Only)
- `kubectl` client (v1.28+).
- Accessible Kubernetes cluster (Minikube, kind, or Docker Desktop Kubernetes) — *not required for core RCA analysis*.

---

## 4. Repository Structure Relevant to Execution

```
basepaper/
├── checkpoints/
│   ├── checkpoint_best.pt        # Trained HolisticRCA model weights (Epoch 3, Val Loss 0.9113)
│   ├── checkpoint_latest.pt      # Latest checkpoint weights
│   └── training_history.json     # Training history, loss curves, and split metadata
├── data_samples/
│   └── dataset_c/                # Dataset C observability CSVs
│       ├── traces.csv            # 60 minutes of distributed trace spans
│       ├── metrics.csv           # 60 minutes of CPU, memory, network, and disk metrics
│       ├── logs.csv              # 60 minutes of raw application logs
│       └── labels.csv            # Ground-truth fault labels (840 entity-window rows)
├── frontend/
│   ├── package.json              # React 19, Vite 8, Oxlint dependencies
│   ├── vite.config.js            # Vite configuration
│   ├── src/
│   │   ├── App.jsx               # Main RCA dashboard component
│   │   ├── index.css             # Glassmorphism styling, ranking tables, and feature bars
│   │   └── main.jsx              # React DOM entry point
│   └── dist/                     # Verified production build output
├── src/
│   ├── api.py                    # FastAPI application and REST endpoints
│   ├── evaluate.py               # AC@k and multi-label classification evaluation utilities
│   ├── train.py                  # End-to-end training pipeline with chronological splits
│   ├── data/
│   │   ├── dataset.py            # Dataset C split loader, collate function, and metadata
│   │   ├── generate_dataset_c.py # Dataset C synthetic generator & label formalization
│   │   ├── label_encoder.py      # Multi-label binary matrix target encoder
│   │   ├── preprocessor.py       # Multimodal feature extraction (Trace, Metric, Log Drain3)
│   │   └── rerg_builder.py       # Dynamic RERG topology builder
│   ├── models/
│   │   ├── checkpoint.py         # PyTorch checkpoint save/load utilities
│   │   ├── holistic_rca.py       # Positional encoding, encoders, GATv2Conv fusion
│   │   ├── losses.py             # WeightedBCELoss (Eq. 6-7)
│   │   └── mask_learning.py      # MaskLearner & HolisticRCAExplainer (Eq. 8)
│   └── services/
│       └── rca_service.py        # Singleton caching inference & explanation service
├── tests/                        # 8 test files (60 automated unit and integration tests)
├── jaeger.yaml                   # Optional Kubernetes manifest for Jaeger All-in-One
├── prometheus.yaml               # Optional Kubernetes manifest for Prometheus RBAC & scraping
├── requirements.txt              # Python package dependencies
└── README.md                     # Project documentation
```

---

## 5. Mode A: Local / Manual Execution (Recommended & Tested)

The application consists of two communicating services:
1. **Backend API**: FastAPI / Uvicorn server running on `http://localhost:8000`.
2. **Frontend Dashboard**: Vite / React server running on `http://localhost:5173`.

> [!IMPORTANT]
> **Working Directory Rule**: Always execute the backend command from the **repository root** (`basepaper/`), not from `src/`. Path resolution for `checkpoints/` and `data_samples/` is relative to the repository root.

### Terminal 1: Start Backend API (FastAPI)

#### Windows (PowerShell):
```powershell
# 1. Navigate to repository root
cd c:\Users\LENOVO\OneDrive\Documents\Major_project\basepaper

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 3. Start Uvicorn server
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

#### Linux / macOS (Bash):
```bash
cd /path/to/basepaper
source venv/bin/activate
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

- **Expected Terminal Output**:
  ```text
  INFO:     Started server process [PID]
  INFO:     Waiting for application startup.
  INFO:     Application startup complete.
  INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
  ```
- **Readiness Verification**:
  ```powershell
  # In PowerShell:
  Invoke-RestMethod http://localhost:8000/api/health
  ```
  Expected JSON output:
  ```json
  {"status":"healthy","is_initialized":true,"device":"cpu","checkpoint":{"available":true,"epoch":3}}
  ```
- **Shutdown**: Press `Ctrl + C` in Terminal 1.

---

### Terminal 2: Start Frontend Dashboard (React / Vite)

#### Windows (PowerShell):
```powershell
# 1. Navigate to frontend directory
cd c:\Users\LENOVO\OneDrive\Documents\Major_project\basepaper\frontend

# 2. Install dependencies (only required on first setup)
npm install

# 3. Start Vite dev server
npm run dev
```

#### Linux / macOS (Bash):
```bash
cd /path/to/basepaper/frontend
npm install
npm run dev
```

- **Expected Terminal Output**:
  ```text
    VITE v8.3.1  ready in 180 ms

    ➜  Local:   http://localhost:5173/
    ➜  Network: use --host to expose
    ➜  press h + enter to show help
  ```
- **Readiness Verification**:
  Open your web browser and navigate to:
  **`http://localhost:5173`**
- **Shutdown**: Press `Ctrl + C` in Terminal 2.

---

## 6. How to Run an RCA Analysis via the Dashboard

1. Ensure both the backend (`http://localhost:8000`) and frontend (`http://localhost:5173`) are running.
2. Open **`http://localhost:5173`** in your browser.
3. Observe the top-right status badge: it should indicate **`Checkpoint Ready (Epoch 3)`**.
4. In the left panel:
   - Select an observation window from the dropdown. For example:
     - **Fault window**: `[VAL] T=1696002220 • FAULT: social_service_3 (cpu_load)`.
     - **Normal window**: `[TRAIN] T=1696000000 • Normal`.
   - Select mask learning epochs (e.g. `50 epochs` for standard convergence).
   - Click **Run Root Cause Analysis**.
5. The spinner will display while mask learning optimizes Equation (8).
6. Inspect the live results:
   - **Fault Status**: Shows `Fault Detected` with predicted probability ($> 90\%$).
   - **Ground Truth Benchmark**: Explicitly displays `social_service_3: cpu_load` for objective comparison.
   - **Candidate Root-Cause Entities (Dimension 1)**: Ranks services by learned relation mask scores and feature attributions.
   - **Implicated Observability Features (Dimension 2)**: Visual score progress bars showing top attributed metrics (e.g. `metric_social_service_3_cpu_usage`), traces, and logs.
   - **Optimization Diagnostics**: Displays initial loss, final loss, loss reduction, and explanation fidelity ($\Delta \text{prob}$).

---

## 7. Mode B: Docker / Containerized Execution

> [!WARNING]
> **Status: Not Configured.**
> There are no `Dockerfile` or `docker-compose.yml` configurations in this repository. Containerized execution is currently unsupported.
>
> To containerize this application in a future release, the following components would need to be authored:
> 1. A Python 3.11 container image building `src/` and exposing port 8000.
> 2. A Node.js container image building `frontend/` with Nginx or Vite preview exposing port 5173.
> 3. A `docker-compose.yml` orchestrating backend, frontend, and shared volumes for `checkpoints/` and `data_samples/`.

---

## 8. Mode C: Kubernetes Manifests (Observability Infrastructure)

The repository provides two standalone Kubernetes manifests in the root directory:
- `jaeger.yaml`: Deploys Jaeger All-in-One distributed tracing.
- `prometheus.yaml`: Deploys Prometheus with pod scraping configuration.

These manifests deploy telemetry collectors for a Kubernetes cluster. They are **optional infrastructure components** and are not required for offline Dataset C model inference.

### Deploying the Observability Stack (if a Kubernetes cluster is available)

```bash
# 1. Apply Jaeger Deployment & Services
kubectl apply -f jaeger.yaml

# 2. Apply Prometheus RBAC, ConfigMap, Deployment & Service
kubectl apply -f prometheus.yaml

# 3. Verify pods are running
kubectl get pods -l 'app in (jaeger, prometheus)'
```

### Accessing Observability UIs (Port Forwarding)

```bash
# Terminal A: Port-forward Jaeger Web UI (Port 16686)
kubectl port-forward svc/jaeger-query 16686:16686

# Terminal B: Port-forward Prometheus Web UI (Port 9090)
kubectl port-forward svc/prometheus 9090:9090
```
- Jaeger Query UI: `http://localhost:16686`
- Prometheus Expression Browser: `http://localhost:9090`

### Tearing Down the Observability Stack

```bash
kubectl delete -f prometheus.yaml
kubectl delete -f jaeger.yaml
```

---

## 9. Service URLs, Ports, and Verification Checklist

| Service / Interface | Host URL | Port | Execution Mode | Verification Command / Check |
| :--- | :--- | :---: | :--- | :--- |
| **Backend Health** | `http://localhost:8000/api/health` | 8000 | Local | `Invoke-RestMethod http://localhost:8000/api/health` |
| **Observation Windows** | `http://localhost:8000/api/windows` | 8000 | Local | `Invoke-RestMethod http://localhost:8000/api/windows` |
| **RCA Analysis API** | `http://localhost:8000/api/analyze` | 8000 | Local | `Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/analyze -Body '{"timestamp":1696002220,"epochs":10}' -ContentType 'application/json'` |
| **Interactive Docs (Swagger)** | `http://localhost:8000/docs` | 8000 | Local | Open in browser |
| **Frontend Dashboard** | `http://localhost:5173` | 5173 | Local | Open in browser; header displays `Checkpoint Ready` |
| **Jaeger UI (Optional)** | `http://localhost:16686` | 16686 | Kubernetes Port-Forward | Open in browser (if `jaeger.yaml` is deployed) |
| **Prometheus (Optional)** | `http://localhost:9090` | 9090 | Kubernetes Port-Forward | Open in browser (if `prometheus.yaml` is deployed) |

---

## 10. Training vs. Inference Workflows

### Running Inference & Explanations (Default)
Inference does not require retraining. The pre-trained checkpoint (`checkpoints/checkpoint_best.pt`) is automatically loaded by the backend service. All 60 observation windows are accessible via the interactive dashboard.

### Retraining the Model from Scratch (Optional)
To retrain HolisticRCA using Dataset C:

```powershell
cd c:\Users\LENOVO\OneDrive\Documents\Major_project\basepaper
.\venv\Scripts\Activate.ps1

python src/train.py `
  --data_dir data_samples/dataset_c `
  --checkpoint_dir checkpoints `
  --epochs 10 `
  --lr 0.001 `
  --batch_size 4 `
  --window_size 1 `
  --seed 42
```
- Reads raw CSVs from `data_samples/dataset_c/`.
- Fits Drain3 log vocabulary strictly on training windows ($T < 1696002160$).
- Computes chronological splits (Train: 36, Val: 12, Test: 12 windows).
- Evaluates macro/micro precision, recall, and F1 at each epoch.
- Saves best validation weights to `checkpoints/checkpoint_best.pt` and history to `checkpoints/training_history.json`.

### Regenerating Dataset C (Optional)
If raw CSVs in `data_samples/dataset_c/` need to be regenerated:
```powershell
python -c "from src.data.generate_dataset_c import generate_social_network_dataset, generate_labels; generate_social_network_dataset(); generate_labels()"
```

---

## 11. Automated Test Suites

The repository includes 8 test suites verifying every layer of the architecture:

```powershell
# Run all test suites from the repository root:
.\venv\Scripts\Activate.ps1

python tests/test_preprocessor.py             # 6 tests: feature extraction, Drain3, TF-IDF
python tests/test_rerg_builder.py             # 11 tests: dynamic RERG graph topology
python tests/test_model.py                    # 8 tests: positional encoding, GATv2Conv fusion
python tests/test_classification_and_loss.py  # 8 tests: LabelEncoder, WeightedBCELoss
python tests/test_dataset_and_training.py     # 8 tests: splits, training step, checkpointing
python tests/test_mask.py                     # 1 test: legacy mask forward pass
python tests/test_mask_learning.py            # 8 tests: mask optimization, gradient flow, frozen weights
python tests/test_rca_service_and_api.py      # 10 tests: RCAService singleton, API endpoints

# Frontend build verification:
cd frontend
npm run build                                 # Builds production bundle in ~1s
npm run lint                                  # Oxlint check (0 errors, 0 warnings)
```

**Total automated test coverage**: **60 tests passed (100%)**.

---

## 12. Troubleshooting & Common Configuration Errors

1. **`FileNotFoundError: Checkpoint file not found: checkpoints/checkpoint_best.pt`**:
   - *Cause*: Starting Uvicorn from inside `src/` instead of the repository root.
   - *Fix*: Run Uvicorn from the repository root:
     `python -m uvicorn src.api:app --host 0.0.0.0 --port 8000`.
2. **Frontend Shows "Checking API..." or "Checkpoint Unavailable"**:
   - *Cause*: Backend API is not running or running on an unexpected port.
   - *Fix*: Start Terminal 1 with `python -m uvicorn src.api:app --port 8000` before opening the dashboard.
3. **PowerShell Script Execution Policy Error**:
   - *Cause*: Execution of PowerShell scripts is restricted by default on Windows.
   - *Fix*: Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating `venv`.
4. **Port 8000 or 5173 Already in Use**:
   - *Cause*: A previous process is still bound to the port.
   - *Fix*: Identify and terminate the lingering process:
     ```powershell
     Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess
     Stop-Process -Id <PID>
     ```

---

## 13. Research Integrity & Known Limitations

- **Dataset C Scale**: The current Dataset C slice contains 60 one-minute observation windows and one continuous `cpu_load` episode on `social_service_3` (6 positive windows in Train, 4 in Val, 0 in Test).
- **Fault Class Distribution**: Classes `network_delay` and `network_loss` have 0 positive occurrences in this slice.
- **Attribution vs. Causality**: Mask learning optimizes Equation (8) to identify the compact subgraph and subset of telemetry features that best preserve model predictions. These scores quantify model attribution and explainability; they do not constitute proven physical causality across arbitrary unseen cloud architectures.
- **Experimental Notice**: All model outputs and explanations generated on this dataset slice are labeled as **experimental**.
