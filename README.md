# HolisticRCA: End-to-End AIOps Framework for Cloud-Native Systems

This repository contains a full-stack, real-time implementation of a Deep Learning-based Root Cause Analysis (RCA) framework, modeled after the academic paper **"HolisticRCA: A Holistic Root Cause Analysis Framework for Cloud-Native Microservices"**. 

This project goes beyond offline CSV datasets by fully integrating a PyTorch Graph Attention Network (GAT) with a **live Kubernetes microservice cluster**, allowing for real-time anomaly detection and fault localization.

---

## 🚀 Tech Stack

### 1. Artificial Intelligence & Deep Learning
*   **PyTorch & PyTorch Geometric**: Core frameworks used to build the Graph Neural Networks (GAT) and Transformer encoders.
*   **Mutual Information (MI) Mask Learning**: Used for providing explainable AI (XAI) insights into which features caused the anomaly.
*   **Pandas & NumPy**: For high-performance data parsing and mathematical evaluation.

### 2. Backend & API
*   **Python**: Core programming language.
*   **FastAPI & Uvicorn**: High-performance asynchronous REST API framework used to wrap the PyTorch model and expose it to the web.

### 3. Frontend Dashboard
*   **React (via Vite)**: Lightning-fast frontend JavaScript framework.
*   **Vanilla CSS**: Premium dark-mode, glassmorphism UI design with dynamic animations for high-quality data visualization.

### 4. Cloud Infrastructure & DevOps
*   **Docker & Kubernetes (Minikube)**: Containerization and orchestration engines running the live cloud environment.
*   **Google Online Boutique (Hipstershop)**: A real-world 10-tier e-commerce microservice application deployed into the cluster to act as the testing target.
*   **Prometheus**: Real-time metric scraping.
*   **Jaeger**: Distributed network tracing.
*   **Chaos Engineering**: Used to manually inject hard crashes (like `CrashLoopBackOff`) into the cluster for the AI to detect.

---

## ⚙️ Architecture

1.  **Observability Data Collection**: The cluster produces Traces, Metrics, and Logs. 
2.  **Preprocessing (`src/data/preprocessor.py`)**: The raw data is vectorized using Transformers and Drain3 log parsing.
3.  **Graph Fusion (`src/models/holistic_rca.py`)**: A Resource Entity Relation Graph (RERG) connects Nodes, Pods, and Services. Information is propagated through a Graph Attention Network (GAT).
4.  **Live Detection (`src/api.py`)**: The FastAPI backend continuously monitors the live Kubernetes orchestrator and Prometheus.
5.  **Visualization (`frontend/src/App.jsx`)**: The React dashboard displays the precise microservice responsible for the fault and recommends actions.

---

## 🛠️ Step-by-Step Execution Guide

Follow these steps to run the complete end-to-end project on your local machine.

### Phase 1: Start the Cloud Environment
1.  Open **Docker Desktop** and ensure **Kubernetes** is enabled in the settings.
2.  Deploy the 10-tier Hipstershop microservice application:
    ```bash
    kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/microservices-demo/main/release/kubernetes-manifests.yaml
    ```
3.  Deploy the Observability Stack (Jaeger & Prometheus):
    ```bash
    kubectl apply -f jaeger.yaml
    kubectl apply -f prometheus.yaml
    ```
4.  Verify all pods are running:
    ```bash
    kubectl get pods
    ```

### Phase 2: Start the Backend AI (FastAPI)
1.  Open a terminal in the project root directory.
2.  Activate the Python virtual environment:
    ```bash
    .\venv\Scripts\Activate.ps1
    ```
3.  Navigate to the `src` folder and start the Uvicorn server:
    ```bash
    cd src
    uvicorn api:app --host 0.0.0.0 --port 8000
    ```
    *(The backend is now actively monitoring the Kubernetes cluster).*

### Phase 3: Start the Frontend Dashboard
1.  Open a second, new terminal in the project root directory.
2.  Navigate to the frontend folder and start the Vite dev server:
    ```bash
    cd frontend
    npm run dev
    ```
3.  Open your browser and navigate to **`http://localhost:5173`**.

### Phase 4: Inject Chaos & Test the AI
1.  In your dashboard, click **"Run Root Cause Analysis"**. The AI will report that the system is perfectly healthy.
2.  To prove the AI works, open a terminal and forcefully break the Checkout microservice:
    ```bash
    kubectl patch deployment checkoutservice -p '{"spec":{"template":{"spec":{"containers":[{"name":"server","command":["/bin/sh", "-c", "exit 1"]}]}}}}'
    ```
3.  Instantly return to the browser and click **"Run Root Cause Analysis"** again.
4.  The dashboard will instantly detect the anomaly, highlight the `checkoutservice`, and display the exact root cause (`CrashLoopBackOff`) along with recommended scaling policies!
