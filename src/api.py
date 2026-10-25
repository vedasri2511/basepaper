import os
import sys
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure repository root is on python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.services.rca_service import RCAService

app = FastAPI(
    title="HolisticRCA Diagnostic & Explanation API",
    description="End-to-End Holistic Root Cause Analysis with Multi-Modal Positional GNN and Mask Learning",
    version="1.8.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------

class AnalysisRequest(BaseModel):
    timestamp: Optional[int] = Field(None, description="Exact observation window bucket timestamp (e.g. 1696002220)")
    timestamp_start: Optional[int] = Field(None, description="Fallback start timestamp from legacy UI")
    timestamp_end: Optional[int] = Field(None, description="Optional end timestamp")
    system_id: Optional[str] = Field("dataset_c_cluster", description="Monitored cluster identifier")
    run_explanation: bool = Field(True, description="Whether to optimize relation and feature masks for RCA")
    epochs: int = Field(50, ge=5, le=200, description="Mask optimization iterations (default: 50)")
    target_entity: Optional[str] = Field(None, description="Specific entity to target for explanation, or auto-detect")


class HealthResponse(BaseModel):
    status: str
    is_initialized: bool
    init_error: Optional[str] = None
    device: str
    checkpoint: Dict[str, Any]
    dataset_info: Dict[str, Any]
    experimental_notice: str


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
def get_health_status():
    """
    Returns service health, checkpoint availability, and dataset window metadata.
    Does not assert unvalidated generalizability.
    """
    service = RCAService.get_instance()
    return service.get_service_status()


@app.get("/api/windows")
def list_observation_windows():
    """
    Returns all 60 chronological Dataset C observation windows with their
    ground-truth labels for reference in the interactive dashboard.
    """
    try:
        service = RCAService.get_instance()
        windows = service.get_available_windows()
        return {
            'total_windows': len(windows),
            'windows': windows
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load observation windows: {str(e)}")


@app.post("/predict")
@app.post("/api/analyze")
def analyze_observation_window(request: AnalysisRequest):
    """
    Main RCA Analysis endpoint:
    Runs frozen HolisticRCA inference and Section IV.F mask learning on the selected window.
    Supports both legacy /predict and canonical /api/analyze endpoints.
    """
    try:
        service = RCAService.get_instance()
        if not service.is_initialized:
            success = service.initialize()
            if not success:
                raise HTTPException(status_code=503, detail=f"RCAService initialization failed: {service._init_error}")

        # Resolve timestamp from request
        target_ts = request.timestamp
        if target_ts is None and request.timestamp_start is not None:
            # Map timestamp_start to the closest available window timestamp
            available_ts = sorted(service.window_index_map.keys())
            if not available_ts:
                raise HTTPException(status_code=500, detail="No observation windows available.")

            # Find closest timestamp >= timestamp_start, or nearest
            matching = [t for t in available_ts if t >= request.timestamp_start]
            target_ts = matching[0] if matching else available_ts[-1]

        if target_ts is None:
            # Default to first positive validation window if none provided
            val_ds = service.val_dataset
            if val_ds and len(val_ds) > 0:
                target_ts = val_ds.timestamps[0]
            else:
                target_ts = list(service.window_index_map.keys())[0]

        result = service.run_rca(
            timestamp=target_ts,
            run_explanation=request.run_explanation,
            epochs=request.epochs,
            target_entity=request.target_entity
        )

        # Add legacy compatibility fields for older components if needed
        result['faulty_entities'] = {
            'service': [p['entity_name'] for p in result['per_entity_predictions'] if p['is_fault']]
        }
        rec_actions = []
        if result['is_fault_predicted']:
            top_e = result.get('top_faulty_entity', 'unknown_service')
            rec_actions.append(f"AI ALERT: Anomaly detected on {top_e}. Root-cause explanation generated below.")
            rec_actions.append(f"Action: Inspect top attributed observability features and relation topology.")
        else:
            rec_actions.append("System is normal in this observation window. Background baseline explanation generated.")
        result['recommended_actions'] = rec_actions

        return result

    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RCA Analysis failed: {str(e)}")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
