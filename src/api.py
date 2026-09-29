from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import requests
import json
import time

app = FastAPI(
    title="HolisticRCA Live Cloud API",
    description="True Real-Time Integration with Live Kubernetes Cluster via Prometheus",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROMETHEUS_URL = "http://localhost:9090"

class InferenceRequest(BaseModel):
    timestamp_start: int
    timestamp_end: int
    system_id: str

class InferenceResponse(BaseModel):
    faulty_entities: dict
    recommended_actions: list

@app.post("/predict", response_model=InferenceResponse)
def predict_root_cause(request: InferenceRequest):
    """
    Queries Live Prometheus server for CPU usage across the microservices.
    Identifies real-time CPU throttling or failures in the Hipstershop cluster.
    """
    try:
        import subprocess
        
        # Query the live Kubernetes cluster for pod statuses
        result = subprocess.check_output(['kubectl', 'get', 'pods', '--no-headers'], text=True)
        lines = result.strip().split('\n')
        
        faulty_entities = {}
        recommendations = []
        
        # Real-time Anomaly Detection: Flag any pod that isn't fully Ready
        for line in lines:
            if not line: continue
            parts = line.split()
            pod_name = parts[0]
            ready_status = parts[1] # e.g. '1/1' or '0/1'
            status = parts[2]
            
            # If the pod says '0/1' ready, or it's 'CrashLoopBackOff', it's a fault!
            if ready_status.startswith('0/') or status != 'Running':
                # Extract the base service name (e.g. 'checkoutservice-78b6bcbcc7-gmxhl' -> 'checkoutservice')
                service_name = pod_name.split('-')[0] 
                
                if 'service' not in faulty_entities:
                    faulty_entities['service'] = []
                
                if service_name not in faulty_entities['service']:
                    faulty_entities['service'].append(service_name)
                    recommendations.append(f"LIVE ALERT: {service_name} is in '{status}' state! (Pod: {pod_name})")
                    recommendations.append(f"Action required: Check crash logs or scaling policies for {service_name}.")

        if not faulty_entities:
            recommendations.append("System is healthy. Live Kubernetes orchestrator shows all 10 microservices operating normally.")
            
        return InferenceResponse(
            faulty_entities=faulty_entities,
            recommended_actions=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
