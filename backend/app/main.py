from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI(
    title="Universal IMU-to-Health API",
    version="0.1.0",
    description="Hackathon prototype API for universal IMU motion embeddings."
)

@app.get("/health")
def health_check():
    return {"status": "ok"}


# ---- Schemas ----

class InferRequest(BaseModel):
    device_style: str  # "A" or "B"
    sampling_rate: int
    window_seconds: float
    imu: List[List[float]]  # shape: [T, 6] = [ax, ay, az, gx, gy, gz]


class InferResponse(BaseModel):
    activity: str
    probabilities: Dict[str, float]
    embedding_2d: List[List[float]]  # e.g., [[x1, y1], [x2, y2], ...]


# ---- Stub inference endpoint ----

@app.post("/infer", response_model=InferResponse)
def infer(request: InferRequest) -> InferResponse:
    # TODO: replace this stub with real model later

    # Very simple rule-based stub: length of sequence drives fake activity
    T = len(request.imu)
    if T < 80:
        activity = "sitting"
    elif T < 150:
        activity = "walking"
    else:
        activity = "running"

    # Dummy probabilities
    probs = {
        "sitting": 0.1,
        "walking": 0.1,
        "running": 0.1,
        "other": 0.1,
    }
    probs[activity] = 0.7

    # Dummy 2D embedding path (line)
    embedding_2d = [[i * 0.01, i * 0.01] for i in range(T)]

    return InferResponse(
        activity=activity,
        probabilities=probs,
        embedding_2d=embedding_2d
    )
