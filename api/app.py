from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib, pandas as pd, uvicorn
from pathlib import Path

MODEL_PATH = "api/model/co2_rf_model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="CO2 Emissions Predictor")

# Serve frontend
frontend_path = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/frontend", StaticFiles(directory=frontend_path), name="frontend")

@app.get("/")
def serve_home():
    return FileResponse(frontend_path / "index.html")

class VehicleRequest(BaseModel):
    year: int
    displ: float = None
    cylinders: float = None
    city08: float = None
    highway08: float = None
    comb08: float = None
    barrels08: float = None
    make: str = None
    model: str = None
    trany: str = None
    VClass: str = None
    fuelType: str = None

@app.post("/predict")
def predict_co2(payload: VehicleRequest):
    try:
        row = pd.DataFrame([payload.model_dump()])
        pred = model.predict(row)[0]
        return {"predicted_co2_g_per_mile": float(pred)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
