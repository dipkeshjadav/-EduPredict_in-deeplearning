from pathlib import Path
import pickle
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tensorflow import keras


# ==============================
# PROJECT PATHS
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "student_model.keras"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"
FRONTEND_DIR = BASE_DIR / "frontend"


# ==============================
# FASTAPI
# ==============================

app = FastAPI(
    title="EduPredict AI",
    description="AI Student Performance Prediction API",
    version="1.0.0"
)


# ==============================
# CORS
# ==============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# LOAD TRAINED MODEL
# ==============================

print("Loading ANN model...")

model = keras.models.load_model(MODEL_PATH)

print("ANN model loaded successfully!")


# ==============================
# LOAD SCALER
# ==============================

with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)

print("Scaler loaded successfully!")


# ==============================
# SERVE FRONTEND FILES
# ==============================

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)


# ==============================
# INPUT MODEL
# ==============================

class StudentData(BaseModel):

    study_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    previous_score: float = Field(
        ...,
        ge=0,
        le=100
    )

    attendance: float = Field(
        ...,
        ge=0,
        le=100
    )

    sleep_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    assignments_completed: float = Field(
        ...,
        ge=0
    )

    practice_test_score: float = Field(
        ...,
        ge=0,
        le=100
    )


# ==============================
# HOME PAGE
# ==============================

@app.get("/", include_in_schema=False)
def home():

    return FileResponse(
        FRONTEND_DIR / "index.html"
    )


# ==============================
# HEALTH CHECK
# ==============================

@app.get("/health")
def health():

    return {
        "status": "online",
        "model": "ANN",
        "model_loaded": True
    }


# ==============================
# PREDICTION
# ==============================

@app.post("/predict")
def predict(data: StudentData):

    student = np.array([[
        data.study_hours,
        data.previous_score,
        data.attendance,
        data.sleep_hours,
        data.assignments_completed,
        data.practice_test_score
    ]], dtype=float)


    # Apply same scaling used during training
    student_scaled = scaler.transform(student)


    # ANN prediction
    prediction = model.predict(
        student_scaled,
        verbose=0
    )


    score = float(prediction[0][0])


    # Keep score between 0 and 100
    score = max(0, min(100, score))


    return {
        "predicted_score": round(score, 2)
    }