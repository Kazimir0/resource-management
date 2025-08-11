from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal, Optional
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path


app = FastAPI(title="ML Inference Service")


MODEL_PATH = Path("/app/model/no_show_model.pkl")
_model = joblib.load(str(MODEL_PATH))


class PredictRequest(BaseModel):
    # Raw fields from the original dataset
    Age: int
    Scholarship: int
    Hipertension: int
    Diabetes: int
    Alcoholism: int
    Handcap: int
    SMS_received: int
    ScheduledDay: str = Field(..., description="ISO datetime, e.g. 2016-04-29T18:38:08Z")
    AppointmentDay: str = Field(..., description="ISO datetime, e.g. 2016-04-29T00:00:00Z")
    Gender: Literal['M', 'F']
    Neighbourhood: str
    threshold: Optional[float] = Field(default=None, description="Optional decision threshold for label; if omitted returns probability only")


class PredictResponse(BaseModel):
    probability_no_show: float
    predicted_label: Optional[int] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Parse datetimes
    scheduled = pd.to_datetime(req.ScheduledDay)
    appointment = pd.to_datetime(req.AppointmentDay)

    # Feature engineering to match training
    days_between = (appointment - scheduled).days
    if days_between < 0:
        days_between = 0
    appointment_dow = appointment.dayofweek
    scheduled_hour = scheduled.hour
    is_weekend = 1 if appointment.dayofweek in (5, 6) else 0

    # Build dataframe in expected shape
    numeric_features = [
        'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism',
        'Handcap', 'SMS_received', 'DaysBetween',
        'AppointmentDayOfWeek', 'ScheduledHour', 'IsWeekend'
    ]
    categorical_features = ['Gender', 'Neighbourhood']

    row = {
        'Age': req.Age,
        'Scholarship': req.Scholarship,
        'Hipertension': req.Hipertension,
        'Diabetes': req.Diabetes,
        'Alcoholism': req.Alcoholism,
        'Handcap': req.Handcap,
        'SMS_received': req.SMS_received,
        'DaysBetween': days_between,
        'AppointmentDayOfWeek': appointment_dow,
        'ScheduledHour': scheduled_hour,
        'IsWeekend': is_weekend,
        'Gender': req.Gender,
        'Neighbourhood': req.Neighbourhood,
    }

    df = pd.DataFrame([row], columns=numeric_features + categorical_features)

    # Predict probability with pipeline
    proba = float(_model.predict_proba(df)[:, 1][0])

    # Optional threshold for a label
    predicted_label = None
    if req.threshold is not None:
        predicted_label = int(proba >= req.threshold)

    return PredictResponse(
        probability_no_show=proba,
        predicted_label=predicted_label,
    )


