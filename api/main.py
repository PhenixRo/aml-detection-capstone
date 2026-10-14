from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import pandas as pd

app = FastAPI(
    title="AML Detection API",
    description="Anti-Money Laundering Prediction API",
    version="1.0"
)

# Load Model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "aml_model.pkl"

model = joblib.load(MODEL_PATH)


class AMLRequest(BaseModel):
    Sender_account: int
    Receiver_account: int
    Amount: float

    Payment_currency: int
    Received_currency: int

    Sender_bank_location: int
    Receiver_bank_location: int

    Payment_type: int

    Cross_Border: int
    Hour: int
    Night_Transaction: int
    Round_Amount: int

    Year: int
    Month: int
    Day: int


@app.get("/")
def home():
    return {
        "message": "AML Detection API Running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: AMLRequest):

    input_df = pd.DataFrame([{
        "Sender_account": data.Sender_account,
        "Receiver_account": data.Receiver_account,
        "Amount": data.Amount,
        "Payment_currency": data.Payment_currency,
        "Received_currency": data.Received_currency,
        "Sender_bank_location": data.Sender_bank_location,
        "Receiver_bank_location": data.Receiver_bank_location,
        "Payment_type": data.Payment_type,
        "Cross_Border": data.Cross_Border,
        "Hour": data.Hour,
        "Night_Transaction": data.Night_Transaction,
        "Round_Amount": data.Round_Amount,
        "Year": data.Year,
        "Month": data.Month,
        "Day": data.Day
    }])

    prediction = model.predict(input_df)[0]

    return {
        "prediction": int(prediction)
    }