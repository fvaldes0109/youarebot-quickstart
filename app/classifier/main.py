from fastapi import FastAPI, requests
from pydantic import BaseModel
import mlflow
import mlflow.pytorch
from transformers import AutoTokenizer
import logging
import torch
import os
import dotenv

dotenv.load_dotenv()

class MessageModel(BaseModel):
    text: str

MODEL_NAME = "lora-bot-detector"
MODEL_VERSION = "latest"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

logger.info(f"Loading model {MODEL_NAME} version {MODEL_VERSION}...")

model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
model.eval()

logger.info("Model loaded successfully!")

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

@app.get("/health")
def health():
    return {"status": "ok from classifier"}

@app.post("/predict")
def predict(msg: MessageModel):
    inputs = tokenizer(
        msg.text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    predicted_proba = float(logits[0][predicted_class])

    if predicted_class == 0:
        predicted_proba = 1 - predicted_proba

    return {"probability": predicted_proba}
