from fastapi import FastAPI, HTTPException
import requests
import dotenv
import os

from models import GetMessageRequestModel, GetMessageResponseModel, IncomingMessage, Prediction

from uuid import uuid4

dotenv.load_dotenv()

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok from api"}

@app.post("/get_message", response_model=GetMessageResponseModel)
async def get_message(body: GetMessageRequestModel):
    """
    This functions receives a message from HumanOrNot and returns a response
        Parameters (JSON from POST-request):
            body (GetMessageRequestModel): model with request data
                dialog_id (UUID4): ID of the dialog where the message was sent
                last_msg_text (str): text of the message
                last_message_id (UUID4): ID of this message

        Returns (JSON from response):
            GetMessageResponseModel: model with response data
                new_msg_text (str): Ответ бота
                dialog_id (str): ID диалога
    """
    print(
        f"Received message dialog_id: {body.dialog_id}, last_msg_id: {body.last_message_id}"
    )

    response = requests.post(
        os.getenv("LLM_URL") + '/v1/chat/completions',
        json={
            "text": body.last_msg_text,
        }
    )

    return GetMessageResponseModel(
        new_msg_text=response.json(), dialog_id=body.dialog_id
    )

@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage) -> Prediction:
    """
    Endpoint to save a message and get the probability
    that this message if from bot .

    Returns a `Prediction` object.
    """

    is_bot_response = requests.post(
        os.getenv("CLASSIFIER_URL") + '/predict',
        json={
            "text": msg.text,
        }
    ).json()
    prediction_id = uuid4()

    bot_proba = is_bot_response["probability"]

    return Prediction(
        id=prediction_id,
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=bot_proba
    )