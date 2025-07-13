from uuid import uuid4

import requests
import streamlit as st
import numpy as np
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import pandas as pd

from app.models import GetMessageRequestModel
from app.api.llm import respond

def send_to_predict(text: str, participant_index: int) -> float:
    pred_resp = requests.post(
        echo_bot_url + "/predict",
        json={
            "text": text,
            "dialog_id": dialog_id,
            "id": str(uuid4()),
            "participant_index": participant_index,
        },
    ).json()
    return pred_resp["is_bot_probability"]

default_echo_bot_url = "http://localhost:6872"
st.set_page_config(initial_sidebar_state="collapsed")

if "y_true" not in st.session_state:
    st.session_state.y_true = []
if "y_pred" not in st.session_state:
    st.session_state.y_pred = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {"accuracy": [], "log_loss": []}

st.markdown("# Echo bot 🚀")
st.sidebar.markdown("# Echo bot 🚀")

if "dialog_id" not in st.session_state:
    st.session_state.dialog_id = str(uuid4())

with st.sidebar:
    if st.button("Reset"):
        st.session_state.pop("messages", None)
        st.session_state.pop("y_true", None)
        st.session_state.pop("y_pred", None)
        st.session_state.pop("metrics", None)
        st.session_state.dialog_id = str(uuid4())

    echo_bot_url = st.text_input(
        "Bot url", key="echo_bot_url", value=default_echo_bot_url, disabled=True
    )

    dialog_id = st.text_input("Dialog id", key="dialog_id", disabled=True)

    # Show chart in sidebar
    if st.session_state.metrics["accuracy"]:
        st.markdown("### 📊 Accuracy / Log-loss")

        acc = np.array(st.session_state.metrics["accuracy"])
        loss = np.array(st.session_state.metrics["log_loss"])
        ratio = acc / loss

        df_ratio = pd.DataFrame({
            "Accuracy / Log-loss": ratio,
        })
        df_ratio.index = range(1, len(df_ratio) + 1)
        df_ratio.index.name = "Dialog #"

        st.line_chart(df_ratio)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Type something"}]

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "prob" in msg:
            st.markdown(f"**Bot probability:** {msg['prob']:.2%}")

if message := st.chat_input():
    # Send this message to the /predict endpoint
    prob = send_to_predict(message, participant_index=1)

    # Dummy label: assume label is 0 for now (human). Replace with actual ground truth when available.
    true_label = 0

    # Store ground truth and prediction
    st.session_state.y_true.append(true_label)
    st.session_state.y_pred.append(prob)

    # Calculate metrics
    y_true_arr = np.array(st.session_state.y_true)
    y_pred_arr = np.array(st.session_state.y_pred)

    accuracy = np.mean((y_pred_arr >= 0.5) == y_true_arr)
    loss = log_loss(y_true_arr, y_pred_arr, labels=[0, 1])

    # Store metrics
    st.session_state.metrics["accuracy"].append(accuracy)
    st.session_state.metrics["log_loss"].append(loss)

    user_msg = {"role": "user", "content": message, "prob": prob}
    st.session_state["messages"].append(user_msg)

    bot_response = respond(message)

    # Send this message to the /predict endpoint
    prob_bot = send_to_predict(bot_response, participant_index=0)

    assistant_msg = {"role": "assistant", "content": bot_response, "prob": prob_bot}
    st.session_state["messages"].append(assistant_msg)

    st.rerun()
