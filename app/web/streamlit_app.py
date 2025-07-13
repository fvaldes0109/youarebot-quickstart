from uuid import uuid4

import requests
import streamlit as st

from app.models import GetMessageRequestModel

default_echo_bot_url = "http://localhost:6872"
st.set_page_config(initial_sidebar_state="collapsed")

st.markdown("# Echo bot 🚀")
st.sidebar.markdown("# Echo bot 🚀")

if "dialog_id" not in st.session_state:
    st.session_state.dialog_id = str(uuid4())

with st.sidebar:
    if st.button("Reset"):
        st.session_state.pop("messages", None)
        st.session_state.dialog_id = str(uuid4())

    echo_bot_url = st.text_input(
        "Bot url", key="echo_bot_url", value=default_echo_bot_url, disabled=True
    )

    dialog_id = st.text_input("Dialog id", key="dialog_id", disabled=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Type something"}]

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "user" and "prob" in msg:
            st.markdown(f"**Bot probability:** {msg['prob']:.2%}")

if message := st.chat_input():
    # Send this message to the /predict endpoint
    pred_resp = requests.post(
        echo_bot_url + "/predict",
        json={
            "text": message,
            "dialog_id": dialog_id,
            "id": str(uuid4()),
            "participant_index": 1,
        },
    ).json()
    prob = pred_resp["is_bot_probability"]

    user_msg = {"role": "user", "content": message, "prob": prob}
    st.session_state["messages"].append(user_msg)

    response = requests.post(
        echo_bot_url + "/get_message",
        json=GetMessageRequestModel(
            dialog_id=dialog_id, last_msg_text=message, last_message_id=uuid4()
        ).model_dump(),
    )
    json_response = response.json()
    assistant_msg = {"role": "assistant", "content": json_response['new_msg_text']}
    st.session_state["messages"].append(assistant_msg)

    st.rerun()
