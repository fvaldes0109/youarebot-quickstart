This runs the 4 main services: mlflow, classifier, llm and orchestrator.

llm requires an OPENROUTER_API_KEY to be set in the .env file.

```bash
docker compose build && docker compose up -d
```

There are other two services: trainer, which is used to finetune a small classification model and register it to the mlflow service, and web, which is a simple streamlit app that allows to chat with the bot and test all the services integrated.

For both of them install the requirements how you prefer. To run trainer just run the jupyter notebook lora.ipynb. To run the streamlit app:

```bash
streamlit run streamlit_app.py
```
