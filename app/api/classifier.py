from transformers import pipeline

zero_shot = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
labels = ["bot", "human"]

def predict_bot(text: str) -> float:

    result = zero_shot(text, candidate_labels=labels)

    bot_label_index = result["labels"].index("bot")
    bot_probability = result["scores"][bot_label_index]

    return bot_probability
