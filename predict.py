from utils.model_utils import load_pickle
import numpy as np

MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

model = load_pickle(MODEL_PATH)
vectorizer = load_pickle(VECTORIZER_PATH)

def predict_sentiment(text):
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    pred = np.argmax(proba)

    sentiment = "Positive 😊" if pred == 1 else "Negative 😞"
    confidence = f"{proba[pred] * 100:.2f}%"

    return sentiment, confidence

if __name__ == "__main__":
    while True:
        text = input("\nEnter a sentence (or 'quit'): ")
        if text.lower() == "quit":
            break

        sentiment, confidence = predict_sentiment(text)
        print(f"Sentiment: {sentiment}  |  Confidence: {confidence}")
