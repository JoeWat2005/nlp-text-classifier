import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from utils.preprocessing import load_dataset
from utils.model_utils import create_vectorizer, create_model, save_pickle
from utils.evaluation import plot_confusion_matrix, plot_roc

TEXT_PATH = "data/data.csv"
MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

print("Loading dataset…")
texts, labels = load_dataset(TEXT_PATH)

print("Vectorizing…")
vectorizer = create_vectorizer()
X = vectorizer.fit_transform(texts)

print("Splitting…")
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

print("Training model…")
model = create_model()
model.fit(X_train, y_train)

print("Evaluating…")
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))

plot_confusion_matrix(y_test, preds, save_path="models/confusion_matrix.png")
plot_roc(y_test, probs, save_path="models/roc_curve.png")

print("Saving model and vectorizer…")
os.makedirs("models", exist_ok=True)
save_pickle(model, MODEL_PATH)
save_pickle(vectorizer, VECTORIZER_PATH)

print("\nTraining complete.")
