import joblib
import os

# Resolve paths relative to the file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the alternate pre-trained model and vectorizer
try:
    model_path = os.path.join(BASE_DIR, "model1.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "vectorizer1.pkl")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"Model files not found. Ensure 'model1.pkl' and 'vectorizer1.pkl' are in the project directory.\n{e}"
    )

def predict_sentiment(texts):
    """
    Predict sentiment from a list of texts using model1 and vectorizer1.

    Args:
        texts (List[str]): Input text data

    Returns:
        List[str]: Predicted sentiment labels
    """
    vectors = vectorizer.transform(texts)
    return model.predict(vectors)
