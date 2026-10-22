
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_model(X, y):
    """Train a baseline crop-stress classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X, y)
    return model


def predict_stress(model, X):
    """Predict classes and stress probabilities."""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    return predictions, probabilities
