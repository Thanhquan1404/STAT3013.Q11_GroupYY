# model.py

from sklearn.ensemble import RandomForestClassifier
from config import RF_PARAMS


def build_model() -> RandomForestClassifier:
    """
    Create a RandomForestClassifier with parameters from config.
    """
    model = RandomForestClassifier(**RF_PARAMS)
    return model
