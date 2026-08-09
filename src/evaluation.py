"""Evaluation helpers: rescaling/clipping to degrees, MAE, and per-horizon error."""
import numpy as np
from sklearn.metrics import mean_absolute_error


def evaluate_model(predict_fn, samples, labels, scaler, clip=True):
    """Run `predict_fn` on `samples`, rescale to degrees, and compute MAE against `labels`.

    `samples` and `labels` are expected in normalized scale; `scaler` must be
    a Standardizer already fit on the training split.
    """
    predictions = predict_fn(samples)
    predictions = scaler.inverse_transform_target(predictions)
    if clip:
        predictions = scaler.clip_to_target_range(predictions)
    labels = scaler.inverse_transform_target(labels)
    return mean_absolute_error(labels, predictions)


def mae_by_horizon(predictions, labels):
    """Per-forecast-step MAE (both args already in degrees), e.g. how error grows from t+1 to t+H."""
    return np.mean(np.abs(predictions - labels), axis=0)
