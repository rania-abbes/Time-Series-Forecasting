"""Naive forecasting baselines used as a sanity check for learned models."""
import numpy as np


def last_step_baseline(inputs, nb_steps_to_predict, label_feature_id):
    """Predict the last observed value for every future time step."""
    last_step_value = inputs[:, -1, label_feature_id]
    return np.tile(last_step_value[:, np.newaxis], (1, nb_steps_to_predict))


def mean_baseline(inputs, nb_steps_to_predict, label_feature_id):
    """Predict the mean of the input window for every future time step."""
    mean_value = np.mean(inputs[:, :, label_feature_id], axis=1)
    return np.tile(mean_value[:, np.newaxis], (1, nb_steps_to_predict))


def repeat_baseline(inputs, nb_steps_to_predict, label_feature_id):
    """Repeat the last `nb_steps_to_predict` observed values (a daily-cycle baseline)."""
    assert inputs.shape[1] >= nb_steps_to_predict, "Not enough steps in input to repeat"
    return inputs[:, -nb_steps_to_predict:, label_feature_id]
