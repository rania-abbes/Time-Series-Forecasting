"""Sliding-window extraction for time series forecasting."""
import numpy as np
import pandas as pd


def have_regular_time_steps(time_series):
    """Return True if every consecutive pair of index timestamps is equally spaced."""
    indexes = pd.Series(time_series.index)
    time_steps_lengths = (indexes - indexes.shift()).value_counts()
    return len(time_steps_lengths) == 1


def extract_windows(data, samples_window_size, labels_window_size, window_step, label_feature_id):
    """Slide a window over `data` and split each window into (samples, labels).

    Windows whose timestamps aren't regularly spaced (e.g. spanning a gap in
    the data) are dropped rather than filled, since there is nothing sound to
    fill them with.
    """
    window_size = samples_window_size + labels_window_size

    windows = [
        data[time_step:time_step + window_size]
        for time_step in range(0, len(data) - window_size + 1, window_step)
        if have_regular_time_steps(data[time_step:time_step + window_size])
    ]
    windows = np.array(windows)

    samples = windows[:, :samples_window_size, :]
    labels = windows[:, samples_window_size:, label_feature_id]

    return samples, labels
