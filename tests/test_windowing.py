import numpy as np
import pandas as pd
import pytest

from src.windowing import extract_windows, have_regular_time_steps


def _hourly_index(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="h")


def test_have_regular_time_steps_true_for_hourly_index():
    df = pd.DataFrame({"a": range(10)}, index=_hourly_index(10))
    assert have_regular_time_steps(df)


def test_have_regular_time_steps_false_with_gap():
    index = _hourly_index(10).delete(5)  # drop one hour -> irregular gap
    df = pd.DataFrame({"a": range(9)}, index=index)
    assert not have_regular_time_steps(df)


def test_have_regular_time_steps_false_for_single_row():
    df = pd.DataFrame({"a": [1]}, index=_hourly_index(1))
    assert not have_regular_time_steps(df)


def test_extract_windows_shapes_no_gaps():
    n = 50
    df = pd.DataFrame({"a": np.arange(n), "b": np.arange(n) * 2.0}, index=_hourly_index(n))
    samples, labels = extract_windows(
        df, samples_window_size=10, labels_window_size=5, window_step=1, label_feature_id=0
    )
    nb_windows = n - (10 + 5) + 1
    assert samples.shape == (nb_windows, 10, 2)
    assert labels.shape == (nb_windows, 5)


def test_extract_windows_drops_windows_spanning_a_gap():
    n = 30
    index = _hourly_index(n).delete(15)  # gap right in the middle
    df = pd.DataFrame({"a": np.arange(n - 1)}, index=index)
    samples, _ = extract_windows(
        df, samples_window_size=10, labels_window_size=5, window_step=1, label_feature_id=0
    )
    # every window whose 15-step span crosses index 15 must be excluded
    assert samples.shape[0] < (len(df) - 15 + 1)


def test_extract_windows_selects_requested_label_feature():
    n = 20
    df = pd.DataFrame({"a": np.zeros(n), "b": np.arange(n) * 1.0}, index=_hourly_index(n))
    _, labels = extract_windows(
        df, samples_window_size=5, labels_window_size=3, window_step=1, label_feature_id=1
    )
    # label_feature_id=1 selects column "b", which is a strictly increasing ramp
    assert np.all(np.diff(labels, axis=1) > 0)
