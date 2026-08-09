import numpy as np
import pandas as pd

from src.scaling import Standardizer


def _reference_df():
    return pd.DataFrame({
        "T (degC)": [0.0, 10.0, 20.0],
        "p (mbar)": [1000.0, 1010.0, 1020.0],
    })


def test_transform_is_zero_mean_unit_std_on_reference_data():
    df = _reference_df()
    scaler = Standardizer(target_feature="T (degC)").fit(df)
    transformed = scaler.transform(df)
    np.testing.assert_allclose(transformed.mean(), 0.0, atol=1e-8)
    np.testing.assert_allclose(transformed.std(), 1.0, atol=1e-8)


def test_inverse_transform_target_round_trips():
    df = _reference_df()
    scaler = Standardizer(target_feature="T (degC)").fit(df)
    normalized_target = scaler.transform(df)["T (degC)"].to_numpy()
    restored = scaler.inverse_transform_target(normalized_target)
    np.testing.assert_allclose(restored, df["T (degC)"].to_numpy())


def test_clip_to_target_range_bounds_to_train_min_max():
    df = _reference_df()
    scaler = Standardizer(target_feature="T (degC)").fit(df)  # train range is [0, 20]
    predictions = np.array([-5.0, 5.0, 25.0])
    clipped = scaler.clip_to_target_range(predictions)
    np.testing.assert_array_equal(clipped, [0.0, 5.0, 20.0])


def test_clip_to_target_range_does_not_mutate_input():
    df = _reference_df()
    scaler = Standardizer(target_feature="T (degC)").fit(df)
    predictions = np.array([-5.0, 25.0])
    scaler.clip_to_target_range(predictions)
    np.testing.assert_array_equal(predictions, [-5.0, 25.0])
