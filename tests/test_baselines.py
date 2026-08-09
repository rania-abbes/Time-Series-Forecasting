import numpy as np
import pytest

from src.baselines import last_step_baseline, mean_baseline, repeat_baseline

# inputs: 2 samples, 4 time steps, 2 features; feature 0 is the target
INPUTS = np.array([
    [[1.0, 9], [2.0, 9], [3.0, 9], [4.0, 9]],
    [[10.0, 9], [20.0, 9], [30.0, 9], [40.0, 9]],
])


def test_last_step_baseline_repeats_the_final_observation():
    predictions = last_step_baseline(INPUTS, nb_steps_to_predict=3, label_feature_id=0)
    assert predictions.shape == (2, 3)
    np.testing.assert_array_equal(predictions, [[4.0, 4.0, 4.0], [40.0, 40.0, 40.0]])


def test_mean_baseline_repeats_the_window_mean():
    predictions = mean_baseline(INPUTS, nb_steps_to_predict=2, label_feature_id=0)
    assert predictions.shape == (2, 2)
    np.testing.assert_array_equal(predictions, [[2.5, 2.5], [25.0, 25.0]])


def test_repeat_baseline_reuses_the_tail_of_the_input():
    predictions = repeat_baseline(INPUTS, nb_steps_to_predict=2, label_feature_id=0)
    assert predictions.shape == (2, 2)
    np.testing.assert_array_equal(predictions, [[3.0, 4.0], [30.0, 40.0]])


def test_repeat_baseline_asserts_when_horizon_exceeds_input_length():
    with pytest.raises(AssertionError, match="Not enough steps"):
        repeat_baseline(INPUTS, nb_steps_to_predict=10, label_feature_id=0)
