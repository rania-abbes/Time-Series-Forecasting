import numpy as np
import pandas as pd

from src.evaluation import evaluate_model, mae_by_horizon
from src.scaling import Standardizer


def test_evaluate_model_perfect_predictions_give_zero_mae():
    df = pd.DataFrame({"T (degC)": [0.0, 10.0, 20.0]})
    scaler = Standardizer(target_feature="T (degC)").fit(df)

    labels = np.array([[1.0, -1.0], [0.5, 0.5]])
    mae = evaluate_model(lambda samples: samples, labels, labels, scaler)
    assert mae == 0.0


def test_evaluate_model_clip_reduces_error_from_out_of_range_predictions():
    df = pd.DataFrame({"T (degC)": [0.0, 10.0, 20.0]})  # normalized range roughly [-1.22, 1.22]
    scaler = Standardizer(target_feature="T (degC)").fit(df)

    labels = np.array([[1.0]])
    wild_prediction = np.array([[100.0]])  # way outside the train range once rescaled

    mae_clipped = evaluate_model(lambda samples: wild_prediction, labels, labels, scaler, clip=True)
    mae_unclipped = evaluate_model(lambda samples: wild_prediction, labels, labels, scaler, clip=False)
    assert mae_clipped < mae_unclipped


def test_mae_by_horizon_matches_manual_computation():
    predictions = np.array([[1.0, 2.0], [3.0, 6.0]])
    labels = np.array([[0.0, 0.0], [0.0, 0.0]])
    result = mae_by_horizon(predictions, labels)
    np.testing.assert_allclose(result, [2.0, 4.0])
