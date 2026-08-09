"""Feature scaling fit on a reference (train) split only."""


class Standardizer:
    """Z-score standardizer that also remembers the target's train-set range.

    Fitting on the train split only (never on val/test) and reusing those
    statistics everywhere downstream is what keeps normalization, clipping,
    and error metrics free of test-set leakage.
    """

    def __init__(self, target_feature):
        self.target_feature = target_feature
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None

    def fit(self, reference_df):
        self.mean_ = reference_df.mean()
        self.std_ = reference_df.std()
        self.min_ = reference_df.min()
        self.max_ = reference_df.max()
        return self

    def transform(self, df):
        return (df - self.mean_) / self.std_

    def inverse_transform_target(self, values):
        """Map normalized target values back to degrees Celsius."""
        return values * self.std_[self.target_feature] + self.mean_[self.target_feature]

    def clip_to_target_range(self, values):
        """Clip (already rescaled, degrees Celsius) predictions to the train-set min/max.

        Models can extrapolate outside the observed range on out-of-distribution
        inputs; clipping avoids letting a handful of extreme predictions dominate
        the MAE.
        """
        values = values.copy()
        values[values < self.min_[self.target_feature]] = self.min_[self.target_feature]
        values[values > self.max_[self.target_feature]] = self.max_[self.target_feature]
        return values
