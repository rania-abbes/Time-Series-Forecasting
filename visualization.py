import matplotlib.pyplot as plt
import numpy as np

def plot_window(samples,
                labels=None,
                predictions=None,
                nb_samples_to_plot=4,
                random_samples=False,
                seed=0):
    """
    Plot the samples, labels, and predictions for a given window.
    Labels and predictions are superposed and optional.
    We suppose that samples, labels, and predictions only have one feature.

    Parameters
    ----------
    samples: Tensor
        The weather data used for predictions.
        It corresponds to the previous steps.
    labels: Optional[Tensor]
        The true values.
    predictions: Optional[Tensor]
        The predicted values.
    nb_samples_to_plot: int
        The number of samples to plot.
    random_samples: bool
        If True, pick `nb_samples_to_plot` random rows (reproducibly, via `seed`)
        instead of always the first ones.
    seed: int
        Seed used to pick rows when `random_samples` is True.
    """
    # select which rows to plot
    nb_available = samples.shape[0]
    nb_samples_to_plot = min(nb_samples_to_plot, nb_available)
    if random_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(nb_available, size=nb_samples_to_plot, replace=False)
    else:
        indices = np.arange(nb_samples_to_plot)

    samples = samples[indices, :]
    if labels is not None:
        labels = labels[indices, :]
    if predictions is not None:
        predictions = predictions[indices, :]

    pred_timesteps = 0 if predictions is None else predictions.shape[1]
    labels_timesteps = 0 if labels is None else labels.shape[1]

    # initiate figure
    fig, axes = plt.subplots(nrows=samples.shape[0], ncols=1, figsize=(10, 2 * samples.shape[0]))

    # plot samples
    for i, sample in enumerate(samples):
        ax = axes[i]
        ax.plot(range(-samples.shape[1], 0), sample, label="samples", color="b")

        # plot labels
        if labels is not None:
            ax.plot(range(0, labels.shape[1]), labels[i], label="labels", color="g")

        # plot predictions
        if predictions is not None:
            ax.plot(range(0, predictions.shape[1]), predictions[i], label="predictions", color="r")

        ax.set_xlabel("Time steps (hours)")
        ax.set_xticks(range(-samples.shape[1], max(pred_timesteps, labels_timesteps), 24))
        ax.legend()

    plt.tight_layout()
    plt.show()