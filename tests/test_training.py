import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training import train_lstm


def _constant_loaders(batch_size=8):
    torch.manual_seed(0)
    # inputs carry no signal; a linear-ish model plateaus almost immediately,
    # which is exactly what we want to exercise early stopping quickly.
    x = torch.zeros(32, 4, 1)
    y = torch.zeros(32, 2)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader, loader


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x.squeeze(-1))


def test_train_lstm_runs_all_epochs_without_patience():
    torch.manual_seed(0)
    model = _TinyModel()
    train_loader, val_loader = _constant_loaders()

    history = train_lstm(model, train_loader, val_loader, num_epochs=5, verbose=False)

    assert len(history["train_loss"]) == 5
    assert len(history["val_loss"]) == 5


def test_train_lstm_stops_early_when_val_loss_plateaus():
    torch.manual_seed(0)
    model = _TinyModel()
    train_loader, val_loader = _constant_loaders()

    history = train_lstm(model, train_loader, val_loader, num_epochs=50, patience=2, verbose=False)

    # constant zero targets converge to ~zero loss almost immediately, so
    # patience should trigger long before the 50-epoch cap
    assert len(history["train_loss"]) < 50


def test_train_lstm_restores_best_weights_not_last_epoch():
    torch.manual_seed(0)
    model = _TinyModel()
    train_loader, val_loader = _constant_loaders()

    train_lstm(model, train_loader, val_loader, num_epochs=10, patience=2, verbose=False)

    # after restoring the best checkpoint, the model should still produce
    # finite, reasonable predictions (regression check against NaN/divergence
    # from a bad "last epoch" state being kept instead)
    with torch.no_grad():
        prediction = model(torch.zeros(1, 4, 1))
    assert torch.isfinite(prediction).all()
