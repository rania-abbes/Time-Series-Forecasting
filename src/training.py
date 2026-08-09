"""Training loop for the LSTM forecaster."""
import copy

import torch


def train_lstm(model, train_loader, val_loader, num_epochs, lr=1e-3, patience=None, min_delta=1e-4, verbose=True):
    """Train `model` with Adam/MSE and return per-epoch train/val loss history.

    If `patience` is set, training stops once `patience` epochs pass without a
    validation loss improvement greater than `min_delta` (a strict "<" check
    would never trigger on a smoothly, infinitesimally decreasing loss), and
    `model`'s weights are restored to the best epoch's (rather than the last
    epoch's) before returning. `history` always covers every epoch actually
    run, so a plateau (or its absence) is visible in the returned curve
    regardless of whether early stopping fired.
    """
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose:
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if val_loss < best_val_loss:
                # still the best-seen checkpoint, just not by more than min_delta
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())

        if patience is not None and epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1} "
                      f"(no val improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history
