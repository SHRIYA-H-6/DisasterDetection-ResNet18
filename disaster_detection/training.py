"""Training utilities shared by the classifier and segmentation trainers."""
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TrainingMonitor:
    """Early stopping on validation loss plus explicit overfitting detection.

    Stops when either
      * validation loss has not improved for `patience` epochs (early stopping), or
      * validation loss rose for `window` consecutive epochs while training loss kept
        falling over the same epochs (overfitting), excluding windows that span a change
        in the set of trainable layers.
    """

    def __init__(self, patience=7, window=3, min_delta=1e-4):
        self.patience, self.window, self.min_delta = patience, window, min_delta
        self.best_loss, self.best_epoch, self.bad_epochs = float("inf"), 0, 0
        self.train_losses, self.val_losses, self.phase_change_epochs = [], [], set()
        self.overfitting_detected = False
        self.stop_reason = ""

    def mark_phase_change(self, epoch):
        self.phase_change_epochs.add(epoch)

    def update(self, epoch, train_loss, val_loss):
        """Record an epoch. Returns (improved, stop)."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        improved = val_loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss, self.best_epoch, self.bad_epochs = val_loss, epoch, 0
        else:
            self.bad_epochs += 1

        w = self.window
        if len(self.val_losses) > w:
            v, t = self.val_losses[-w - 1:], self.train_losses[-w - 1:]
            val_rising = all(b > a for a, b in zip(v, v[1:]))
            train_falling = all(b < a for a, b in zip(t, t[1:]))
            spans_phase_change = any(e in self.phase_change_epochs for e in range(epoch - w + 1, epoch + 1))
            if val_rising and train_falling and not spans_phase_change:
                self.overfitting_detected = True
                self.stop_reason = (
                    f"OVERFITTING DETECTED at epoch {epoch}: validation loss rose {v[0]:.4f} -> {v[-1]:.4f} "
                    f"over the last {w} epochs while training loss fell {t[0]:.4f} -> {t[-1]:.4f}. "
                    f"Stopping and restoring the best weights from epoch {self.best_epoch}.")
                return improved, True
        if self.bad_epochs >= self.patience:
            self.stop_reason = (f"Early stopping at epoch {epoch}: no validation-loss improvement for "
                                f"{self.patience} epochs (best {self.best_loss:.4f} at epoch {self.best_epoch}).")
            return improved, True
        return improved, False
