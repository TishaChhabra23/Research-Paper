"""
Learning rate scheduling utilities, including cosine decay with linear warmup.
"""

import numpy as np
from typing import Callable


def cosine_decay_schedule(
    initial_lr: float,
    total_epochs: int,
    warmup_epochs: int = 3,
    min_lr: float = 1e-6
) -> Callable[[int, float], float]:
    """
    Returns a Keras-compatible LearningRateScheduler callback function.
    Implements a linear warmup followed by half-period cosine decay.

    Args:
        initial_lr: Peak learning rate after warmup.
        total_epochs: Total number of epochs for this training phase.
        warmup_epochs: Number of initial epochs for linear warmup.
        min_lr: Lower bound for the decayed learning rate.

    Returns:
        Callable[[int, float], float]: Scheduler function taking (epoch, current_lr).
    """
    def scheduler(epoch: int, lr: float) -> float:
        if epoch < warmup_epochs:
            return initial_lr * float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return float(min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + np.cos(np.pi * progress)))

    return scheduler
