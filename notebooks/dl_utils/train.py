"""Reusable training helpers for TensorFlow/Keras models."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

import tensorflow as tf

CallbackIterable = Iterable[tf.keras.callbacks.Callback]


def build_callbacks(
    *,
    log_root: str | Path = "runs",
    experiment_name: str | None = None,
    patience: int = 5,
    monitor: str = "val_loss",
    mode: str = "auto",
    checkpoint_path: str | Path | None = None,
    tensorboard: bool = True,
) -> tuple[list[tf.keras.callbacks.Callback], Path]:
    """Create a standard callback list for course notebooks."""

    log_root_path = Path(log_root)
    log_root_path.mkdir(parents=True, exist_ok=True)

    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = log_root_path / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[tf.keras.callbacks.Callback] = []

    if tensorboard:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            restore_best_weights=True,
        )
    )

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=monitor,
                mode=mode,
                save_best_only=True,
            )
        )

    return callbacks, log_dir


def compile_and_fit(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    *,
    optimizer: tf.keras.optimizers.Optimizer,
    loss: str | tf.keras.losses.Loss,
    metrics: Sequence[str | tf.keras.metrics.Metric] | None = None,
    epochs: int = 10,
    validation_ds: tf.data.Dataset | None = None,
    callbacks: CallbackIterable | None = None,
    **fit_kwargs,
) -> tf.keras.callbacks.History:
    """Compile the model with the supplied configuration and run ``model.fit``."""

    model.compile(optimizer=optimizer, loss=loss, metrics=list(metrics or []))

    history = model.fit(
        train_ds,
        epochs=epochs, 
        validation_data=validation_ds,
        callbacks=list(callbacks or []),
        **fit_kwargs,
    )
    return history
