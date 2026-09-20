"""Visualization utilities for Keras model development."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import plot_model

HistoryLike = tf.keras.callbacks.History | dict[str, list[float]]


def plot_history(
    history: HistoryLike,
    *,
    metrics: Iterable[str] | None = None,
    title: str | None = None,
) -> None:
    """Plot training and validation metrics from a History object.

    Parameters
    ----------
    history:
        The training history returned by ``Model.fit`` or a mapping of metrics.
    metrics:
        Iterable of metric names to visualize. When ``None`` all metrics present in
        the history are plotted.
    title:
        Optional string used as the title template for each plot. When provided the
        ``metric`` name is injected via ``str.format`` using the ``metric`` keyword
        (e.g. ``"My experiment - {metric}"``).
    """

    if isinstance(history, tf.keras.callbacks.History):
        history_data = history.history
    else:
        history_data = history

    if not history_data:
        raise ValueError("History is empty; nothing to plot.")

    metrics_to_plot = list(metrics or {k.replace("val_", "") for k in history_data})

    for metric in metrics_to_plot:
        train_values = history_data.get(metric)
        val_values = history_data.get(f"val_{metric}")

        if train_values is None and val_values is None:
            continue

        epochs = range(1, len(train_values or val_values) + 1)
        plt.figure()
        if train_values is not None:
            plt.plot(epochs, train_values, label=f"Train {metric}")
        if val_values is not None:
            plt.plot(epochs, val_values, label=f"Val {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        if title is not None:
            plot_title = title.format(metric=metric)
        else:
            plot_title = f"Training history: {metric}"
        plt.title(plot_title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()


def visualize_model(
    model: tf.keras.Model,
    *,
    filename: str | Path = "model.png",
    show_shapes: bool = True,
    show_layer_names: bool = True,
) -> Path:
    """Render the model architecture graph to an image file."""

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_model(
        model, to_file=str(output_path), show_shapes=show_shapes, show_layer_names=show_layer_names
    )
    return output_path
