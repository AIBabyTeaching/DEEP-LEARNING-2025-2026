"""Analysis helpers for summarizing Keras training histories."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import tensorflow as tf

HistoryLike = tf.keras.callbacks.History | Mapping[str, list[float]]


def _best_index(values: list[float], metric: str) -> int:
    """Return the index of the best metric value depending on the metric name."""

    if not values:
        return -1
    metric_lower = metric.lower()
    if "loss" in metric_lower or "error" in metric_lower:
        return int(min(range(len(values)), key=values.__getitem__))
    return int(max(range(len(values)), key=values.__getitem__))


def summarize_history(
    history: HistoryLike,
    *,
    metrics: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Produce a compact summary of the key metrics recorded during training.

    Parameters
    ----------
    history:
        Either a :class:`tf.keras.callbacks.History` instance or a mapping with
        metric names as keys and per-epoch values as lists.
    metrics:
        Optional iterable selecting which base metrics to summarize. When not
        provided the function inspects the history keys and derives a set of
        base metrics (e.g. ``["loss", "accuracy"]``) by stripping the
        ``"val_"`` prefix.

    Returns
    -------
    list of dict
        Each dictionary contains the metric name along with the best training
        and validation values and the epochs at which they were observed.
    """

    if isinstance(history, tf.keras.callbacks.History):
        history_data: Mapping[str, list[float]] = history.history
    else:
        history_data = history

    if not history_data:
        return []

    if metrics is None:
        base_metrics = {
            key.replace("val_", "")
            for key in history_data
            if not key.startswith("lr") and history_data[key]
        }
        metrics_to_use = sorted(base_metrics)
    else:
        metrics_to_use = list(metrics)

    summaries: list[dict[str, Any]] = []
    for metric in metrics_to_use:
        train_values = list(history_data.get(metric, []))
        val_metric_name = f"val_{metric}"
        val_values = list(history_data.get(val_metric_name, []))

        train_epoch = _best_index(train_values, metric)
        val_epoch = _best_index(val_values, metric)

        summary = {
            "metric": metric,
            "train_best": train_values[train_epoch] if train_epoch >= 0 else None,
            "train_epoch": train_epoch + 1 if train_epoch >= 0 else None,
            "val_best": val_values[val_epoch] if val_epoch >= 0 else None,
            "val_epoch": val_epoch + 1 if val_epoch >= 0 else None,
        }
        summaries.append(summary)

    return summaries


__all__ = ["summarize_history"]
