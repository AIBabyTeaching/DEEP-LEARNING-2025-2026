"""Data-loading helpers built on top of TensorFlow Datasets and `tf.data`."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from typing import Any

import tensorflow as tf
import tensorflow_datasets as tfds
import zipfile


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_kaggle_competition(
    name: str,
    *,
    target_dir: str | Path,
    force: bool = False,
    unzip: bool = True,
) -> Path:
    """Download competition files from Kaggle using the authenticated API.

    Parameters
    ----------
    name:
        Slug of the Kaggle competition, e.g. ``"dogs-vs-cats"``.
    target_dir:
        Destination directory where archives will be stored.
    force:
        When ``True`` an existing archive will be replaced by a fresh download.
    unzip:
        If enabled, extract the downloaded ``.zip`` file into ``target_dir``.

    Returns
    -------
    pathlib.Path
        Path to the directory that contains the downloaded files.
    """

    target_path = _ensure_directory(Path(target_dir))
    archive_path = target_path / f"{name}.zip"

    def _download_archive() -> None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(name, path=target_path, quiet=False)

    if force and archive_path.exists():
        archive_path.unlink()

    if not archive_path.exists():
        _download_archive()

    if unzip and archive_path.exists():

        def _extract_archive() -> None:
            if not zipfile.is_zipfile(archive_path):
                raise zipfile.BadZipFile(f"Downloaded file {archive_path} is not a valid ZIP archive")
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(target_path)

        try:
            _extract_archive()
        except zipfile.BadZipFile as exc:
            # The Kaggle CLI occasionally writes HTML error responses to disk when authentication
            # fails or the download is interrupted. Retry once with a fresh download to avoid
            # leaving the user with a confusing ``BadZipFile`` error.
            archive_path.unlink(missing_ok=True)

            if force:
                raise RuntimeError(
                    "The downloaded competition archive appears to be corrupted even with "
                    "`force=True`. Remove the archive manually and verify Kaggle credentials."
                ) from exc

            _download_archive()
            try:
                _extract_archive()
            except zipfile.BadZipFile as exc2:
                raise RuntimeError(
                    "Failed to extract Kaggle competition files because the downloaded archive "
                    "is not a valid ZIP file. Try deleting the archive and downloading again "
                    "after verifying Kaggle API access."
                ) from exc2

    return target_path


def load_tfds_dataset( # download from TF datasets and return it.
    name: str,
    *,
    split: str = "train",
    data_dir: str | None = None,
    as_supervised: bool = True,
    shuffle_files: bool = True,
    with_info: bool = False,
    try_gcs: bool = True,
) -> Any:
    """Download and load a TensorFlow Datasets dataset.

    Parameters
    ----------
    name:
        Dataset identifier, e.g. ``"mnist"`` or ``"imdb_reviews"``.
    split:
        Split spec passed directly to ``tfds.load``.
    data_dir:
        Optional directory for cached downloads.
    as_supervised:
        Return ``(features, label)`` pairs if supported.
    shuffle_files:
        Shuffle input files before reading.
    with_info:
        When ``True`` the :class:`tfds.core.DatasetInfo` is returned alongside the dataset.
    try_gcs:
        Allow TFDS to use public GCS mirrors when available.
    """

    ds, info = tfds.load(
        name, # Dataset identifier (In our case "fashion_mnist")
        split=split, # Train/Test
        data_dir=data_dir, # Directory of the data to be cached in
        as_supervised=as_supervised,
        shuffle_files=shuffle_files,
        with_info=True,
        try_gcs=try_gcs,
    )
    return (ds, info) if with_info else ds


def prepare_for_training(
    dataset: tf.data.Dataset,
    *,
    batch_size: int = 32, # How frequent the model weights are updated.
    shuffle_buffer: int | None = 1000,
    cache: bool = True,
    augment_fn: Callable[[Any], Any] | None = None,
    prefetch: bool = True,
) -> tf.data.Dataset:
    """Apply standard ``tf.data`` transformations for training workflows."""

    ds = dataset
    if cache:
        ds = ds.cache()
    if shuffle_buffer:
        ds = ds.shuffle(int(shuffle_buffer))
    if augment_fn is not None:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
