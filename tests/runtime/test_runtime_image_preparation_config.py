"""Parsing of the ``[image_preparation]`` configuration section (per-camera masks)."""

from __future__ import annotations

from dataclasses import MISSING, fields
from pathlib import Path

import pytest

from contextmap.runtime import resolve_effective_config
from contextmap.runtime.configuration import (
    CameraPreparationConfig,
    ConfigurationError,
    RuntimeConfig,
)

_CAMERA = "camera_1_image_raw-calib"


def _write(tmp_path: Path, body: str) -> str:
    path = tmp_path / "prep.toml"
    path.write_text('schema_version = "0.1.0"\n' + body)
    return str(path)


def test_defaults_to_empty_mapping() -> None:
    config = resolve_effective_config().config
    assert config.image_preparation == {}


def test_the_empty_default_comes_from_a_factory() -> None:
    """``dataclasses`` refuses a ``mappingproxy`` field default before Python 3.12.

    The project supports Python 3.11, where such a default makes the class definition
    raise ``ValueError: mutable default ... use default_factory`` while the module is
    imported, so the whole runtime becomes unimportable. Only a ``default_factory``
    loads on every supported version.
    """
    field = {entry.name: entry for entry in fields(RuntimeConfig)}["image_preparation"]

    assert field.default is MISSING
    assert field.default_factory is not MISSING
    assert field.default_factory() == {}


def test_parses_valid_region_and_named_exclusions(tmp_path: Path) -> None:
    file = _write(
        tmp_path,
        f'[image_preparation."{_CAMERA}"]\n'
        'valid_region = "assets/corridor-02/valid.npy"\n'
        'exclusions.ego = "assets/corridor-02/ego.npy"\n',
    )

    config = resolve_effective_config(files=[file]).config

    prep = config.image_preparation[_CAMERA]
    assert isinstance(prep, CameraPreparationConfig)
    assert prep.valid_region == "assets/corridor-02/valid.npy"
    assert prep.exclusions == {"ego": "assets/corridor-02/ego.npy"}


def test_valid_region_is_optional(tmp_path: Path) -> None:
    file = _write(
        tmp_path,
        f'[image_preparation."{_CAMERA}"]\nexclusions.ego = "assets/corridor-02/ego.npy"\n',
    )

    prep = resolve_effective_config(files=[file]).config.image_preparation[_CAMERA]

    assert prep.valid_region is None
    assert prep.exclusions == {"ego": "assets/corridor-02/ego.npy"}


def test_round_trips_through_to_document(tmp_path: Path) -> None:
    file = _write(
        tmp_path,
        f'[image_preparation."{_CAMERA}"]\nvalid_region = "v.npy"\nexclusions.ego = "e.npy"\n',
    )
    config = resolve_effective_config(files=[file]).config

    document = config.to_document()

    assert document["image_preparation"][_CAMERA]["valid_region"] == "v.npy"
    assert document["image_preparation"][_CAMERA]["exclusions"] == {"ego": "e.npy"}


def test_mask_path_participates_in_the_digest(tmp_path: Path) -> None:
    def digest(ego: str) -> str:
        path = tmp_path / f"{ego}.toml"
        path.write_text(
            f'schema_version = "0.1.0"\n[image_preparation."{_CAMERA}"]\nexclusions.ego = "{ego}"\n'
        )
        return resolve_effective_config(files=[str(path)]).digest

    assert digest("a.npy") != digest("b.npy")


def test_rejects_non_string_mask_path(tmp_path: Path) -> None:
    file = _write(tmp_path, f'[image_preparation."{_CAMERA}"]\nvalid_region = 3\n')
    with pytest.raises(ConfigurationError):
        resolve_effective_config(files=[file])
