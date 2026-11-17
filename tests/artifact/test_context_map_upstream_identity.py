"""Upstream manifests publish exactly one explicit, nonempty string identity."""

import json
from pathlib import Path
from typing import Any

import pytest

from contextmap.artifact import artifact_digest
from contextmap.artifact.serialization.errors import UpstreamArtifactError
from contextmap.artifact.serialization.manifest import run_artifact_digest


def _manifest(directory: Path, identities: dict[str, Any]) -> None:
    """Write the minimal inventory manifest consumed by artifact_digest."""
    record = {"schema_version": "0.1.0", "file_inventory": [], **identities}
    (directory / "manifest.json").write_text(json.dumps(record), encoding="utf-8")


@pytest.mark.parametrize("field", ["run_id", "artifact_id"])
def test_one_identity_key_preserves_the_owner_digest(tmp_path: Path, field: str) -> None:
    _manifest(tmp_path, {field: "upstream-1"})

    assert artifact_digest(tmp_path) == run_artifact_digest("upstream-1", "0.1.0", ())


@pytest.mark.parametrize("artifact_id", [None, 7])
def test_a_second_identity_key_is_refused_even_when_its_value_is_not_a_string(
    tmp_path: Path, artifact_id: Any
) -> None:
    _manifest(tmp_path, {"run_id": "run-1", "artifact_id": artifact_id})

    with pytest.raises(UpstreamArtifactError, match="exactly one"):
        artifact_digest(tmp_path)


@pytest.mark.parametrize("field", ["run_id", "artifact_id"])
@pytest.mark.parametrize("identity", [None, 7, "", " \t\n"])
def test_a_single_identity_must_be_a_nonempty_string(
    tmp_path: Path, field: str, identity: Any
) -> None:
    _manifest(tmp_path, {field: identity})

    with pytest.raises(UpstreamArtifactError):
        artifact_digest(tmp_path)
