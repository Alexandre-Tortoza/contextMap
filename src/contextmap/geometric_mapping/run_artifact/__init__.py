"""Persisted, immutable, self-describing Geometric Mapping run artifacts.

A ``GeometricMapArtifact`` is a directory holding the map one run built from a
selected segment of a canonical sequence and a State Estimation trajectory, with
the lineage, configuration, metrics and debug evidence needed to trust it. It
opens without ROS, an estimator, or a model library, and a
:class:`~contextmap.geometric_mapping.GeometryReference` resolves after the run is
closed and reopened to the same authoritative XYZ and provenance. See
``src/contextmap/geometric_mapping/docs/artifact.md`` for the layout.

Contractual data (``outputs/``, ``metrics/``, ``lineage.json``, ``config.json`` and
``environment.json``) is inventoried with size and SHA-256 in the manifest;
``__DEBUG__/`` is human evidence that is never inventoried, so removing it cannot
invalidate the run and no downstream stage may depend on it. Writing follows
:class:`~contextmap.shared.AtomicRunDirectory`: an interrupted write never looks
like a finished run and a finished run is never modified.
"""

from .manifest import (
    SCHEMA_VERSION,
    GeometricMapArtifactManifest,
    GeometricMapRunId,
    IncompleteMapArtifactError,
    MapArtifactError,
    MapDebugLevel,
)
from .reader import GeometricMapArtifactReader
from .writer import (
    GeometricMapArtifactWriter,
    encode_mapping_configuration,
    mapping_configuration_fingerprint,
)

__all__ = [
    "SCHEMA_VERSION",
    "GeometricMapArtifactManifest",
    "GeometricMapArtifactReader",
    "GeometricMapArtifactWriter",
    "GeometricMapRunId",
    "IncompleteMapArtifactError",
    "MapArtifactError",
    "MapDebugLevel",
    "encode_mapping_configuration",
    "mapping_configuration_fingerprint",
]
