"""API pública de Sensor Association: evidência visual 2D ancorada na geometria persistente.

Expõe contratos, políticas e execução; os passos científicos permanecem internos.
Observações referenciam geometria, features e claims sem consolidar entidades ou crenças.
"""

from contextmap.sensor_association.artifacts import (
    IncompleteRunArtifactError,
    RunArtifactError,
    SensorAssociationDebugLevel,
    SensorAssociationRunId,
    SensorAssociationRunManifest,
    SensorAssociationRunReader,
    SensorAssociationRunWriter,
)
from contextmap.sensor_association.camera import (
    CameraIdentity,
    CameraProjection,
    PixelProjection,
    camera_projection_for,
)
from contextmap.sensor_association.dense_sampling import InterpolationPolicy
from contextmap.sensor_association.diagnostics import (
    DiagnosticTolerances,
    TrustedCorrespondences,
)
from contextmap.sensor_association.errors import AssociationInputError
from contextmap.sensor_association.membership import BoxDepthComponentPolicy
from contextmap.sensor_association.models import (
    AssociationProvenance,
    CalibrationRef,
    DepthMetric,
    FootprintKind,
    PixelCoordinate,
    PointCorrespondence,
    PoseRef,
    ProjectionSummary,
    SpatialObservation,
    SpatialObservationId,
    VisibilityDiagnostics,
    VisibilityState,
    spatial_observation_id_for,
)
from contextmap.sensor_association.projection_chain.candidate_geometry import (
    CandidateGeometryPolicy,
)
from contextmap.sensor_association.projection_chain.visibility import OcclusionPolicy
from contextmap.sensor_association.quality import (
    ObservationQuality,
    QualityComponent,
    ReprojectionStatistics,
    ValueSummary,
)
from contextmap.sensor_association.service import (
    AssociationFrameInput,
    DenseChannel,
    FrameAssociation,
    FrameSink,
    SensorAssociationOutcome,
    SensorAssociationRequest,
    SensorAssociationService,
)

__all__ = [
    "AssociationFrameInput",
    "AssociationInputError",
    "AssociationProvenance",
    "BoxDepthComponentPolicy",
    "CalibrationRef",
    "CameraIdentity",
    "CameraProjection",
    "CandidateGeometryPolicy",
    "DenseChannel",
    "DepthMetric",
    "DiagnosticTolerances",
    "FootprintKind",
    "FrameAssociation",
    "FrameSink",
    "IncompleteRunArtifactError",
    "InterpolationPolicy",
    "ObservationQuality",
    "OcclusionPolicy",
    "PixelCoordinate",
    "PixelProjection",
    "PointCorrespondence",
    "PoseRef",
    "ProjectionSummary",
    "QualityComponent",
    "ReprojectionStatistics",
    "RunArtifactError",
    "SensorAssociationDebugLevel",
    "SensorAssociationOutcome",
    "SensorAssociationRequest",
    "SensorAssociationRunId",
    "SensorAssociationRunManifest",
    "SensorAssociationRunReader",
    "SensorAssociationRunWriter",
    "SensorAssociationService",
    "SpatialObservation",
    "SpatialObservationId",
    "TrustedCorrespondences",
    "ValueSummary",
    "VisibilityDiagnostics",
    "VisibilityState",
    "camera_projection_for",
    "spatial_observation_id_for",
]
