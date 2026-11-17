"""Every dense adapter hands the sink the contractual geometry of the map it computed (#651).

The sink a feature extractor receives carries payloads and, next to them, the
``FeatureExtractionDiagnostic`` of each produced dense feature, whose ``DenseFeatureDiagnostic`` is
the ``DenseFeatureSampling`` the adapter itself computed. Given to a ``VisualEvidenceRunWriter``, it
becomes ``metrics/feature-extraction.jsonl`` and the reader rebuilds the same ``DenseFeatureMap``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    DenseFeatureDiagnostic,
    FeatureEventStatus,
    FeatureExtractionDiagnostic,
    PreparedImage,
    VisualEvidenceResult,
    VisualEvidenceRunId,
    VisualEvidenceRunReader,
    VisualEvidenceRunWriter,
)
from contextmap.visual_perception.visual_features import FeatureScope, VisualFeature
from contextmap.visual_perception.visual_features.adapters.dinov2 import (
    DinoV2Config,
    DinoV2DenseFeatureBackend,
    DinoV2NativeOutput,
)
from contextmap.visual_perception.visual_features.adapters.dinov3 import (
    DinoV3Config,
    DinoV3DenseFeatureBackend,
    DinoV3NativeOutput,
)
from contextmap.visual_perception.visual_features.adapters.siglip2 import (
    Siglip2Config,
    Siglip2FeatureBackend,
    Siglip2NativeOutput,
)

_REVISION = "c" * 40
_RUN = VisualEvidenceRunId("run-0001")
_STAGE = "dense_feature_extraction"


class _Sink:
    """Records what an adapter hands its sink, payloads and diagnostics apart."""

    def __init__(self) -> None:
        self.payloads: list[tuple[VisualFeature, SourceObservationId, Any]] = []
        self.diagnostics: list[FeatureExtractionDiagnostic] = []

    def add_feature_payload(
        self, feature: VisualFeature, source_observation_id: SourceObservationId, array: Any
    ) -> None:
        self.payloads.append((feature, source_observation_id, array))

    def add_feature_diagnostic(self, diagnostic: FeatureExtractionDiagnostic) -> None:
        self.diagnostics.append(diagnostic)


class _Runtime:
    def __init__(self, output: Any) -> None:
        self._output = output

    def infer(self, image: PreparedImage) -> Any:
        return self._output


def _grid(height: int, width: int) -> Any:
    return np.arange(height * width * 3, dtype=np.float32).reshape(height, width, 3) + 1.0


def _dinov2(sink: Any) -> DinoV2DenseFeatureBackend:
    return DinoV2DenseFeatureBackend(
        config=DinoV2Config(
            checkpoint="facebook/dinov2-small", revision=_REVISION, input_width=6, input_height=4
        ),
        run_id=_RUN,
        feature_stage_id=_STAGE,
        source_artifact_id=str(_RUN),
        payload_sink=sink,
        runtime=_Runtime(
            DinoV2NativeOutput(
                array=_grid(2, 3),
                model_input_width=6,
                model_input_height=4,
                patch_width=2,
                patch_height=2,
            )
        ),
    )


def _dinov3(sink: Any) -> DinoV3DenseFeatureBackend:
    return DinoV3DenseFeatureBackend(
        config=DinoV3Config(
            checkpoint="facebook/dinov3-vits16-pretrain-lvd1689m",
            revision=_REVISION,
            input_width=6,
            input_height=4,
        ),
        run_id=_RUN,
        feature_stage_id=_STAGE,
        source_artifact_id=str(_RUN),
        payload_sink=sink,
        runtime=_Runtime(
            DinoV3NativeOutput(
                array=_grid(2, 3),
                model_input_width=6,
                model_input_height=4,
                patch_width=2,
                patch_height=2,
                register_token_count=4,
            )
        ),
    )


def _siglip2(sink: Any) -> Siglip2FeatureBackend:
    return Siglip2FeatureBackend(
        config=Siglip2Config(
            checkpoint="google/siglip2-base-patch16-32",
            revision=_REVISION,
            scope=FeatureScope.DENSE,
            input_size=32,
        ),
        run_id=_RUN,
        feature_stage_id=_STAGE,
        source_artifact_id=str(_RUN),
        payload_sink=sink,
        runtime=_Runtime(
            Siglip2NativeOutput(
                array=_grid(2, 2),
                model_input_width=32,
                model_input_height=32,
                patch_width=16,
                patch_height=16,
            )
        ),
    )


_ADAPTERS = pytest.mark.parametrize(
    "build", [_dinov2, _dinov3, _siglip2], ids=["dinov2", "dinov3", "siglip2"]
)


def _image() -> PreparedImage:
    return PreparedImage(
        source_observation_id=SourceObservationId("frame-0001"),
        payload_reference="prepared/frame-0001.png",
        width=12,
        height=8,
        transformations=(),
    )


@_ADAPTERS
def test_extract_dense_hands_the_sink_the_geometry_of_the_map_it_computed(
    build: Callable[[Any], Any],
) -> None:
    sink = _Sink()

    extraction = build(sink).extract_dense(_image())

    (diagnostic,) = sink.diagnostics
    dense_map = extraction.dense_map
    feature, sampling = dense_map.feature, dense_map.sampling
    assert diagnostic.status is FeatureEventStatus.SUCCEEDED
    assert diagnostic.stage_id == _STAGE
    assert diagnostic.source_observation_id == _image().source_observation_id
    assert diagnostic.source_prepared_image_reference == _image().payload_reference
    assert (diagnostic.source_image_width, diagnostic.source_image_height) == (12, 8)
    assert diagnostic.feature_id == feature.feature_id
    assert diagnostic.scope is FeatureScope.DENSE
    assert diagnostic.embedding_space == extraction.embedding_space
    assert diagnostic.output_shape == feature.shape
    assert diagnostic.dtype == feature.dtype
    assert diagnostic.normalization == feature.normalization
    assert diagnostic.payload_reference == feature.payload_reference
    assert diagnostic.backend == feature.provenance
    assert diagnostic.dense == DenseFeatureDiagnostic(
        source_artifact_id=dense_map.source_artifact_id,
        grid_width=sampling.grid_width,
        grid_height=sampling.grid_height,
        origin_x=sampling.origin_x,
        origin_y=sampling.origin_y,
        stride_x=sampling.stride_x,
        stride_y=sampling.stride_y,
        support_width=sampling.support_width,
        support_height=sampling.support_height,
        coordinate_transform_id=sampling.coordinate_transform_id,
    )


@_ADAPTERS
def test_a_run_written_through_the_sink_is_reopened_as_the_same_dense_map(
    tmp_path: Path, build: Callable[[Any], Any]
) -> None:
    with VisualEvidenceRunWriter(
        output_dir=tmp_path / "run",
        sequence_name="corridor",
        run_id=_RUN,
        run_index=1,
        sequence_artifact_id="corridor-artifact",
        region_artifact_id="regions",
        selection_id="selection-0001",
        configuration_digest="sha256:pipeline",
    ) as writer:
        extraction = build(writer).extract_dense(_image())
        feature = extraction.dense_map.feature
        writer.add_result(
            VisualEvidenceResult(
                source_observation_id=_image().source_observation_id,
                features=(feature,),
            )
        )
        writer.finalize()

    recorded = VisualEvidenceRunReader(tmp_path / "run").dense_feature_map(
        _image().source_observation_id, feature
    )

    assert recorded.stage_id == _STAGE
    assert recorded.dense_map == extraction.dense_map
