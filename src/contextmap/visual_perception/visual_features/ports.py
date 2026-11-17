"""Pontos de substituição dos backends de extração e realce de features visuais.

``FeatureExtractor`` cobre uma variação já exercida por cinco adapters (DINOv2,
DINOv3, CLIP, AlphaCLIP, SigLIP2), que diferem no escopo que produzem.
``FeatureResolutionEnhancement`` isola o realce de resolução, cuja fronteira
validada vive em :mod:`contextmap.visual_perception.visual_features.feature_enhancement`.

Nenhum port constrói backend nem importa SDK de modelo: isso pertence ao
composition root.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from contextmap.region_perception import Region2D
from contextmap.shared import BackendProvenance
from contextmap.visual_perception.models import PreparedImage
from contextmap.visual_perception.visual_features.region_feature_pooling import DenseFeatureMap
from contextmap.visual_perception.visual_features.visual_evidence import FeatureScope, VisualFeature


@runtime_checkable
class FeatureExtractor(Protocol):
    """Capability port: extract visual features from a prepared image.

    A dense/global extractor only ever receives ``image``; a region-scoped
    extractor also receives ``regions``. Which one applies is declared by
    :meth:`required_scope`, not encoded in a shared mandatory signature.
    """

    def backend_provenance(self) -> BackendProvenance:
        """Report this backend's identity and effective configuration.

        Returns:
            The provenance to attach to every feature this call produces.
        """
        ...

    def required_scope(self) -> FeatureScope:
        """Report which scope this extractor produces.

        Returns:
            ``DENSE`` or ``GLOBAL`` for an extractor that only needs ``image``;
            ``REGION`` for one that also needs ``regions``.
        """
        ...

    def extract(
        self, image: PreparedImage, regions: Sequence[Region2D] = ()
    ) -> Sequence[VisualFeature]:
        """Extract visual features.

        Args:
            image: The prepared image to extract from.
            regions: Regions to extract per-region features for. Ignored by an
                extractor whose :meth:`required_scope` is not ``REGION``;
                required (non-empty) when it is.

        Returns:
            Extracted features, matching :meth:`required_scope`.
        """
        ...


@runtime_checkable
class FeatureResolutionEnhancement(Protocol):
    """Capability port: increase a dense feature map's spatial resolution."""

    def backend_provenance(self) -> BackendProvenance:
        """Report the selected enhancement model and effective configuration."""
        ...

    def enhance(self, source: DenseFeatureMap) -> DenseFeatureMap:
        """Produce a separately identified dense map from one source artifact.

        Args:
            source: Immutable source dense map, including payload/artifact
                references and exact sampling semantics.

        Returns:
            Enhanced map with complete enhancement lineage. Representation
            changes must use a distinct embedding-space fingerprint.
        """
        ...


__all__ = ["FeatureExtractor", "FeatureResolutionEnhancement"]
