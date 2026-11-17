"""Semantic request execution owned by the semantic interpretation capability."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    IneligibleRegion,
    IneligibleRegionReason,
    Region2D,
    RegionId,
    RegionMaskResolver,
)
from contextmap.semantic_interpretation.evidence_views import (
    MaterializedSemanticView,
    materialize_region_views,
    materialize_scene_view,
)
from contextmap.semantic_interpretation.interpretation_execution import (
    SemanticBackendCallFailedError,
    SemanticInterpretationExecution,
    SemanticInterpretationFailedError,
)
from contextmap.semantic_interpretation.interpretation_policy import (
    SemanticModePrompt,
    SemanticRequestPolicy,
)
from contextmap.semantic_interpretation.interpretation_request import (
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticRequestId,
)
from contextmap.semantic_interpretation.ports import SemanticInterpreter
from contextmap.semantic_interpretation.run_artifact import (
    SemanticInterpretationRunError,
    SemanticInterpretationRunWriter,
)
from contextmap.semantic_interpretation.scene_context import SceneContext
from contextmap.semantic_interpretation.semantic_claims import (
    SemanticClaim,
    SemanticEvidenceReference,
    SemanticInterpretationRunId,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import PreparedImage


class SemanticInterpretationService:
    """Execute explicit scene and region request policies with auditable outcomes.

    Requests use the semantic producer identity independently from visual features. Exact
    views, successful executions, rejected responses and backend failures reach the bound
    writer immediately. Region failures are isolated; requested scene conditioning is never
    silently dropped. Regions excluded by view requirements or budget remain auditable.
    """

    def __init__(
        self,
        *,
        interpreter: SemanticInterpreter,
        run_id: SemanticInterpretationRunId,
        view_root: Path,
        policy: SemanticRequestPolicy,
        mask_resolver: RegionMaskResolver | None = None,
    ) -> None:
        """Bind the bridge to the interpreter, the run, its view directory and request policy."""
        self._mask_resolver = mask_resolver
        self._interpreter = interpreter
        self._policy = policy
        self._view_policy = policy.views
        self._run_id = run_id
        self._view_root = view_root
        provenance = interpreter.backend_provenance()
        self._configuration_fingerprint = provenance.configuration_fingerprint or provenance.model
        self._writer: SemanticInterpretationRunWriter | None = None
        # O contexto de cena de cada frame, por frame: a cena é interpretada antes das regiões
        # do mesmo frame (ordem topológica do preset), e num run estágio-major todas as cenas
        # vêm antes de todas as regiões, então nenhum frame lê o contexto de outro.
        self._scene_contexts: dict[SourceObservationId, SceneContext | None] = {}

    def bind(self, writer: SemanticInterpretationRunWriter) -> None:
        """Bind every subsequent execution to the real writer.

        Same ordering cycle as :class:`_DeferredFeaturePayloadSink`: the bridge has to exist
        before ``resolve_pipeline()`` can build the stage graph, but the writer needs that
        graph's ``configuration_digest``. Binding happens before any image is processed, so
        every execution is forwarded, never buffered.
        """
        self._writer = writer

    def release(self) -> None:
        """Release the wrapped interpreter's model, when it can release one."""
        release = getattr(self._interpreter, "release", None)
        if callable(release):
            release()

    def backend_provenance(self) -> BackendProvenance:
        """Pass through the wrapped interpreter's provenance unchanged."""
        return self._interpreter.backend_provenance()

    def _require_writer(self) -> SemanticInterpretationRunWriter:
        """Require a bound artifact sink before processing any request."""
        if self._writer is None:
            raise SemanticInterpretationRunError(
                "semantic bridge used before bind(): no writer to publish to"
            )
        return self._writer

    def _publish(
        self,
        execution: SemanticInterpretationExecution,
        views: Sequence[MaterializedSemanticView],
    ) -> None:
        """Hand one finished execution and the bytes of its views to the writer.

        Nothing is retained: holding the payloads until the image loop ended cost
        O(semantic requests x image size) resident and undid the writer's own streaming.

        The execution is the request's outcome, so it adds no record to
        ``metrics/stage-timings.jsonl``: the stage it ran in has exactly one there, recorded by
        the executor (#654).
        """
        self._require_writer().add_semantic_execution(execution)
        self._publish_views(views)

    def _publish_views(self, views: Sequence[MaterializedSemanticView]) -> None:
        """Hand the exact bytes of each view a request was issued with to the writer.

        Without this the payload would stay in the scratch the executor deletes, and the
        request's contractual record would cite a payload nobody can recover.
        """
        writer = self._require_writer()
        for item in views:
            writer.add_semantic_view_payload(item.view, item.payload)

    def _frame(self, image: PreparedImage) -> tuple[Any, str]:
        """Decode the prepared image to RGB pixels and identify its exact encoded bytes."""
        import importlib
        import io

        import numpy as np

        encoded = (self._view_root / image.payload_reference).read_bytes()
        image_module = importlib.import_module("PIL.Image")
        with image_module.open(io.BytesIO(encoded)) as opened:
            pixels = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        return pixels, hashlib.sha256(encoded).hexdigest()

    def _stage_views(self, views: Sequence[MaterializedSemanticView]) -> None:
        """Write each view where the interpreter's runtime reads it by its recorded SHA-256."""
        for item in views:
            target = self._view_root / item.view.payload_reference
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.payload)

    def _prompt(self, mode: SemanticInterpretationMode) -> SemanticModePrompt:
        """Return the prompt policy composed for ``mode``; a missing one is never defaulted.

        Raises:
            SemanticInterpretationRunError: If no prompt policy was composed for the mode.
        """
        prompt = self._policy.prompt_for(mode)
        if prompt is None:
            raise SemanticInterpretationRunError(
                f"no semantic prompt policy was composed for {mode.value} requests; the "
                "configured semantic interpreter cannot interpret this mode"
            )
        return prompt

    def _interpret_preserving_failures(
        self,
        request: SemanticInterpretationRequest,
        views: Sequence[MaterializedSemanticView],
    ) -> SemanticInterpretationExecution:
        """Run one interpretation, recording a failed request's outcome before re-raising it.

        The model's real answer is never reduced to an error string: a response the parser
        rejects reaches the artifact's failures stream, and a call that failed before answering
        reaches the backend call failures stream (#654), each with the views it was issued
        with. ``attempted`` is then reconciled against ``parsed``, ``parse_failed`` and
        ``call_failed`` by ``request_id`` instead of being inferred. The error is re-raised so
        the caller decides what the failed request means for its stage.
        """
        try:
            return self._interpreter.interpret(request)
        except (SemanticBackendCallFailedError, SemanticInterpretationFailedError) as failed:
            if self._writer is None:
                raise SemanticInterpretationRunError(
                    "semantic bridge used before bind(): no writer to record the failure in"
                ) from failed
            # As views entram antes da falha: sao a evidencia visual exata com que o request
            # foi emitido, e sem isto ficariam so no scratch, que o executor apaga -- o
            # registro contratual citaria um payload_reference irrecuperavel.
            self._publish_views(views)
            if isinstance(failed, SemanticBackendCallFailedError):
                self._writer.add_failed_semantic_backend_call(failed.failure)
            else:
                self._writer.add_failed_semantic_interpretation(failed.failure)
            raise

    def _conditioning(self, image: PreparedImage) -> SceneContext:
        """Return the scene context this frame's scene request produced (#529).

        Raises:
            SemanticInterpretationRunError: If the scene request failed, abstained or has
                not run. The region request is refused, never sent without the context
                the configuration asked for.
        """
        cached = self._scene_contexts.get(image.source_observation_id)
        if cached is None:
            raise SemanticInterpretationRunError(
                f"region requests are conditioned on scene context, but frame "
                f"{image.source_observation_id!r} has no scene context from its scene request"
            )
        return cached

    def interpret_scene(self, image: PreparedImage) -> SceneContext | None:
        """Build the frame's SCENE request from its full-frame view and delegate to interpret().

        A scene request is never conditioned on scene context, so conditioning is acyclic.
        """
        self._scene_contexts[image.source_observation_id] = None
        prompt = self._prompt(SemanticInterpretationMode.SCENE)
        pixels, source_sha256 = self._frame(image)
        views = (
            materialize_scene_view(
                pixels,
                source_observation_id=image.source_observation_id,
                source_image_sha256=source_sha256,
                policy=self._view_policy,
            ),
        )
        self._stage_views(views)
        request = SemanticInterpretationRequest(
            request_id=SemanticRequestId(f"{self._run_id}--scene-{image.source_observation_id}"),
            source_observation_id=image.source_observation_id,
            semantic_interpretation_run_id=self._run_id,
            mode=SemanticInterpretationMode.SCENE,
            visual_views=tuple(item.view for item in views),
            prompt_template_id=prompt.template_id,
            requested_output_schema=prompt.output_schema,
            configuration_fingerprint=self._configuration_fingerprint,
        )
        execution = self._interpret_preserving_failures(request, views)
        self._publish(execution, views)
        self._scene_contexts[image.source_observation_id] = execution.parsed.scene_context
        return execution.parsed.scene_context

    def interpret_regions(
        self, image: PreparedImage, regions: Sequence[Region2D]
    ) -> Sequence[SemanticClaim]:
        """Build one REGION request per region with the policy's views, in order, and delegate.

        Every region gets its request, whatever happened to the previous ones (#654): a failed
        request is already recorded in its contractual stream, so it contributes no claim and
        the next region is requested. The claims of every request that answered are returned,
        in region order, even when all others failed or none answered.

        A region the policy's views cannot be built for is never requested and never dropped
        silently (#650): under a policy that declares ``masked_subject``, a region without an
        inline mask (a grounded box) is handed to the writer as an ineligible region of
        ``region_interpretation``, and the next region is requested.
        """
        if not regions:
            return ()
        pixels, source_sha256 = self._frame(image)
        claims: list[SemanticClaim] = []
        budget = self._policy.region_budget
        requested = (
            None if budget is None else budget.select(regions, mask_resolver=self._mask_resolver)
        )
        for region in regions:
            if requested is not None and region.region_id not in requested:
                # Fora do orçamento a região não recebe request, mas continua no resultado e
                # segue para a associação e o mapa; o motivo fica gravado, nunca some.
                assert budget is not None
                self._require_writer().add_ineligible_region(
                    IneligibleRegion(
                        source_observation_id=image.source_observation_id,
                        region_id=region.region_id,
                        stage_id="region_interpretation",
                        reason=IneligibleRegionReason.REGION_BUDGET,
                        detail=(
                            f"{budget.policy_id} requests the {budget.max_regions} largest "
                            f"of the frame's {len(regions)} regions"
                        ),
                    )
                )
                continue
            if (
                self._view_policy.requires_region_mask
                and region.mask is None
                and region.mask_reference is None
            ):
                # masked_subject pinta tudo fora da máscara e as views de contorno desenham em
                # volta dela; uma caixa não tem máscara, e inventá-la a partir da caixa fabricaria
                # evidência que o grounding não produziu.
                views_needing_mask = ", ".join(
                    kind.value for kind in self._view_policy.mask_bound_views
                )
                self._require_writer().add_ineligible_region(
                    IneligibleRegion(
                        source_observation_id=image.source_observation_id,
                        region_id=region.region_id,
                        stage_id="region_interpretation",
                        reason=IneligibleRegionReason.NO_MASK,
                        detail=(
                            f"the view policy declares {views_needing_mask}, built from the "
                            "region's inline mask, and the region is box-only "
                            f"({region.provenance.capability})"
                        ),
                    )
                )
                continue
            # Consultado por região, como antes: um frame sem regiões não exige política.
            prompt = self._prompt(SemanticInterpretationMode.REGION)
            context = self._conditioning(image) if self._policy.region_scene_context else None
            views = materialize_region_views(
                pixels,
                source_observation_id=image.source_observation_id,
                source_image_sha256=source_sha256,
                region=region,
                policy=self._view_policy,
                mask_resolver=self._mask_resolver,
            )
            self._stage_views(views)
            request = SemanticInterpretationRequest(
                request_id=SemanticRequestId(
                    f"{self._run_id}--region-{image.source_observation_id}-{region.region_id}"
                ),
                source_observation_id=image.source_observation_id,
                semantic_interpretation_run_id=self._run_id,
                mode=SemanticInterpretationMode.REGION,
                visual_views=tuple(item.view for item in views),
                region_id=RegionId(str(region.region_id)),
                prompt_template_id=prompt.template_id,
                requested_output_schema=prompt.output_schema,
                configuration_fingerprint=self._configuration_fingerprint,
                scene_context_reference=(
                    None
                    if context is None
                    else SemanticEvidenceReference(
                        evidence_type="scene_context",
                        evidence_id=str(context.context_id),
                    )
                ),
                scene_context=context,
            )
            try:
                execution = self._interpret_preserving_failures(request, views)
            except (SemanticBackendCallFailedError, SemanticInterpretationFailedError):
                # O desfecho deste request ja esta no seu stream contratual; o request da regiao
                # seguinte e independente dele, e parar aqui descartaria as claims ja
                # respondidas e deixaria as regioes restantes sem request (#654).
                continue
            self._publish(execution, views)
            claims.extend(execution.parsed.claims)
        return tuple(claims)
