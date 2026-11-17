"""Auditoria humana de interpretação semântica e redação de segredos."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from contextmap.semantic_interpretation.serialization import (
    encode_claim,
    encode_scene_context,
)

if TYPE_CHECKING:
    from contextmap.semantic_interpretation.interpretation_execution import (
        FailedSemanticBackendCall,
        FailedSemanticInterpretation,
        SemanticInterpretationExecution,
    )

from contextmap.shared import DEBUG_DIRECTORY

SEMANTIC_DEBUG_ROOT = f"{DEBUG_DIRECTORY}/40-semantic-interpretation"
"""Run-relative root for non-contractual semantic diagnostics."""

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "client_secret",
    "credential",
    "credentials",
    "password",
    "access_token",
    "refresh_token",
}


class SemanticDebugLevel(Enum):
    """Amount of non-contractual semantic diagnostics to persist."""

    NONE = "none"
    STANDARD = "standard"
    FULL = "full"


def redact_semantic_secrets(value: Any) -> Any:
    """Recursively redact known credential fields before serialization."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized = key.casefold().replace("-", "_")
            redacted[key] = (
                "[REDACTED]" if normalized in _SENSITIVE_KEYS else redact_semantic_secrets(item)
            )
        return redacted
    if isinstance(value, (list, tuple)):
        return [redact_semantic_secrets(item) for item in value]
    return value


def write_semantic_audit(
    *,
    run_root: Path,
    execution: (
        SemanticInterpretationExecution | FailedSemanticInterpretation | FailedSemanticBackendCall
    ),
    debug_level: SemanticDebugLevel,
) -> tuple[str, ...]:
    """Write request/prompt/outcome/diagnostics for one request, per the debug level.

    Accepts every outcome of a request that reached the backend. All land in the same
    ``__DEBUG__/40-semantic-interpretation/<request-id>/`` directory and share request, prompt and
    diagnostics; they differ in how far the request got, and the files say so:

    - an execution that materialized adds ``parsed-response.json`` and its claims or scene
      context, plus the raw response at ``FULL``;
    - a response the parser rejected adds ``error.json`` (``stage="response_parsing"``), plus
      the raw response at ``FULL``;
    - a backend call that failed before answering adds only ``error.json``
      (``stage="backend_invocation"``): there is no response to copy.

    One writer, so the outcomes can never drift apart. Nothing here is contractual: the
    streams in ``outputs/`` and ``metrics/stage-timings.jsonl`` stay authoritative.
    """
    if debug_level is SemanticDebugLevel.NONE:
        return ()
    from contextmap.semantic_interpretation.interpretation_execution import (
        FailedSemanticBackendCall,
        FailedSemanticInterpretation,
    )

    request_id = str(execution.request.request_id)
    request_path = PurePosixPath(request_id)
    if request_path.name != request_id or request_id in {".", ".."}:
        raise ValueError(f"semantic request_id is not a safe path segment: {request_id!r}")
    relative_root = f"{SEMANTIC_DEBUG_ROOT}/{request_id}"
    root = run_root / relative_root
    root.mkdir(parents=True, exist_ok=True)
    if isinstance(execution, FailedSemanticBackendCall):
        return _write_shared_audit(
            root=root,
            relative_root=relative_root,
            execution=execution,
            debug_level=debug_level,
            raw_response=None,
            outcome={
                "error.json": _error_record(
                    execution,
                    stage="backend_invocation",
                    error_type=execution.error_type,
                    message=execution.message,
                    raw_response_sha256=None,
                )
            },
        )
    if isinstance(execution, FailedSemanticInterpretation):
        return _write_shared_audit(
            root=root,
            relative_root=relative_root,
            execution=execution,
            debug_level=debug_level,
            raw_response=execution.raw_response,
            outcome={
                "error.json": _error_record(
                    execution,
                    stage="response_parsing",
                    error_type=execution.failure.kind,
                    message=execution.failure.message,
                    raw_response_sha256=execution.raw_response_sha256,
                )
            },
        )
    parsed = {
        "abstained": execution.parsed.abstained,
        "raw_response_sha256": execution.parsed.raw_response_sha256,
        "diagnostics": [
            {"code": item.code, "message": item.message} for item in execution.parsed.diagnostics
        ],
        "claims": [encode_claim(claim) for claim in execution.parsed.claims],
        "scene_context": (
            None
            if execution.parsed.scene_context is None
            else encode_scene_context(execution.parsed.scene_context)
        ),
    }
    outcome = {"parsed-response.json": json.dumps(parsed, indent=2, sort_keys=True) + "\n"}
    if execution.parsed.scene_context is None:
        outcome["semantic-claims.json"] = (
            json.dumps(parsed["claims"], indent=2, sort_keys=True) + "\n"
        )
    else:
        outcome["scene-context.json"] = (
            json.dumps(parsed["scene_context"], indent=2, sort_keys=True) + "\n"
        )
    return _write_shared_audit(
        root=root,
        relative_root=relative_root,
        execution=execution,
        debug_level=debug_level,
        raw_response=execution.raw_response,
        outcome=outcome,
    )


def _error_record(
    failed: FailedSemanticInterpretation | FailedSemanticBackendCall,
    *,
    stage: str,
    error_type: str,
    message: str,
    raw_response_sha256: str | None,
) -> str:
    """Render the machine-readable ``error.json`` of one failed request (#558).

    ``stage`` names the lifecycle point the request failed at: ``backend_invocation`` (no
    response exists, so ``raw_response_sha256`` is null) or ``response_parsing`` (the parser,
    which also validates the schema and canonicalizes, rejected an observed response). The
    type and message are the failing exception's, verbatim: no failure taxonomy, field path
    or expected value is inferred from them.
    """
    backend = failed.provenance.backend
    record = {
        "stage": stage,
        "error_type": error_type,
        "message": message,
        "request_id": str(failed.request.request_id),
        "backend": {
            "backend_id": backend.backend_id,
            "provider": backend.provider,
            "model": backend.model,
            "version": backend.version,
            "configuration_fingerprint": backend.configuration_fingerprint,
        },
        "task_identity": failed.provenance.task_identity,
        "retries": failed.diagnostics.retries,
        "raw_response_sha256": raw_response_sha256,
        "occurred_at": failed.occurred_at,
    }
    return json.dumps(redact_semantic_secrets(record), indent=2, sort_keys=True) + "\n"


def _write_shared_audit(
    *,
    root: Path,
    relative_root: str,
    execution: (
        SemanticInterpretationExecution | FailedSemanticInterpretation | FailedSemanticBackendCall
    ),
    debug_level: SemanticDebugLevel,
    raw_response: str | None,
    outcome: dict[str, str],
) -> tuple[str, ...]:
    """Write what every outcome shares, plus the caller's outcome-specific files.

    ``raw_response`` is ``None`` when the backend never answered; otherwise ``FULL`` copies it.
    """
    # Import tardio: semantic_backend importa este módulo (redação) no carregamento.
    from contextmap.semantic_interpretation.interpretation_execution import (
        encode_semantic_backend_diagnostics,
    )
    from contextmap.semantic_interpretation.interpretation_request import (
        encode_semantic_request,
    )

    records: dict[str, str] = {
        "request.json": json.dumps(
            redact_semantic_secrets(encode_semantic_request(execution.request)),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        "prompt.txt": execution.rendered_prompt.text,
        "diagnostics.json": json.dumps(
            redact_semantic_secrets(
                {
                    **encode_semantic_backend_diagnostics(execution.diagnostics),
                    "effective_configuration": dict(execution.effective_configuration),
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        **outcome,
    }
    if debug_level is SemanticDebugLevel.FULL and raw_response is not None:
        records["raw-response.txt"] = raw_response
    for filename, content in records.items():
        (root / filename).write_text(content, encoding="utf-8")
    return tuple(f"{relative_root}/{filename}" for filename in sorted(records))
