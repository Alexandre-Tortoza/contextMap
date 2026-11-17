"""Vocabulário de validação compartilhado pelos contratos de Semantic Fusion.

Reúne as invariantes que mais de um contrato da capability precisa afirmar: presença
de identidade, ordem canônica (ordenada e única, para que evidência equivalente
codifique de forma idêntica), fator em ``[0, 1]``, backend identificado, suporte de
geometria não vazio, intervalo de aquisição num único domínio de clock e a chave de
ordenação de um produtor.

São funções públicas *dentro* da capability, não fora dela: a fachada não as exporta.
Elas não constroem contrato, não decidem política científica e não têm side effect.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import TypeVar

from contextmap.geometric_mapping import GeometryReference
from contextmap.shared import BackendProvenance, SourceTimestamp
from contextmap.state_estimation import TimeBounds

_Item = TypeVar("_Item")


def require_unit_factor(name: str, value: float) -> None:
    """Require a dimensionless factor inside the closed unit interval.

    Args:
        name: Field name, used in the error message.
        value: The factor to check.

    Raises:
        ValueError: If the value is not finite or lies outside ``[0, 1]``.
    """
    if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be within [0, 1], got {value!r}")


def require_present(owner: object, *names: str) -> None:
    """Require that each named attribute of ``owner`` holds a non-blank value.

    Args:
        owner: The contract being validated.
        *names: Attribute names that must be present.

    Raises:
        ValueError: If an attribute is empty or only whitespace.
    """
    for name in names:
        if not str(getattr(owner, name)).strip():
            raise ValueError(f"{name} must not be empty")


def require_canonical(
    name: str, items: Sequence[_Item], key: Callable[[_Item], tuple[str, ...]], *, detail: str = ""
) -> None:
    """Require strictly increasing keys, so equivalent evidence encodes identically."""
    keys = [key(item) for item in items]
    if any(left >= right for left, right in pairwise(keys)):
        raise ValueError(f"{name} must be sorted {detail}and unique")


def time_bounds_of(stamps: Sequence[SourceTimestamp], *, owner: str) -> TimeBounds:
    """Span the given acquisition timestamps, which must share one clock domain."""
    clocks = {stamp.clock_id for stamp in stamps}
    if len(clocks) != 1:
        raise ValueError(f"{owner} span more than one clock domain: {sorted(clocks)!r}")
    return TimeBounds(
        start=min(stamps, key=SourceTimestamp.total_nanoseconds),
        end=max(stamps, key=SourceTimestamp.total_nanoseconds),
    )


def producer_key(producer: BackendProvenance) -> tuple[str, ...]:
    """Compute the canonical ordering key of a producer.

    Two producers compare equal only when backend, capability, provider, model, version and
    configuration identify the same execution semantics.

    Args:
        producer: The interpreter or scorer provenance.

    Returns:
        The key, with an absent configuration fingerprint spelled as an empty string.
    """
    return (
        producer.backend_id,
        producer.capability,
        producer.provider,
        producer.model,
        producer.version,
        producer.configuration_fingerprint or "",
    )


def require_identified(name: str, producer: BackendProvenance) -> None:
    """Require a producer that names the backend, model and version behind a value.

    Args:
        name: Field name, used in the error message.
        producer: The provenance to check.

    Raises:
        ValueError: If the backend, the model or the version is missing.
    """
    if not (producer.backend_id.strip() and producer.model.strip() and producer.version.strip()):
        raise ValueError(f"{name} must identify its backend_id, model and version")


def require_geometry_support(name: str, references: Sequence[GeometryReference]) -> None:
    """Require a non-empty geometry support in canonical order.

    Args:
        name: Field name, used in the error message.
        references: The geometry the contract sees.

    Raises:
        ValueError: If there is no geometry or it is not sorted by ``geometry_id`` and unique.
    """
    if not references:
        raise ValueError(f"{name} must not be empty")
    require_canonical(
        name, references, lambda reference: (reference.geometry_id,), detail="by geometry_id "
    )
