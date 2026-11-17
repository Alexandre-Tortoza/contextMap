"""Vocabulário de validação compartilhado pelos contratos de Semantic Mapping.

Reúne as invariantes que mais de um contrato da capability precisa afirmar: presença de uma
identidade, contagem inteira de pelo menos um e ordem canônica (ordenada e única). A ordem
canônica aparece em quase todo contrato daqui porque é ela que garante que duas entidades
equivalentes sempre codifiquem para os mesmos bytes, e portanto que um artifact persistido
seja reproduzível.

São funções públicas *dentro* da capability, não fora dela: a fachada não as exporta. Elas não
constroem contrato, não decidem política científica, não escolhem limiar e não têm side effect.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import TypeVar

_Item = TypeVar("_Item")


def require_present(owner: object, *names: str) -> None:
    """Require that the named string attributes of ``owner`` are not blank.

    Args:
        owner: The object being validated.
        *names: Attribute names to check.

    Raises:
        ValueError: If an attribute is empty or only whitespace.
    """
    for name in names:
        if not str(getattr(owner, name)).strip():
            raise ValueError(f"{name} must not be empty")


def require_count_of_at_least_one(name: str, value: object) -> None:
    """Require a whole count of at least one.

    Args:
        name: Name of the value, for the error message.
        value: The value to check; a ``bool`` is not a count, even though Python treats it as an
            integer.

    Raises:
        ValueError: If the value is not an integer or is below one.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer of at least 1, got {value!r}")


def require_canonical(
    name: str, items: Sequence[_Item], key: Callable[[_Item], tuple[str, ...]], *, detail: str = ""
) -> None:
    """Require strictly increasing keys, so equivalent content always encodes identically.

    Args:
        name: Name of the collection, for the error message.
        items: The collection to check.
        key: Sort key of one item.
        detail: What the collection is sorted by, when not obvious.

    Raises:
        ValueError: If the keys are not strictly increasing, which also refuses duplicates.
    """
    keys = [key(item) for item in items]
    if any(left >= right for left, right in pairwise(keys)):
        raise ValueError(f"{name} must be sorted {detail}and unique")
