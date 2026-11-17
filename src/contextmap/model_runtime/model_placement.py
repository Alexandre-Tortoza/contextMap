"""Conferência de device e dtype de um modelo torch já carregado e injetado.

Quando o composition root carrega o modelo e o entrega pronto ao adapter, o device e a
precisão entram na provenance pela configuração — então um modelo colocado em outro lugar
seria registrado como algo que não é. Este módulo compara todos os tensores do modelo com
o que a configuração declara, em vez de confiar numa amostra.

É infraestrutura de adapter, não domínio: nada aqui conhece região, feature ou claim.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def verify_model_placement(
    model: object,
    *,
    device: str,
    precision: str,
    backend: str,
    accepted_dtypes: frozenset[str] | None = None,
) -> object:
    """Compare every tensor of the model with the configured device and precision.

    The composition root hands some runtimes a model it loaded itself; its device and
    dtype enter provenance through the configuration, so a model placed elsewhere would
    be recorded as something it is not. One sample is not enough: a model moved only in
    part, or holding weights of several dtypes, would pass a check of its first parameter.

    Every parameter and buffer has to sit on one and the same device, the configured one.
    A model distributed over several devices (``device_map``, offload) is refused: the
    configuration names a single device, and nothing declares a distribution. Every
    parameter dtype has to be accepted by ``precision``; buffers are not weights (they may
    hold integer ids, for instance), so their dtype is not checked.

    Args:
        model: Loaded torch module exposing ``parameters()`` and, optionally, ``buffers()``.
        device: Configured torch device, such as ``"cuda"`` or ``"cuda:0"``. Without an
            index, any index of that device type matches, as ``model.to("cuda")`` does.
        precision: Configured precision name, such as ``"float32"``.
        backend: Backend name used in the error message.
        accepted_dtypes: Weight dtype names under which the runtime really infers in
            ``precision``. Defaults to ``{precision}``, right for a runtime without
            autocast, where the weight dtype is the inference precision.

    Returns:
        The dtype of the first parameter, to cast inputs to without importing torch. With
        the default ``accepted_dtypes`` every parameter shares it.

    Raises:
        ValueError: If the model has no parameter to inspect, if its tensors are not all on
            the configured device, or if it holds weights in a dtype the configured
            precision does not accept. The message counts the tensors per device or dtype.
    """
    parameters = _tensors(model, "parameters")
    if not parameters:
        raise ValueError(f"{backend} model exposes no parameter to verify its device and dtype")
    placements = Counter(
        str(tensor.device) for tensor in (*parameters, *_tensors(model, "buffers"))
    )
    if len(placements) > 1 or not names_device(next(iter(placements)), device):
        distributed = " (distributed placement is not supported)" if len(placements) > 1 else ""
        raise ValueError(
            f"{backend} model tensors are on {_counted(placements)}{distributed}, but the "
            f"configuration declares device {device!r}"
        )
    dtypes = Counter(str(tensor.dtype) for tensor in parameters)
    accepted = accepted_dtypes or frozenset({precision})
    if any(dtype.removeprefix("torch.") not in accepted for dtype in dtypes):
        raise ValueError(
            f"{backend} model parameters are {_counted(dtypes)}, but the configuration "
            f"declares precision {precision!r}, which accepts weights in "
            f"{', '.join(sorted(accepted))}"
        )
    return parameters[0].dtype


def names_device(actual: str, configured: str) -> bool:
    """Tell whether a device named ``actual`` (``"cuda:1"``) is the ``configured`` one.

    Without an index, the configured device matches any index of its type, as
    ``model.to("cuda")`` does.
    """
    actual_type, _, actual_index = actual.partition(":")
    configured_type, _, configured_index = configured.partition(":")
    return actual_type == configured_type and (
        not configured_index or actual_index == configured_index
    )


def _tensors(model: object, accessor: str) -> tuple[Any, ...]:
    """Return what ``model.<accessor>()`` yields, or nothing when the model has no such call."""
    method = getattr(model, accessor, None)
    return tuple(method()) if callable(method) else ()


def _counted(values: Counter[str]) -> str:
    """Render ``values`` as ``"cpu (1 tensor), cuda:0 (2 tensors)"``, in name order."""
    return ", ".join(
        f"{name} ({count} tensor{'' if count == 1 else 's'})"
        for name, count in sorted(values.items())
    )
