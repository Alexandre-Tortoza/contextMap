"""The one module that imports an optional model library or stubs ``pointcept`` in ``sys.modules``.

Side effect, stated explicitly: :func:`import_pointcept_ptv3` writes five entries into the
process-global ``sys.modules``: ``pointcept``, ``pointcept.models``,
``pointcept.models.point_transformer_v3``, ``pointcept.engines`` and ``pointcept.engines.hooks``.
The first four are empty packages whose ``__path__`` points at the Pointcept checkout, and the last
carries a fabricated, empty ``HookBase``. The reason is to avoid the training imports in Pointcept's
own ``__init__`` files (every model, torch_cluster, wandb, tensorboard, compiled ``pointops``),
none of which inference needs, while the backbone's own imports still resolve to the real files of
the checkout. The effect is idempotent: after the first call ``pointcept`` is in ``sys.modules`` and
nothing is written again. It is skipped altogether when a real or already-stubbed ``pointcept`` is
already loaded in the process, which is then used as it is. The stubs are never removed.

Nothing else in the capability imports a model library or touches ``sys.modules``: the rest of the
backend receives what this module returns. Importing this module itself loads no model library.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

from contextmap.point_representation.backends.ptv3.adapter import PTv3RuntimeUnavailableError

_PTV3_MODULE = "pointcept.models.point_transformer_v3.point_transformer_v3m1_base"


def import_optional_module(name: str) -> Any:
    """Import a module; the seam that lets tests simulate a missing model library."""
    return importlib.import_module(name)


def require_dependency(name: str) -> Any:
    """Import an optional model library or fail with what to install.

    Raises:
        PTv3RuntimeUnavailableError: If the library cannot be imported.
    """
    try:
        return import_optional_module(name)
    except ImportError as error:
        raise PTv3RuntimeUnavailableError(
            f"the real PTv3 runtime needs the optional dependency {name!r}, which cannot be "
            f"imported ({error}); install it in the model environment"
        ) from error


def import_pointcept_ptv3(pointcept_root: Path) -> Any:
    """Import only the PTv3 backbone module of a Pointcept checkout.

    Importing ``pointcept.models`` runs an ``__init__`` that pulls every model and the training
    engine (torch_cluster, wandb, tensorboard, compiled ``pointops``), none of which inference
    needs. Instead, lightweight package stubs whose ``__path__`` points at the checkout make the
    backbone's own imports resolve to the real files; only the training-hook base class, which the
    backbone module imports and never uses at inference, is replaced by an empty class. If a real
    or already-stubbed ``pointcept`` is loaded in the process, it is used as it is.

    Side effect: on the first call without a loaded ``pointcept`` it writes five entries into
    ``sys.modules`` (see the module docstring); later calls write nothing.

    Args:
        pointcept_root: Clone of the official Pointcept repository.

    Returns:
        The imported ``point_transformer_v3m1_base`` module.

    Raises:
        PTv3RuntimeUnavailableError: If the module cannot be imported from the checkout.
    """
    if "pointcept" not in sys.modules:
        # Substitui os __init__ de treino do Pointcept por pacotes vazios apontando para o clone;
        # só o HookBase do motor de treino é fabricado, porque o backbone o importa sem usá-lo.
        package = pointcept_root / "pointcept"
        for name, path in (
            ("pointcept", package),
            ("pointcept.models", package / "models"),
            ("pointcept.models.point_transformer_v3", package / "models" / "point_transformer_v3"),
            ("pointcept.engines", package / "engines"),
        ):
            stub = types.ModuleType(name)
            stub.__path__ = [str(path)]
            sys.modules[name] = stub
        hooks = types.ModuleType("pointcept.engines.hooks")
        hooks.HookBase = type("HookBase", (), {})  # type: ignore[attr-defined]
        sys.modules["pointcept.engines.hooks"] = hooks
    try:
        return importlib.import_module(_PTV3_MODULE)
    except ImportError as error:
        raise PTv3RuntimeUnavailableError(
            f"the Pointcept PTv3 module cannot be imported from {pointcept_root} ({error})"
        ) from error
