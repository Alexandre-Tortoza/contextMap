"""Simulate a changed code default of a configuration dataclass, for identity tests (#636)."""

from __future__ import annotations

import dataclasses

import pytest


def replace_default(
    monkeypatch: pytest.MonkeyPatch, config_type: type, name: str, value: object
) -> None:
    """Make every ``config_type`` built without ``name`` get ``value``, as if the code changed.

    The generated ``__init__`` keeps the defaults of its trailing parameters in
    ``__defaults__``; replacing the one of ``name`` is what editing the field's default would do,
    for every caller, without touching the class. ``monkeypatch`` restores it after the test.
    """
    init = vars(config_type)["__init__"]  # o __init__ gerado pelo dataclass, da própria classe
    if init.__kwdefaults__ and name in init.__kwdefaults__:
        # Um campo kw_only guarda o default em __kwdefaults__, não em __defaults__.
        monkeypatch.setitem(init.__kwdefaults__, name, value)
        return
    names = [field.name for field in dataclasses.fields(config_type) if field.init]
    defaults = list(init.__defaults__)
    defaults[names.index(name) - (len(names) - len(defaults))] = value
    monkeypatch.setattr(init, "__defaults__", tuple(defaults))
