"""Run one ``contextmap`` command in this process, as the installed binary would.

Usage: ``python context_step.py <request.json>``, written by ``steps.ContextCli``.

The command goes through ``contextmap.runtime.cli.main`` unchanged, with nothing injected: the
CLI composes every stage from the configuration (perception through ``resources.providers``)
and records the code identity the command passes (``--code-identity``), exactly as the
installed ``contextmap`` binary does (#659).

A dry run only: the executors a ``"module:attribute"`` target returns replace their stages
(perception, on a machine without models), through the Python-embedding path the CLI documents
(``main(executors=...)``), and optional modules are not required, since the replaced stage
imports none.

The CLI's JSON answer is written to the request's ``answer`` path, so nothing a library prints
on standard output can corrupt it. The exit status is the CLI's.
"""

from __future__ import annotations

import importlib
import io
import json
import sys
from pathlib import Path
from typing import Any

from contextmap.runtime.cli import main as contextmap


def _dry_run_executors(target: str) -> dict[str, Any]:
    """Import the ``"module:attribute"`` factory of a dry run's executors and call it."""
    module, _, attribute = target.partition(":")
    executors: dict[str, Any] = getattr(importlib.import_module(module), attribute)()
    return executors


def main(request_path: str) -> int:
    """Run the requested command and write its JSON answer; return its exit status."""
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    options: dict[str, Any] = {}
    if request["dry_run_executors"] is not None:
        options["executors"] = _dry_run_executors(request["dry_run_executors"])
        options["module_available"] = lambda _name: True
    argv = list(request["argv"])
    for config in request["configs"]:
        argv += ["-c", config]
    for override in request["overrides"]:
        argv += ["--set", override]
    out = io.StringIO()
    code = contextmap(argv, stdout=out, **options)
    Path(request["answer"]).write_text(out.getvalue(), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
