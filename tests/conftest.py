"""Pytest configuration shared by the whole suite."""

from __future__ import annotations

import importlib.util

# Módulos de teste que exercitam o extra opcional ``contextmap[ros1]``/``[ros2]``
# e importam o ``rosbags`` já na coleta. Em uma instalação base (só NumPy) eles não
# são coletados em vez de quebrar a suíte inteira; o cabeçalho do relatório
# torna a omissão explícita. Caminhos relativos a este diretório.
_REQUIRES_ROSBAGS = (
    "ingestion/adapters/test_ros1_bag.py",
    "ingestion/adapters/test_ros2_bag.py",
    "ingestion/test_windowed_ingestion_integration.py",
    "ingestion/test_corridor_window_ingestion.py",
    "state_estimation/test_state_estimation_fast_lio_process.py",
)
_ROSBAGS_INSTALLED = importlib.util.find_spec("rosbags") is not None

# Testes de propriedade vivem em módulos ``test_*_properties.py`` e importam o
# ``hypothesis``, que só o extra ``dev`` instala. A CI também roda a suíte numa
# instalação base; lá esses módulos não são coletados, pela mesma razão acima.
_PROPERTY_MODULES = "*/test_*_properties.py"
_HYPOTHESIS_INSTALLED = importlib.util.find_spec("hypothesis") is not None

collect_ignore: list[str] = [] if _ROSBAGS_INSTALLED else list(_REQUIRES_ROSBAGS)
collect_ignore_glob: list[str] = [] if _HYPOTHESIS_INSTALLED else [_PROPERTY_MODULES]

if _HYPOTHESIS_INSTALLED:
    from hypothesis import settings

    # "ci" é o perfil padrão: derandomize deriva a semente de cada propriedade do próprio
    # teste, então toda execução gera os mesmos exemplos (até mudar o teste ou o Hypothesis)
    # e uma falha na CI se reproduz localmente. Sem banco de exemplos e sem deadline: nenhum
    # exemplo vem de uma execução anterior nem depende do relógio da máquina.
    settings.register_profile(
        "ci",
        derandomize=True,
        database=None,
        deadline=None,
        max_examples=100,
        print_blob=True,
    )
    # "explore" é para caçar contraexemplos localmente (``--hypothesis-profile=explore``):
    # aleatório, dez vezes mais exemplos e com o banco padrão em ``.hypothesis/``, que
    # reapresenta as falhas já encontradas. Um contraexemplo achado aqui vira ``@example``.
    settings.register_profile(
        "explore",
        deadline=None,
        max_examples=1000,
        print_blob=True,
    )
    settings.load_profile("ci")


def pytest_report_header() -> list[str]:
    """Report which modules the installed extras leave out, so the omission is visible."""
    lines = []
    if not _ROSBAGS_INSTALLED:
        lines.append(
            "rosbags not installed (base install); not collected: " + ", ".join(_REQUIRES_ROSBAGS)
        )
    if not _HYPOTHESIS_INSTALLED:
        lines.append(f"hypothesis not installed (base install); not collected: {_PROPERTY_MODULES}")
    return lines
