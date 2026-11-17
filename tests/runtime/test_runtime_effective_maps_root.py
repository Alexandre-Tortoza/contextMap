"""``effective_maps_root`` é a única definição de onde um build publica o seu mapa."""

from pathlib import Path

from contextmap.runtime import effective_maps_root


def test_an_unconfigured_maps_root_is_a_clean_sibling_of_the_run_store(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"

    assert effective_maps_root(workspace, None) == tmp_path / "maps"


def test_a_configured_maps_root_wins(tmp_path: Path) -> None:
    configured = tmp_path / "elsewhere"

    assert effective_maps_root(tmp_path / "ws", str(configured)) == configured
