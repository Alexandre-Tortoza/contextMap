from pathlib import Path

import pytest

from contextmap.evaluation._persistence import write_immutable_json
from sync_recorder import SyncRecorder


def test_a_published_document_keeps_its_exact_bytes(tmp_path: Path) -> None:
    """#640 characterization: indented, key-sorted, UTF-8 without escaping, final newline."""
    path = tmp_path / "reports" / "report.json"

    write_immutable_json(path, {"b": [1, "ação"], "a": {"z": 1.5}}, "evaluation report")

    assert path.read_bytes() == (
        '{\n  "a": {\n    "z": 1.5\n  },\n  "b": [\n    1,\n    "ação"\n  ]\n}\n'.encode()
    )
    assert [item.name for item in path.parent.iterdir()] == ["report.json"]


def test_an_existing_document_is_never_replaced(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text("immutable", encoding="utf-8")

    with pytest.raises(FileExistsError, match="evaluation report already exists"):
        write_immutable_json(path, {"a": 1}, "evaluation report")

    assert path.read_text(encoding="utf-8") == "immutable"
    assert [item.name for item in tmp_path.iterdir()] == ["report.json"]


def test_the_document_reaches_the_disk_before_its_name_and_the_directory_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A disciplina de runtime/_files.py (#616): sem ela, uma queda de energia pode persistir o
    # link antes do conteúdo e deixar um relatório vazio com nome definitivo (#640).
    path = tmp_path / "report.json"
    recorder = SyncRecorder(path, monkeypatch)

    write_immutable_json(path, {"a": 1}, "evaluation report")

    assert recorder.synced_before_publication(path)
    assert recorder.synced_after_publication(tmp_path)
