from pathlib import Path

import pytest

from runbook.docs import load, split

CORPUS = Path(__file__).parent.parent / "corpus"


def test_splits_at_headings_and_keeps_the_document_title():
    md = "# Runbook: Disk full\n\nIntro text.\n\n## 1. Check\n\nRun df.\n\n## 2. Fix\n\nDelete logs.\n"
    sections = split("disk", md)
    assert [s.id for s in sections] == ["disk#0", "disk#1", "disk#2"]
    assert [s.title for s in sections] == ["Runbook: Disk full", "1. Check", "2. Fix"]
    assert all(s.doc == "Runbook: Disk full" for s in sections)
    assert sections[2].text == "Delete logs."


def test_hash_comments_inside_code_blocks_are_not_headings():
    md = "# Doc\n\n## Query\n\n```promql\n# error ratio\nrate(x[5m])\n```\n"
    sections = split("q", md)
    assert len(sections) == 1
    assert "# error ratio" in sections[0].text


def test_empty_sections_are_skipped():
    sections = split("e", "# Doc\n\n## Empty\n\n## Full\n\nText.\n")
    assert [s.title for s in sections] == ["Full"]
    assert sections[0].id == "e#0"


def test_a_file_without_headings_is_one_section():
    sections = split("notes", "Just some text.\n")
    assert len(sections) == 1
    assert sections[0].title == "notes"


def test_render_shows_the_citation_id_first():
    section = split("d", "# Doc\n\n## A\n\nBody.\n")[0]
    assert section.render().startswith("[d#0] Doc > A\n")


def test_the_corpus_loads_with_unique_ids():
    sections = load(CORPUS)
    ids = [s.id for s in sections]
    assert len(ids) == len(set(ids))
    assert any(s.title.startswith("2. Check the usual causes") for s in sections)


def test_an_empty_folder_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="No Markdown sections"):
        load(tmp_path)
