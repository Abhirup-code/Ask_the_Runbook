from pathlib import Path

from runbook import lint

ROOT = Path(__file__).resolve().parents[1]


def write(tmp_path, name, text):
    (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def test_the_projects_own_runbooks_are_clean():
    assert lint.check(ROOT / "corpus") == []


def test_an_empty_heading_never_reaches_the_index(tmp_path):
    write(
        tmp_path,
        "r.md",
        "# Doc\n\n## Written up\n\n"
        + ("a real sentence of runbook text here. " * 3)
        + "\n\n## Still to write\n",
    )
    problems = lint.check(tmp_path)
    assert [p.where for p in problems] == ["r.md"]
    assert "Still to write" in problems[0].message
    assert "not indexed" in problems[0].message


def test_a_file_with_no_title_line_is_reported(tmp_path):
    write(tmp_path, "r.md", "## Section\n\nbody long enough to count as a real section of a runbook here.\n")
    assert any("no '# Title' line" in p.message for p in lint.check(tmp_path))


def test_two_sections_with_the_same_title_are_reported(tmp_path):
    body = "body long enough to count as a real section of a runbook, with words in it.\n"
    write(tmp_path, "r.md", f"# Doc\n\n## Step\n\n{body}\n## Step\n\n{body}")
    assert any("2 sections are titled 'Step'" in p.message for p in lint.check(tmp_path))


def test_a_section_over_the_budget_is_reported(tmp_path):
    write(tmp_path, "r.md", "# Doc\n\n## Long\n\n" + ("filler words here " * 300))
    problems = lint.check(tmp_path)
    assert any("over the" in p.message and p.where == "r#0" for p in problems)


def test_a_section_with_almost_no_body_is_reported(tmp_path):
    write(tmp_path, "r.md", "# Doc\n\n## Thin\n\ntoo short.\n")
    assert any("too little to answer from" in p.message for p in lint.check(tmp_path))


def test_a_file_with_no_sections_at_all_is_reported(tmp_path):
    write(tmp_path, "r.md", "# Doc\n")
    assert any("no sections at all" in p.message for p in lint.check(tmp_path))


def test_the_report_reads_as_prose(tmp_path):
    write(tmp_path, "r.md", "# Doc\n\n## Thin\n\ntoo short.\n")
    text = lint.report(lint.check(tmp_path), tmp_path)
    assert text.startswith("1 problem(s)")
    assert "r#0: only" in text


def test_the_report_says_so_when_there_is_nothing_to_say(tmp_path):
    write(tmp_path, "r.md", "# Doc\n\n## Fine\n\n" + ("a real sentence of runbook text here. " * 4))
    assert lint.check(tmp_path) == []
    assert lint.report([], tmp_path).startswith("No problems found")
