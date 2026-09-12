def test_readme_exists_and_has_kdp_summary():
    readme_path = __import__("pathlib").Path(__file__).parent.parent / "README.md"
    assert readme_path.exists(), "README.md should exist"
    content = readme_path.read_text()
    assert "PageLoom" in content
    assert "KDP" in content
    assert "pipeline" in content.lower()
    # Should be roughly one paragraph (few lines, no multiple headings)
    lines = content.splitlines()
    assert len(lines) >= 2
