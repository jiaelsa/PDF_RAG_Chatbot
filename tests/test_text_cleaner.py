from rag.text_cleaner import normalize_text, strip_headers_footers


def test_dehyphenation_and_line_joining():
    raw = "Retrieval aug-\nmented generation is\nuseful.\n\nSecond paragraph."
    assert normalize_text(raw) == "Retrieval augmented generation is useful.\n\nSecond paragraph."


def test_ligatures_and_control_chars():
    assert normalize_text("\ufb01nd\x00 this") == "find this"


def test_empty():
    assert normalize_text("") == ""
    assert normalize_text("   \n \n ") == ""


def test_headers_footers_removed_on_long_docs():
    pages = [f"ACME Confidential Report\nReal content number {i}.\nPage {i + 1}" for i in range(6)]
    out = strip_headers_footers(pages)
    assert all("ACME" not in p and "Page" not in p for p in out)
    assert "Real content number 3." in out[3]


def test_short_docs_keep_repeated_lines():
    pages = ["Title\nA", "Title\nB"]
    assert "Title" in strip_headers_footers(pages)[0]
