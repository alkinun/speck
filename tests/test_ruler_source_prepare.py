import hashlib

from scripts.ruler_source_prepare import parse_paul_html


def test_paul_html_canonicalization_ignores_dynamic_footer():
    first = b"<html><font>Essay <b>body</b>.</font><footer>time 1</footer></html>"
    second = b"<html><font>Essay <b>body</b>.</font><footer>time 2</footer></html>"

    assert parse_paul_html(first) == parse_paul_html(second)
    assert "Essay" in parse_paul_html(first)


def test_paul_html_canonicalization_changes_with_the_essay():
    first = parse_paul_html(b"<font>First essay.</font>").encode()
    second = parse_paul_html(b"<font>Second essay.</font>").encode()

    assert hashlib.sha256(first).digest() != hashlib.sha256(second).digest()
