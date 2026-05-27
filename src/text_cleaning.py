from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser


class _VisibleTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if data:
            self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def strip_html(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""

    if "<" not in value and ">" not in value:
        return unescape(value)

    parser = _VisibleTextExtractor()
    parser.feed(value)
    parser.close()
    return unescape(parser.get_text())


def clean_news_text(value: str) -> str:
    text = strip_html(value)
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text