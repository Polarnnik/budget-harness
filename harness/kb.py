"""Knowledge-base cards as a dict. Imported by generated programs via the prelude.

A card is a text file with a `# Title` line and `Field: value` lines. Programs
address it as kb['Title']['Field'], so a multi-hop question costs the model a
single short expression and the answer is the exact string from the data.
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path

CARD_EXT = {".md", ".txt"}
FIELD = re.compile(r"^\s*(?:[-*]\s+)?\**([A-Za-z][\w /-]{0,40}?)\**\s*:\s*(\S.*?)\s*$")


def _norm(s) -> str:
    return " ".join(str(s).casefold().split())


def _value(raw: str):
    try:
        return int(raw) if str(int(raw)) == raw else raw
    except ValueError:
        return raw


class _Lenient(dict):
    """Dict that forgives case and spacing in keys and explains a miss."""

    what = "key"

    def __missing__(self, key):
        want = _norm(key)
        for k in self:
            if _norm(k) == want:
                return self[k]
        near = difflib.get_close_matches(str(key), [str(k) for k in self], n=3, cutoff=0.75)
        raise KeyError(f"no {self.what} {key!r}" + (f"; similar: {near}" if near else ""))

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        return dict.__contains__(self, key) or any(_norm(k) == _norm(key) for k in self)


class Card(_Lenient):
    what = "field"


class KB(_Lenient):
    what = "card titled"


def parse_card(text: str, fallback_title: str) -> tuple[str, Card]:
    title, fields = None, Card()
    for line in text.splitlines():
        if title is None and line.startswith("#"):
            title = line.lstrip("#").strip()
            continue
        m = FIELD.match(line)
        if m:
            fields.setdefault(m[1].strip(), _value(m[2]))
    return title or fallback_title, fields


def load_kb(root: str = ".") -> KB:
    kb = KB()
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() in CARD_EXT and p.is_file():
            title, fields = parse_card(p.read_text(encoding="utf-8", errors="replace"), p.stem)
            kb.setdefault(title, fields)   # first file wins on a duplicate title
    return kb
