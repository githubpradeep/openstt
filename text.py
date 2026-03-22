from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


BLANK_TOKEN = "<blank>"
_UNSUPPORTED_PATTERN = re.compile(r"[^a-z' ]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = normalized.replace("-", " ")
    normalized = _UNSUPPORTED_PATTERN.sub(" ", normalized)
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


@dataclass
class CharTokenizer:
    token_to_id: Dict[str, int]

    @classmethod
    def build(cls, texts: Iterable[str]) -> "CharTokenizer":
        charset = sorted({char for text in texts for char in normalize_text(text)})
        token_to_id = {BLANK_TOKEN: 0}
        for char in charset:
            if char not in token_to_id:
                token_to_id[char] = len(token_to_id)
        return cls(token_to_id=token_to_id)

    @classmethod
    def from_file(cls, path: str | Path) -> "CharTokenizer":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(token_to_id=payload["token_to_id"])

    @property
    def blank_id(self) -> int:
        return self.token_to_id[BLANK_TOKEN]

    @property
    def id_to_token(self) -> Dict[int, str]:
        return {idx: token for token, idx in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def save(self, path: str | Path) -> None:
        payload = {
            "blank_token": BLANK_TOKEN,
            "token_to_id": self.token_to_id,
        }
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)

    def text_to_ids(self, text: str) -> List[int]:
        normalized = normalize_text(text)
        return [self.token_to_id[char] for char in normalized if char in self.token_to_id]

    def ids_to_text(self, ids: Iterable[int]) -> str:
        table = self.id_to_token
        return "".join(table[idx] for idx in ids if idx in table and idx != self.blank_id)

    def decode_ctc(self, token_ids: Iterable[int]) -> str:
        collapsed: List[int] = []
        previous = None
        for idx in token_ids:
            if idx == self.blank_id:
                previous = None
                continue
            if idx != previous:
                collapsed.append(idx)
            previous = idx
        return self.ids_to_text(collapsed)
