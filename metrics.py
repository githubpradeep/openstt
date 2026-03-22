from __future__ import annotations

from typing import Iterable, Sequence


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    table = [[0] * cols for _ in range(rows)]

    for row in range(rows):
        table[row][0] = row
    for col in range(cols):
        table[0][col] = col

    for row in range(1, rows):
        for col in range(1, cols):
            cost = 0 if reference[row - 1] == hypothesis[col - 1] else 1
            table[row][col] = min(
                table[row - 1][col] + 1,
                table[row][col - 1] + 1,
                table[row - 1][col - 1] + cost,
            )
    return table[-1][-1]


def cer_stats(predictions: Iterable[str], references: Iterable[str]) -> tuple[int, int]:
    edits = 0
    total = 0
    for prediction, reference in zip(predictions, references):
        edits += edit_distance(list(reference), list(prediction))
        total += max(len(reference), 1)
    return edits, total


def wer_stats(predictions: Iterable[str], references: Iterable[str]) -> tuple[int, int]:
    edits = 0
    total = 0
    for prediction, reference in zip(predictions, references):
        ref_words = reference.split()
        hyp_words = prediction.split()
        edits += edit_distance(ref_words, hyp_words)
        total += max(len(ref_words), 1)
    return edits, total


def error_rate(edits: int, total: int) -> float:
    return float(edits) / float(max(total, 1))
