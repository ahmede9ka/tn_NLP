"""Text normalization and corpus-level speech evaluation helpers."""

from __future__ import annotations

import unicodedata


def minimally_normalize(text: str) -> str:
    """Apply whitespace and Unicode NFC normalization only."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def normalized_for_scoring(text: str) -> str:
    """Normalize Unicode, punctuation, and whitespace without rewriting dialect words."""
    text = unicodedata.normalize("NFC", text).casefold()
    text = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in text
    )
    return " ".join(text.split())


def error_rates(references: list[str], predictions: list[str]) -> dict[str, float]:
    """Return corpus WER/CER using edit distance and explicit empty-reference guards."""
    if len(references) != len(predictions):
        raise ValueError(
            f"Reference/prediction length mismatch: {len(references)} != {len(predictions)}"
        )
    if not references:
        raise ValueError("Cannot score an empty evaluation set")

    word_errors = word_count = char_errors = char_count = 0
    for reference, prediction in zip(references, predictions):
        ref_words = minimally_normalize(reference).split()
        pred_words = minimally_normalize(prediction).split()
        word_errors += _levenshtein(ref_words, pred_words)
        word_count += len(ref_words)
        ref_chars = "".join(minimally_normalize(reference).split())
        pred_chars = "".join(minimally_normalize(prediction).split())
        char_errors += _levenshtein(list(ref_chars), list(pred_chars))
        char_count += len(ref_chars)
    if not word_count or not char_count:
        raise ValueError("References must contain at least one word and one non-whitespace character")
    return {
        "wer": word_errors / word_count,
        "cer": char_errors / char_count,
    }


def _levenshtein(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]
