"""Cheap review hints, never a semantic validator or an output filter."""
from __future__ import annotations

import unicodedata


def expansion_hint(glosses, text):
    """Flag a large expansion from one observed word; retain native text.

    Character counts are a UI review heuristic, not correctness probabilities.
    Multi-word translations and unflagged strings remain unassessed.
    """
    if len(glosses) != 1 or not text:
        return None
    def units(value):
        return sum(unicodedata.category(char)[0] in {'L','N'}
                   for char in unicodedata.normalize('NFKC',value))
    word_units, text_units = units(glosses[0]), units(text)
    if word_units and text_units > max(4,word_units*2):
        return dict(version=1, reason='single_word_expansion', gloss_count=1,
                    gloss_units=word_units, text_units=text_units)
    return None
