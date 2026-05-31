from typing import Iterator

"""CMU Dictionary data utilities for grapheme-to-phoneme (G2P) learning.

References:
    CMU Pronouncing Dictionary: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
"""
import cmudict

input_set: set[str] = {c for k in cmudict.dict() for c in k}
"""Set of all unique graphemes (input characters) in the CMU Dictionary.

This represents all distinct characters that appear in English words
in the CMU Dictionary. Used as the input alphabet for G2P transducers.
"""

output_set: set[str] = {c for _, v in cmudict.entries() for c in v}
"""Set of all unique phonemes (output characters) in the CMU Dictionary.

This represents all distinct ARPABET-based symbols used in phonetic transcriptions.
Phonemes are represented as letters (e.g., 'AH', 'S', 'T') with optional numeric
stress markers (0, 1, 2). Used as the output alphabet for G2P transducers.
"""


def make_deterministic_sample() -> Iterator[tuple[str, list[str]]]:
    """Create a sample of deterministic G2P mappings from CMU Dictionary.

    For each word in the CMU Dictionary that has at least one pronunciation,
    extracts the first listed pronunciation. This includes words with multiple
    pronunciations (e.g., "read" has two pronunciations), but only uses the
    primary one listed in the dictionary.

    Yields:
        (word, phoneme_sequence) tuples. The phoneme_sequence is the
        first pronunciation for each word.
        Example: ('hello', ['HH', 'AH0', 'L', 'OW1'])
    """
    for k, v in cmudict.dict().items():
        if v != []:
            yield k, v[0]
