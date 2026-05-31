"""Grapheme-to-phoneme (G2P) transducer learner using OSTIA algorithm.

This module applies the OSTIA (Onward Subsequential Transducers Inference Algorithm)
to learn a grapheme-to-phoneme mapping from the CMU Dictionary. The learned transducer
maps English words (grapheme sequences) to their phonetic pronunciations (phoneme sequences).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import random
from itertools import count
from typing import cast

import data.cmudict as cmudict
from algorithms.ostia import ostia
from automata.SFST import SFST


def learn_cmudict(seed: int | None = None) -> SFST[int, str, list[str]]:
    """Learn a grapheme-to-phoneme transducer from CMU Dictionary using OSTIA.

    Applies the OSTIA (Onward Subsequential Transducers Inference Algorithm) to
    infer a subsequential finite-state transducer from the CMU Dictionary's English
    word-to-phoneme mappings.

    Algorithm Configuration:
        - Input alphabet: All unique graphemes (letters, punctuation) in CMU Dict
        - Output: Phoneme sequences (lists of ARPABET phoneme symbols)
        - Selection of next state to attempt to merge: Random choice
        - Search order for compatible states: Random permutation
        - State supply: Infinite sequence of ints (count() from itertools)

    Args:
        seed: Optional random seed for reproducible learning. If provided,
            random.seed(seed) is called to ensure deterministic results.
            If None (default), learning uses random behavior (different results
            each run).

    Returns:
        An SFST[int, str, list[str]] where:
        - States are identified by integers
        - Input labels are single characters (graphemes)
        - Output values are lists of phoneme strings
    """
    if seed is not None:
        random.seed(seed)

    return cast(SFST[int, str, list[str]], ostia(
        input_set=cmudict.input_set,
        dataset=cmudict.make_deterministic_sample(),
        epsilon=[],
        concat=lambda v1, v2: cast(list[str], v1) + cast(list[str], v2),
        choose_transition=lambda _, trs: random.sample(trs, k=1)[0],
        search_iter=lambda _, qs: random.sample(qs, k=len(qs)),
        state_supply=count()
    ))
