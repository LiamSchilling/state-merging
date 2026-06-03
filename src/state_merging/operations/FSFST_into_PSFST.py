"""Conversion from Frequency SFST to Probabilistic SFST.

This module provides functionality to convert a frequency-annotated finite state
transducer (FSFST) to a probabilistically-annotated finite state transducer
(PSFST) by normalizing frequency counts to probabilities.

Type Parameters:
    Q: State type
    U: Input symbol type
    V: Output value type
"""
from typing import MutableMapping, TypeVar, cast

from state_merging.automata.FSFST import FSFST, accumulate_outgoing_frequencies
from state_merging.automata.PSFST import PSFST

Q = TypeVar('Q')
U = TypeVar('U')
V = TypeVar('V')


def FSFST_into_PSFST(
    fst: FSFST[Q, U, V],
    zero_populated_freqs: MutableMapping[Q, int]
) -> PSFST[Q, U, V]:
    """Convert FSFST to PSFST by normalizing frequencies to probabilities.

    Transforms a frequency-annotated SFST into a probabilistically-annotated
    SFST by dividing each frequency count by the total outgoing frequency from
    its source state. This creates a proper probability distribution where
    outgoing probabilities from each state sum to 1.0.

    Note:
        The input fst is modified in place and effectively consumed by this function.

    Args:
        fst: A frequency-annotated SFST to convert.
        zero_populated_freqs: A mutable mapping from states to 0 that will be populated with
            cumulative outgoing frequencies for each state. The caller is responsible for
            providing a mapping pre-populated with all states mapped to 0.

    Returns:
        A probabilistically-annotated SFST.

    Raises:
        ZeroDivisionError: If any state with an outgoing transition has outgoing frequency of 0.
    """
    accumulate_outgoing_frequencies(fst, outgoing_freqs := zero_populated_freqs)

    v, _ = fst.initial_output
    fst.initial_output= v, 1.0 # type: ignore

    for (q, c), (q_, (v, f)) in fst.transitions.items():
        fst.transitions[(q, c)] = q_, (v, f / outgoing_freqs[q]) # type: ignore

    for q, (v, f) in fst.final_outputs.items():
        fst.final_outputs[q] = v, f / outgoing_freqs[q] # type: ignore

    return cast(PSFST[Q, U, V], fst)
