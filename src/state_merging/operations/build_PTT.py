"""Prefix Tree Transducers (PTT) implementation.

This module provides functions to build prefix tree transducers,
which are finite state transducers organized as tries. PTTs accumulate data values
along input paths. The implementation allows flexible data accumulation through three customizable
operations: incrementation on revisits, insertion on first occurrence, and
contribution when merging values at shared final states.

Type Parameters:
    Q: State type
    U: Input symbol type
    V: Output value type
    D: Output component of data pair type
"""
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, MutableMapping, MutableSet, Sequence, Set, TypeVar

from state_merging.automata.SFST import SFST

Q = TypeVar('Q')
U = TypeVar('U')
V = TypeVar('V')
D = TypeVar('D')


@dataclass
class PTTResults:
    """Results of building a prefix tree transducer."""
    sample_count: int = field(default=0)
    sample_input_size: int = field(default=0)
    sample_output_size: int = field(default=0)
    ptt_state_count: int = field(default=0)


def insert_data_PTT(
    fst: SFST[Q, U, V],
    results: PTTResults,
    data: tuple[Sequence[U], D],
    data_len: Callable[[D], int],
    epsilon: V,
    incr: Callable[[V], V],
    insertion: Callable[[D], V],
    contribute: Callable[[V, D], V],
    state_supply: Iterator[Q]
) -> None:
    """Insert a single (input sequence, data output value) pair into a prefix tree transducer.

    This function modifies the transducer in-place by creating or following a path
    through the trie corresponding to the input sequence, accumulating the data value
    at the terminal state using customizable operations.

    Args:
        fst: The prefix tree transducer to modify (modified in-place).
        data: Tuple of (input_sequence, data_value) to insert.
        epsilon: The identity/empty value for the output type V
                 (used to initialize new transitions).
        incr: Unary operation applied to outputs when a path is revisited.
              Used to track path frequency. Signature: V -> V
        insertion: Unary operation converting a data value into an output on first insertion.
                   Used to wrap/initialize the data. Signature: D -> V
        contribute: Binary operation to merge new data into existing output at a final state.
                    Used to accumulate values. Signature: (V, D) -> V
        state_supply: Iterator that generates fresh state identifiers for new trie nodes.
    """
    u, d = data

    results.sample_count += 1
    results.sample_input_size += len(u)
    results.sample_output_size += data_len(d)

    fst.initial_output = incr(fst.initial_output)

    q = fst.initial_state
    for c in u:
        try:
            q_, v = fst.transitions[(q, c)]
        except KeyError:
            q_, v = next(state_supply), epsilon
            fst.state_set.add(q_)
            results.ptt_state_count += 1
        fst.transitions[(q, c)] = q_, incr(v)
        q = q_

    try:
        v = fst.final_outputs[q]
    except KeyError:
        fst.final_outputs[q] = insertion(d)
    else:
        print("contribute")
        fst.final_outputs[q] = contribute(v, d)


def build_PTT(
    input_set: Set[U],
    dataset: Iterable[tuple[Sequence[U], D]],
    data_len: Callable[[D], int],
    epsilon: V,
    incr: Callable[[V], V],
    insertion: Callable[[D], V],
    contribute: Callable[[V, D], V],
    state_supply: Iterator[Q],
    empty_state_set: MutableSet[Q],
    empty_transition_mapping: MutableMapping[tuple[Q, U], tuple[Q, V]],
    empty_final_output_mapping: MutableMapping[Q, V]
) -> tuple[SFST[Q, U, V], PTTResults]:
    """Build a prefix tree transducer from a collection of (sequence, data) pairs.

    Constructs a trie-structured transducer that stores data values at terminal states.
    All input sequences in the dataset create paths through the transducer, with
    shared prefixes reusing states. Data values are accumulated at final states
    using the customizable accumulation operations.

    Args:
        input_set: The alphabet of input symbols that the PTT accepts.
        dataset: Collection of (input_sequence, data_value) pairs to insert into the trie.
                 Sequences with common prefixes will share trie paths.
        epsilon: The identity/empty value for output type V (initializes new transitions).
        incr: Unary operation applied when paths are revisited (e.g., increment counter).
              Signature: V -> V
        insertion: Unary operation to wrap/initialize data values on first insertion.
                   Signature: D -> V
        contribute: Binary operation to accumulate data values at shared final states.
                    Signature: (V, D) -> V
        state_supply: Iterator that generates fresh state identifiers for new trie nodes.
        empty_transition_mapping: An empty mapping for transitions. Allows the caller to
                                  optimize mapping type (e.g., DenseIntDict) while keeping
                                  the function generic.
        empty_final_output_mapping: An empty mapping for final outputs. Allows the caller to
                                    optimize mapping type while keeping the function generic.

    Returns:
        A new SFST[Q, U, V] representing the prefix tree structure with accumulated values.
    """
    q0 = next(state_supply)
    empty_state_set.add(q0)

    fst: SFST[Q, U, V] = SFST(
        state_set=empty_state_set,
        input_set=input_set,
        initial_state=q0,
        transitions=empty_transition_mapping,
        initial_output=epsilon,
        final_outputs=empty_final_output_mapping
    )

    results = PTTResults(ptt_state_count=1)

    for data in dataset:
        insert_data_PTT(
            fst,
            results,
            data,
            data_len,
            epsilon,
            incr,
            insertion,
            contribute,
            state_supply
    )

    return fst, results
