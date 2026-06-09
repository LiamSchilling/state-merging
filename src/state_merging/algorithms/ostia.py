"""Onward Subsequential Transducer Inference Algorithm (OSTIA).

Type Parameters:
    Q: State type
    U: Input symbol type
    V: Output symbol type
"""

from typing import Callable, Iterable, Iterator, MutableMapping, MutableSet, Sequence, Set, TypeVar

from state_merging.automata.SFST import SFST
from state_merging.operations.learner import MergeResults, learn_by_state_merging
from state_merging.util import lcp, ldiv, match, unify

Q = TypeVar('Q')
U = TypeVar('U')
V = TypeVar('V')


def ostia(
    input_set: set[U],
    dataset: Iterable[tuple[Sequence[U], Sequence[V]]],
    epsilon: Sequence[V],
    concat: Callable[[Sequence[V], Sequence[V]], Sequence[V]],
    choose_transition: Callable[[SFST[Q, U, Sequence[V]], Set[Q]], Q],
    search_iter: Callable[[SFST[Q, U, Sequence[V]], Set[Q]], Iterable[Q]],
    state_supply: Iterator[Q],
    empty_state_set: MutableSet[Q],
    empty_transition_mapping: MutableMapping[tuple[Q, U], tuple[Q, Sequence[V]]],
    empty_final_output_mapping: MutableMapping[Q, Sequence[V]],
    make_empty_state_set: Callable[[MutableSet[Q]], MutableSet[Q]],
    make_default_populated_mapping: Callable[[MutableSet[Q]], MutableMapping[Q, list[tuple[Q, U]]]],
    verbose: bool = False
) -> tuple[SFST[Q, U, Sequence[V]], MergeResults]:
    """Learn an SFST from input-output pairs using OSTIA.

    Constructs a minimal subsequential finite-state transducer that maps input sequences
    to output sequences through iterative state merging, with onwardization to ensure
    the result is in a canonical form.

    Args:
        input_set: The alphabet (set of input symbols).
        dataset: Collection of input-output sequence pairs for learning.
        epsilon: The empty/identity sequence.
        concat: Function for concatenating two sequences.
        choose_transition: Heuristic for selecting which frontier transition to process.
        search_iter: Heuristic for iterating through promoted states to try merging with.
        state_supply: Iterator providing fresh state identifiers as needed.
        empty_state_set: An empty set for FST states.
        empty_transition_mapping: An empty mapping for transitions.
        empty_final_output_mapping: An empty mapping for final outputs.
        make_empty_state_set: Factory creating an empty set for visited states during
                              onwardization. Takes the total state set as argument.
        make_default_populated_mapping: Factory creating a mapping from states to incoming
                                        transition lists.
        verbose: Whether to print progress information during learning.

    Returns:
        An SFST that transforms input sequences according to the learned transduction
        from the dataset.
    """
    return learn_by_state_merging(
        input_set=input_set,
        dataset=dataset,
        data_len=len,
        epsilon=epsilon,
        incr=lambda v: v,
        insertion=lambda v: v,
        contribute=match,
        lmul=concat,
        rmul=concat,
        ldiv=ldiv,
        lcp=lambda vs, epsilon=epsilon: lcp(vs, epsilon),
        try_unify=unify,
        is_epsilon=lambda v: len(v) == 0,
        check_merge=lambda _: True,
        choose_transition=choose_transition,
        search_iter=search_iter,
        state_supply=state_supply,
        empty_state_set=empty_state_set,
        empty_transition_mapping=empty_transition_mapping,
        empty_final_output_mapping=empty_final_output_mapping,
        make_empty_state_set=make_empty_state_set,
        make_default_populated_mapping=make_default_populated_mapping,
        postprocess=lambda dfa: dfa,
        verbose=verbose
    )
