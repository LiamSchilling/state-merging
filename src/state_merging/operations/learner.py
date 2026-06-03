"""Learning finite-state transducers by state merging.

This module implements the core learning algorithm that constructs finite-state transducers
from sample data through state merging. The process involves three main stages:
1. Building a prefix tree transducer (PTT) from the dataset
2. Onwardizing the PTT to a canonical form
3. Iteratively merging states based on provided compatibility operations

Type Parameters:
    Q: State type
    U: Input symbol type
    V: Output value type
    V_: Final output value type (may differ from V after application of the continuation)
    D: Output component of data pair type
    T: Result of LCP type
"""
import time
from dataclasses import dataclass, field
from typing import (
    Callable,
    Collection,
    Iterable,
    Iterator,
    MutableMapping,
    MutableSet,
    Sequence,
    Set,
    TypeVar,
)

from state_merging.automata.SFST import SFST
from state_merging.operations.build_PTT import PTTResults, build_PTT
from state_merging.operations.onwardize import onwardize_trim_acyclic
from state_merging.operations.state_merging import iterate_merge, merge

Q = TypeVar('Q')
U = TypeVar('U')
V = TypeVar('V')
V_ = TypeVar('V_')
D = TypeVar('D')
T = TypeVar('T')


@dataclass
class MergeResults(PTTResults):
    """Results of a state-merging learner."""
    merged_state_count: int = field(default=0)
    result_state_count: int = field(default=0)
    build_ptt_time: float = field(default=0.0)
    merge_states_time: float = field(default=0.0)
    postprocess_time: float = field(default=0.0)

    @classmethod
    def from_ptt_results(cls, ptt_results: PTTResults) -> 'MergeResults':
        """Create MergeResults from PTTResults with default values for merge fields."""
        return cls(**vars(ptt_results))


def learn_by_state_merging(
    input_set: set[U],
    dataset: Iterable[tuple[Sequence[U], D]],
    data_len: Callable[[D], int],
    epsilon: V,
    incr: Callable[[V], V],
    insertion: Callable[[D], V],
    contribute: Callable[[V, D], V],
    lmul: Callable[[T, V], V],
    rmul: Callable[[V, T], V],
    ldiv: Callable[[T, V], V],
    lcp: Callable[[Collection[V]], T],
    try_unify: Callable[[V, V], tuple[V, T] | None],
    is_epsilon: Callable[[T], bool],
    check_merge: Callable[[SFST[Q, U, V]], bool],
    choose_transition: Callable[[SFST[Q, U, V], Set[Q]], Q],
    search_iter: Callable[[SFST[Q, U, V], Set[Q]], Iterable[Q]],
    state_supply: Iterator[Q],
    postprocess: Callable[[SFST[Q, U, V]], SFST[Q, U, V_]],
    empty_fst_state_set: MutableSet[Q],
    empty_transition_mapping: MutableMapping[tuple[Q, U], tuple[Q, V]],
    empty_final_output_mapping: MutableMapping[Q, V],
    make_empty_visited_state_set: Callable[[MutableSet[Q]], MutableSet[Q]],
    make_default_populated_mapping: Callable[[MutableSet[Q]], MutableMapping[Q, list[tuple[Q, U]]]],
    verbose: bool = False
) -> tuple[SFST[Q, U, V_], MergeResults]:
    """Learn an SFST from training data through prefix-tree construction and state merging.

    This is the main entry point for learning FSTs.
    paradigm: build a prefix tree from the data, normalize it, then iteratively merge
    compatible states based on provided search and merge strategies.

    Args:
        input_set: The set of possible input symbols.
        dataset: Collection of (input_sequence, output_data) pairs to learn from.
        epsilon: Identity element for output values (acts as the empty output).
        incr: Increment function called when revisiting a transition with new data.
        insertion: Function converting output data to output values on first visit.
        contribute: Function combining an accumulated value with new data at final states.
        lmul: Left multiply - applies remainder to left of output value.
        rmul: Right multiply - applies remainder to right of output value.
        ldiv: Left divide - removes remainder from left of output value.
        lcp: Longest common prefix function for outputs.
        try_unify: Attempts to unify two output values, returning unified value and
                   remainder suffixes, or None on conflict.
        is_epsilon: Predicate to check if a value is epsilon.
        check_merge: Validation function that checks if a tentative merge should be accepted.
                     Called after merge succeeds; returns True to accept, False to reject.
        choose_transition: Heuristic selecting which transition to process from the frontier.
        search_iter: Heuristic yielding candidate states to try merging with.
        state_supply: Iterator providing fresh state identifiers as needed.
        postprocess: Function to transform final transducer (e.g., normalize outputs).
        empty_fst_state_set: An empty set for FST states passed to build_PTT.
                             States are guaranteed to be inserted in the order provided by
                             state_supply.
        empty_transition_mapping: An empty mapping for transitions passed to build_PTT.
                                  Allows optimizing mapping type (e.g., DenseIntDict)
                                  while keeping the function generic.
        empty_final_output_mapping: An empty mapping for final outputs passed to build_PTT.
                                    Allows optimizing mapping type while keeping the
                                    function generic.
        make_empty_visited_state_set: Factory creating an empty set for visited states during
                                      onwardization. Takes the total state set as argument to
                                      allow optimizations based on state set size.
        make_default_populated_mapping: Factory creating a mapping from states to incoming
                                        transition lists. Allows optimizing mapping type
                                        (e.g., DenseIntDict) while keeping the function generic.
        verbose: Whether to print progress information during learning.

    Returns:
        A learned SFST of type SFST[Q, U, V_].

    Algorithm:
        1. Builds a prefix tree transducer from the dataset
        2. Onwardizes it to canonical form with outputs pushed forward
        3. Iteratively merges compatible states
        4. Applies an output-modifying continuation to the final transducer
    """
    if verbose:
        print(
            "learning from the following positive data by state-merging: [\n\t" +
            "\n\t".join(f"{u}, {d}" for u, d in dataset) +
            "\n]\n"
        )

    build_ptt_start = time.perf_counter()

    fst: SFST[Q, U, V]
    fst, ptt_results = build_PTT(
        input_set,
        dataset,
        data_len,
        epsilon,
        incr,
        insertion,
        contribute,
        state_supply,
        empty_fst_state_set,
        empty_transition_mapping,
        empty_final_output_mapping
    )

    results = MergeResults.from_ptt_results(ptt_results)

    if verbose:
        print(f"naively constructed PTT:\n{fst}\n")

    onwardize_trim_acyclic(
        fst,
        rmul,
        ldiv,
        lcp,
        make_empty_visited_state_set(fst.state_set),
        make_default_populated_mapping(fst.state_set)
    )

    if verbose:
        print(f"onwardized PTT:\n{fst}\n")

    build_ptt_end = time.perf_counter()
    results.build_ptt_time = build_ptt_end - build_ptt_start

    merge_states_start = time.perf_counter()

    def try_merge(
        fst_: SFST[Q, U, V],
        q_src: Q,
        q_dest: Q,
        tr_src_ingoing: tuple[Q, U] | None
    ) -> bool:
        match merge(
            fst_,
            q_src,
            q_dest,
            tr_src_ingoing,
            lmul,
            ldiv,
            try_unify,
            is_epsilon,
            verbose=verbose
        ):
            case None:
                return False
            case undo_merge:
                if check_merge(fst_):
                    return True
                else:
                    undo_merge()
                    return False

    iterate_merge(fst, try_merge, choose_transition, search_iter, verbose=verbose)
    results.merged_state_count = len(fst.state_set)

    if verbose:
        print(f"FST after state-merging:\n{fst}\n")

    merge_states_end = time.perf_counter()
    results.merge_states_time = merge_states_end - merge_states_start

    postprocess_start = time.perf_counter()

    fst_: SFST[Q, U, V_] = postprocess(fst)
    results.result_state_count = len(fst_.state_set)

    if verbose:
        print(f"result FST of learning algorithm:\n{fst_}\n")

    postprocess_end = time.perf_counter()
    results.postprocess_time = postprocess_start - postprocess_end

    if verbose:
        print(f"collected meta results:\n{results}")

    return fst_, results
