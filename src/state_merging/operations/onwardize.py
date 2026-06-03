"""Onwardization of finite state transducers.

This module implements the onwardization algorithm, which transforms subsequential
finite state transducers (SFSTs) into a canonical form by pushing outputs forward
through the state machine.

Type Parameters:
    Q: State type
    U: Input symbol type
    V: Output value type
    T: Result of LCP type
"""
from typing import Callable, Collection, MutableMapping, MutableSet, TypeVar

from state_merging.automata.SFST import SFST
from state_merging.operations.push_output import push_forward

Q = TypeVar('Q')
U = TypeVar('U')
V = TypeVar('V')
T = TypeVar('T')


def onwardize_trim_acyclic(
    fst: SFST[Q, U, V],
    rmul: Callable[[V, T], V],
    ldiv: Callable[[T, V], V],
    lcp: Callable[[Collection[V]], T],
    empty_set: MutableSet[Q],
    default_populated_mapping: MutableMapping[Q, list[tuple[Q, U]]]
) -> None:
    """Onwardize a trim, acyclic SFST by pushing outputs forward through states.

    This function modifies the SFST in-place by computing a common prefix of local outputs
    at each state and pushing that prefix forward to transitions entering the state.
    The states must be processed in a reverse topological order.

    The caller must provide the mapping implementations because constructing them internally
    would require the function to specialize on concrete types, breaking its generic nature.

    Args:
        fst: The trim, acyclic SFST to onwardize (modified in-place).
        rmul: Binary right multiplication operation for outputs (e.g., concatenation).
              Combines existing output with prefix: rmul(output, prefix).
        ldiv: Binary left division operation for outputs (e.g., string difference).
              Removes prefix from output: ldiv(prefix, output).
        lcp: Function to compute the longest common prefix from a set of outputs.
             Returns the longest prefix common to all outputs in the set.
        default_populated_mapping: A pre-initialized mutable mapping with empty lists
                                   for each state in fst.state_set. This mapping will be
                                   populated with incoming transitions and is used during
                                   onwardization. The caller is responsible for initialization.
    """
    fst.accumulate_ingoing_transitions(ingoing_trs := default_populated_mapping)
    for q in fst.iter_accessible_states_from(fst.initial_state, empty_set):
        pref = lcp(list(fst.iter_outgoing_from(q)))
        push_forward(fst, q, pref, rmul, ldiv, ingoing_trs)
