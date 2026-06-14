"""Utilities to generalize transitions of an SFST by learning
decision trees from observed outgoing transitions.

This module exposes `generalize_transitions`, which takes a
sequential finite-state transducer (SFST) and, for every state,
learns two decision trees:

- a destination decision tree that maps an input label to the
    next state, and
- an output decision tree that maps an input label to the output
    symbol (or final output when the input is ``None``).

The learned trees are then used to populate the FST's transition
function (`fst.transitions`) and final outputs (`fst.final_outputs`).

Internal helpers compute simple entropy and consolidation metrics
used as impurity/label selectors when inferring decision trees.

Type Parameters:
    C: Classifier label type used by the decision tree learner.
    Q: State type.
    U: Input symbol type.
    V: Output value type.
"""

import math
from typing import Callable, Collection, Iterable, Iterator, MutableMapping, TypeVar

from decision_tree_inference.decision_trees.decision_tree import DecisionTree, decide
from decision_tree_inference.operations.inference import infer_by_iterative_splitting
from state_merging.automata.SFST import SFST

C = TypeVar('C')
Q = TypeVar('Q')
U = TypeVar('U')
V = TypeVar('V')


def _entropy(dataset: Iterable[Q]) -> float:
    """Compute the Shannon entropy (base 2) of the values in dataset.

    The function consumes the provided iterable and computes the empirical
    distribution of the labels to produce the entropy in bits.

    Args:
        dataset: An iterable of class labels. The iterable is consumed by
                 the function.

    Returns:
        The entropy in bits as a float.

    Raises:
        ZeroDivisionError: If ``dataset`` contains no elements (normalization
                           by zero). Callers should avoid passing empty collections.
    """
    total_count = 0
    counts: dict[Q, int] = {}

    for d in dataset:
        total_count += 1
        counts[d] = counts.get(d, 0) + 1

    entropy = 0.0
    for c in counts.values():
        p = c / total_count
        entropy -= p * math.log2(p)

    return entropy


def _consolidate(dataset: Iterable[Q]) -> Q | None:
    """Return a single consolidated label if all items agree.

    Args:
        dataset: An iterable of labels. The iterable is consumed by this
                 function.

    Returns:
        The unanimous label if every element in ``dataset`` equals the
        first element; otherwise ``None``.

    Raises:
        ValueError: If ``dataset`` contains no elements.
    """
    dataset_iterator: Iterator[Q] = iter(dataset)

    try:
        d = next(dataset_iterator)
    except StopIteration:
        raise ValueError("encountered class split containing no data points")

    if all(d_ == d for d_ in dataset_iterator):
        return d
    else:
        return None


def generalize_transitions(
    fst: SFST[Q, U, V],
    num_labels_per_class: MutableMapping[C, int],
    classify: Callable[[U | None, C], int],
    mean_impurity: Callable[[Collection[tuple[U | None, V]]], float],
    consolidate: Callable[[Collection[tuple[U | None, V]]], V | None]
) -> None:
    """Learn and write generalized transitions for every state in ``fst``.

    For each state ``q`` in the transducer this function:
    1. Collects observed outgoing transitions as datasets for destination
       states and outputs (including the state's final output when
       present).
    2. Infers two decision trees using iterative splitting:
       - a destination tree predicting the next state, and
       - an output tree predicting emitted outputs.
    3. Populates ``fst.transitions`` for every input symbol and sets
       ``fst.final_outputs[q]`` from the output tree's prediction for
       input ``None``.

    Args:
        fst: The sequential finite-state transducer to update in-place.
        num_labels_per_class: Mapping from class identifiers to the
                              maximum number of labels allowed for that class (used by the
                              decision tree learner).
        classify: A function mapping an input label and class id to a
                  numeric label index for the decision tree learner.
        mean_impurity: Impurity function used when learning output
                       decision trees. It receives a collection of ``(input, output)``
                       pairs.
        consolidate: Function that returns a consolidated output value
                     when the outputs in a split are unanimous, otherwise ``None``.
    """
    for q in fst.state_set:
        destination_dataset: list[tuple[U, Q]]
        output_dataset: list[tuple[U | None, V]]
        destination_dataset = [(u, q) for u, q, _ in fst.iter_outgoing_states_from(q)]
        output_dataset = [(u, v) for u, _, v in fst.iter_outgoing_states_from(q)]

        try:
            v = fst.final_outputs[q]
        except KeyError:
            pass
        else:
            output_dataset.append((None, v))

        destination_dt: DecisionTree[C, Q] = infer_by_iterative_splitting(
            num_labels_per_class,
            destination_dataset,
            classify,
            lambda dataset: _entropy(q for _, q in dataset),
            lambda dataset: _consolidate(q for _, q in dataset)
        )

        output_dt: DecisionTree[C, V] = infer_by_iterative_splitting(
            num_labels_per_class,
            output_dataset,
            classify,
            mean_impurity,
            consolidate
        )

        for u in fst.input_set:
            fst.transitions[(q, u)] = (
                decide(u, classify, destination_dt),
                decide(u, classify, output_dt)
            )

        fst.final_outputs[q] = decide(None, classify, output_dt)
