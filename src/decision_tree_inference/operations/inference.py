"""Decision-tree inference operations.

Generic utilities for inferring decision trees by greedy iterative
splitting. The implementation follows the ID3-style greedy splitting
strategy: at each internal node the algorithm selects the class
that minimises a weighted impurity measure (equivalently maximises
information gain for an entropy-based impurity). Using Shannon entropy
as the impurity function recovers the ID3 algorithm's information-gain
split criterion.

Callers supply a classification function, an impurity metric and a
consolidation predicate that decides when a node becomes a leaf.

Type parameters:
    C: Class / split identifier type.
    U: Input instance type supplied to the classifier.
    V: Output / label value type stored at leaves.
    D: Datum type stored alongside inputs in the dataset.
"""
from typing import Callable, Collection, Mapping, MutableMapping, TypeVar

from decision_tree_inference.decision_trees.decision_tree import DecisionTree
from decision_tree_inference.util import Leaf, Node, Tree

C = TypeVar('C')
U = TypeVar('U')
V = TypeVar('V')
D = TypeVar('D')


def split_dataset_on_class(
    num_labels_per_class: Mapping[C, int],
    split: C,
    dataset: Collection[tuple[U, D]],
    classify: Callable[[U, C], int]
) -> Collection[Collection[tuple[U, D]]]:
    """Partition a dataset by labels for a chosen class.

    The function allocates one bucket for each label index in
    ``range(num_labels_per_class[split])`` and appends each `(input, datum)`
    pair to the bucket indicated by ``classify(input, split)``.

    Args:
        num_labels_per_class: Mapping from class to number of labels.
        split: Class identifier to partition on.
        dataset: Iterable of pairs `(input, datum)`.
        classify: Function `(input, split) -> int` returning a label index.

    Returns:
        A sequence of buckets (one per label index). Buckets may be empty.
    """
    num_labels = num_labels_per_class[split]

    data_per_label: dict[int, list[tuple[U, D]]] = {l: [] for l in range(num_labels)}
    for u, d in dataset:
        data_per_label[classify(u, split)].append((u, d))

    split_datasets: list[list[tuple[U, D]]] = []
    for l in range(num_labels):
        split_datasets.append(data_per_label[l])

    return split_datasets


def choose_split_of_min_impurity(
    num_labels_per_class: Mapping[C, int],
    dataset: Collection[tuple[U, D]],
    classify: Callable[[U, C], int],
    mean_impurity: Callable[[Collection[tuple[U, D]]], float],
) -> tuple[float, C, Collection[Collection[tuple[U, D]]]]:
    """Choose the class split with minimal weighted impurity.

    For each candidate class the dataset is partitioned and the total weighted
    impurity is computed as the sum of ``len(bucket) * mean_impurity(bucket)``.

    Args:
        num_labels_per_class: Mapping from class to number of labels.
        dataset: Iterable of `(input, datum)` pairs.
        classify: Function `(input, split) -> int` returning a label index.
        mean_impurity: Function computing the mean impurity of a bucket.

    Returns:
        A tuple ``(min_impurity, min_class, split_datasets)`` describing the
        best split found.

    Raises:
        ValueError: If no candidate class is available to split on.
    """
    min_impurity = float('inf')
    min_class: C | None = None
    min_split: Collection[Collection[tuple[U, D]]] | None = None

    for c in num_labels_per_class:
            split_datasets: Collection[Collection[tuple[U, D]]] = split_dataset_on_class(
                num_labels_per_class,
                c,
                dataset,
                classify
            )

            total_impurity = 0.0
            for split_dataset in split_datasets:
                total_impurity += len(split_dataset) * mean_impurity(split_dataset)

            if total_impurity < min_impurity:
                min_impurity = total_impurity
                min_class = c
                min_split = split_datasets

    if min_class is None or min_split is None:
        raise ValueError("found no remaining classes on which to split")
    else:
        return min_impurity, min_class, min_split


def infer_tree_by_iterative_splitting(
    num_labels_per_class: MutableMapping[C, int],
    dataset: Collection[tuple[U, D]],
    classify: Callable[[U, C], int],
    mean_impurity: Callable[[Collection[tuple[U, D]]], float],
    consolidate: Callable[[Collection[tuple[U, D]]], V | None]
) -> Tree[C, V]:
    """Recursively build a `Tree` by greedy iterative splitting.

    The procedure implements an ID3-like greedy split: ``choose_split_of_min_impurity``
    selects the class that minimises the weighted impurity. In particular,
    when Shannon entropy is used as ``mean_impurity``, the selection is
    equivalent to ID3's information-gain split criterion. ``consolidate``
    decides whether to stop and return a leaf value or to split further.

    Args:
        num_labels_per_class: Mutable mapping from class to number of labels.
        dataset: Iterable of `(input, datum)` pairs.
        classify: Function `(input, split) -> int` returning a label index.
        mean_impurity: Function computing mean impurity for a bucket.
        consolidate: Function returning a leaf value or ``None`` to continue.

    Returns:
        A ``Node`` when split, or a ``Leaf`` when consolidated.
    """
    match consolidate(dataset):
        case None:
            split_datasets: Collection[Collection[tuple[U, D]]]
            _, c, split_datasets = choose_split_of_min_impurity(
                num_labels_per_class,
                dataset,
                classify,
                mean_impurity
            )

            num_labels = num_labels_per_class.pop(c)

            tree = Node(c, [
                infer_tree_by_iterative_splitting(
                    num_labels_per_class,
                    split_dataset,
                    classify,
                    mean_impurity,
                    consolidate
                )
                for split_dataset in split_datasets
            ])

            num_labels_per_class[c] = num_labels

            return tree

        case v:
            return Leaf(v)


def infer_by_iterative_splitting(
    num_labels_per_class: MutableMapping[C, int],
    dataset: Collection[tuple[U, D]],
    classify: Callable[[U, C], int],
    mean_impurity: Callable[[Collection[tuple[U, D]]], float],
    consolidate: Callable[[Collection[tuple[U, D]]], V | None]
) -> DecisionTree[C, V]:
    """Build and return a `DecisionTree` using iterative splitting.

    This is a thin wrapper that constructs a `DecisionTree` object from the
    root `Tree` produced by `infer_tree_by_iterative_splitting`.
    """
    return DecisionTree(num_labels_per_class, infer_tree_by_iterative_splitting(
        num_labels_per_class,
        dataset,
        classify,
        mean_impurity,
        consolidate
    ))
