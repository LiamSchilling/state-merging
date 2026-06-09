"""Decision tree data structures and inference utilities.

This module defines a simple decision-tree data structure and
basic operations for validating and performing decisions.

Type parameters:
    C: Class / split identifier type used at internal nodes.
    U: Input instance type supplied to the classifier.
    V: Output / label value type stored at leaves.
"""
from dataclasses import dataclass
from typing import Callable, Generic, Mapping, TypeVar

from decision_tree_inference.util import Leaf, Node, Tree

C = TypeVar('C')
U = TypeVar('U')
V = TypeVar('V')


@dataclass
class DecisionTree(Generic[C, V]):
    """A decision tree with class label cardinality metadata.

    Attributes:
        num_labels_per_class: Mapping from class/split identifiers to the
                              integer number of labels available for that split. This mapping is
                              used to validate branching arities at internal nodes.
        tree: The actual tree structure.
    """
    num_labels_per_class: Mapping[C, int]
    tree: Tree[C, V]


@dataclass
class DecisionTreeMatchingError(Exception, Generic[C, V]):
    """Raised when pattern-matching on a `DecisionTree` instance fails.

    The `tree` attribute contains the offending `DecisionTree` instance and can
    be used by callers to report diagnostic information.
    """
    tree: DecisionTree[C, V]

    def __str__(self) -> str:
        return f"attempted to pattern-match on ill-formed decision tree {self.tree!r}"


def assert_DecisionTree(dt: DecisionTree[C, V]) -> None:
    """Assert that a `DecisionTree` is well-formed.

    Checks the following invariants recursively:
      - A `Leaf` is always valid.
      - A `Node(split, children)` must have `len(children)` equal to the
        value `num_labels_per_class[split]`.
      - All child subtrees must themselves satisfy these invariants.

    Args:
        dt: The `DecisionTree` to validate.

    Raises:
        AssertionError: If any invariant is violated.
        DecisionTreeMatchingError: If the provided value is not a `DecisionTree`.
    """
    match dt:
        case DecisionTree(num_labels_per_class, Leaf(_)):
            return

        case DecisionTree(num_labels_per_class, Node(split, children)):
            num_children = len(children)
            num_labels = num_labels_per_class[split]

            assert num_children == num_labels, \
                f"number of children {num_children} does not match class split {split} " \
                f"with {num_labels} labels"

            for dt_ in children:
                assert_DecisionTree(DecisionTree(num_labels_per_class, dt_))

        case _:
            raise DecisionTreeMatchingError(dt)


def decide(u: U, classify: Callable[[U, C], int], dt: DecisionTree[C, V]) -> V:
    """Apply a decision tree to an input instance using a classifier.

    The `classify` callback is responsible for returning an integer label for a
    given input instance `x` and a split identifier `C`. That integer is used to
    select the next child branch when traversing `Node` values. Leaves return
    their stored value.

    Args:
        x: The input instance to classify / decide on.
        classify: A function with signature `(x: U, split: C) -> int` that
                  returns a label index in `[0, num_labels_per_class[split])`.
        dt: The `DecisionTree` to evaluate.

    Returns:
        The value `V` stored at the reached leaf.

    Raises:
        ValueError: If `classify` returns an out-of-range label for a split.
        DecisionTreeMatchingError: If `dt` is not a valid `DecisionTree`.
    """
    match dt:
        case DecisionTree(num_labels_per_class, Leaf(value)):
            return value

        case DecisionTree(num_labels_per_class, Node(split, children)):
            label = classify(u, split)
            num_labels = num_labels_per_class[split]

            if 0 > label or label >= num_labels:
                raise ValueError(f"bad label {label} for class {split} with {num_labels} labels")

            return decide(
                u,
                classify,
                DecisionTree(num_labels_per_class, children[label])
            )

        case _:
            raise DecisionTreeMatchingError(dt)
