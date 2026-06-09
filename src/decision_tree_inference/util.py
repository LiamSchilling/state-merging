"""Tree container types used by decision-tree utilities.

This small module provides two container types used to build simple
decision-tree structures:

    - `Leaf[L]` holds a value of type `L` and represents a terminal node.
    - `Node[N, L]` stores a value of type `N` and a list of child `Tree[N, L]` values.

The public `Tree` alias represents either a `Node` or a `Leaf`.

Type parameters:
    N: Value type stored at internal (non-leaf) nodes.
    L: Value type stored at leaves.
"""

from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

N = TypeVar('N')
L = TypeVar('L')


@dataclass
class Leaf(Generic[L]):
    """A terminal tree node containing a value.

    Attributes:
        value: The stored leaf value of type `L`.
    """
    value: L


@dataclass
class Node(Generic[N, L]):
    """An internal branching node.

    Attributes:
        value: The value stored at the internal node (type `N`). It is used
            by external classifier logic to select which child branch to follow.
        children: A list of child `Tree[N, L]` values.
    """
    value: N
    children: list["Tree[N, L]"]


Tree: TypeAlias = Node[N, L] | Leaf[L]
"""Recursive tree type alias.

Represents either an internal `Node` carrying a value of type `N` and a list
of child `Tree[N, L]` values, or a terminal `Leaf` carrying a value of type
`L`.

Type parameters:
    N: Value type stored at internal nodes.
    L: Leaf value type.
"""
