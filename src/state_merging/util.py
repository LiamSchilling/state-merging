"""Utility functions for sequence operations.

Type Parameters:
    U: Sequence element type
    U: Another sequence element type
"""
from typing import Callable, Collection, Sequence, TypeVar

U = TypeVar('U')
V = TypeVar('V')


def lcp(us: Collection[Sequence[U]], epsilon: Sequence[U]) -> Sequence[U]:
    """Compute the longest common prefix from a collection of sequences.

    We prefer the semantics of Collection to Iterable for the input because it will be
    traversed multiple times and must not be consumed on the first use.
    Additionally, we take special care to maintain true type-parametricity with
    respect to the implementation of the Sequence type, even requiring the
    epsilon sequence as an argument, to be used when the collection is empty.
    As a further measure, we wrap some instances of U in a tuple singleton
    to properly distinguish values of U from None in the case that U is instantiated as None.

    Args:
        us: A set of tuples over the same element type.
        epsilon: The empty/identity sequence.

    Returns:
        The longest common prefix sequence of the input collection.
    """
    some_u: Sequence[U] | None = None

    i = 0
    while True:
        success = True
        c: tuple[U] | None = None
        for u in us:
            some_u = u
            if i < len(u):
                c_: tuple[U] = (u[i],)
                if c is None:
                    c = c_
                elif c != c_:
                    success = False
                    break
            else:
                success = False
                break
        if not success or c is None:
            break
        i += 1

    if some_u is not None:
        return some_u[:i]
    else:
        return epsilon


def edit_distance(
    u: Sequence[U],
    v: Sequence[V],
    del_cost: Callable[[U], int],
    ins_cost: Callable[[V], int],
    subst_cost: Callable[[U, V], int]
) -> int:
    """Compute the minimum cost edit distance between two sequences.

    Uses dynamic programming to find the minimum cost sequence of edit operations
    (insertions, deletions, and substitutions) needed to transform sequence u
    into sequence v. Each operation has an associated cost provided by the
    corresponding cost function.

    Args:
        u: The source sequence to transform.
        v: The target sequence to transform into.
        del_cost: Function that returns the cost of deleting an element from u.
        ins_cost: Function that returns the cost of inserting an element from v.
        subst_cost: Function that returns the cost of substituting an element from u
                    with an element from v.

    Returns:
        The minimum cost to transform u into v.
    """
    pref_costs = [[0 for _ in range(len(v) + 1)] for _ in range(len(u) + 1)]

    for i in range(len(u)):
        pref_costs[i+1][0] = pref_costs[i][0] + del_cost(u[i])

    for j in range(len(v)):
        pref_costs[0][j+1] = pref_costs[0][j] + ins_cost(v[j])

    for i in range(len(u)):
        for j in range(len(v)):
            pref_costs[i+1][j+1] = min(
                pref_costs[i][j+1] + del_cost(u[i]),
                pref_costs[i+1][j] + ins_cost(v[j]),
                pref_costs[i][j] + subst_cost(u[i], v[j])
            )

    return pref_costs[len(u)][len(v)]


def is_pref(pref: Sequence[U], u: Sequence[U]) -> bool:
    """Check if u1 is a prefix of u2.

    Args:
        pref: The potential prefix sequence.
        u: The sequence to check against.

    Returns:
        True if pref is a prefix of u, False otherwise.
    """
    return u[:len(pref)] == pref


def is_suff(u: Sequence[U], suff: Sequence[U]) -> bool:
    """Check if u2 is a suffix of u1.

    Args:
        u: The sequence to check for the suffix.
        suff: The potential suffix sequence.

    Returns:
        True if suff is a suffix of u, False otherwise.
    """
    return len(suff) == 0 or u[-len(suff):] == suff


def ldiv(pref: Sequence[U], u: Sequence[U]) -> Sequence[U]:
    """Compute left division: remove a prefix from a sequence.

    Args:
        pref: The prefix to remove.
        u: The sequence to remove the prefix from.

    Returns:
        The suffix of u after removing prefix pref.
    """
    return u[len(pref):]


def rdiv(u: Sequence[U], suff: Sequence[U]) -> Sequence[U]:
    """Compute right division: remove a suffix from a sequence.

    Args:
        u: The sequence to remove the suffix from.
        suff: The suffix to remove.

    Returns:
        The prefix of u after removing suffix suff.
    """
    return u if len(suff) == 0 else u[:-len(suff)] # 0 is an edge case for negative indexing


def match(u_src: Sequence[U], u_dest: Sequence[U]) -> Sequence[U]:
    """Asserts that two sequences are equal and returns the sequences.

    Args:
        u_src: One string to compare.
        u_dest: The string to compare that will be returned.

    Returns:
        Exactly u_dest.

    Raises:
        AssertionError: if u_src is not equal to u_dest
    """
    assert u_src == u_dest, \
        f"expected {u_src} to equal {u_dest}"
    return u_dest


def unify(u_src: Sequence[U], u_dest: Sequence[U]) -> tuple[Sequence[U], Sequence[U]] | None:
    """Unifies a sequence with a prefix pattern.

    Args:
        u_src: The sequence to unify.
        u_dest: The prefix that must be encountered.

    Returns:
        The tuple (u_dest, suff) where u_dest + suff is exactly u_src.
    """
    if is_pref(u_dest, u_src):
        return u_dest, ldiv(u_dest, u_src)
    else:
        return None
