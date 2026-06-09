"""Custom list-based collection type implementing a positive-integer set.

Wraps DenseIntDict[int, None] to provide an efficient set implementation
for dense ranges of non-negative integers.
"""

from typing import Iterator, MutableSet

from state_merging.collections.dense_int_dict import DenseIntDict


class DenseIntSet(MutableSet[int]):
    """A MutableSet[int] backed by a list for access time optimization.

    Though we hoped that these would perform better than Python dicts in the case of
    integer-named states and integer input characters, this turned out not to be the case :(
    """

    def __init__(self, size: int = 0) -> None:
        """Initialize the set with optional pre-allocation.

        Args:
            size: Optional size for pre-allocating the underlying storage.
                  Defaults to 0 for an empty set.
        """
        self._dict: DenseIntDict[None] = DenseIntDict(size=size)

    def __contains__(self, key: object) -> bool:
        """Check if an integer is in the set.

        Supports the 'in' operator for efficient membership testing.

        Args:
            key: The value to check for membership.

        Returns:
            True if key is an integer in the set, False otherwise.
        """
        return key in self._dict

    def __iter__(self) -> Iterator[int]:
        """Iterate over integers in the set.

        Yields:
            Each integer currently in the set.
        """
        return iter(self._dict)

    def __len__(self) -> int:
        """Return the number of integers in the set.

        Returns:
            The size of the set.
        """
        return len(self._dict)

    def __repr__(self) -> str:
        """Return a string representation of the set."""
        items = ', '.join(str(i) for i in self)
        return f'{self.__class__.__name__}({{{items}}})'

    def add(self, value: int) -> None:
        """Add an integer to the set.

        If the value is already in the set, this has no effect.

        Args:
            value: The integer to add.

        Raises:
            KeyError: If value is negative.
        """
        self._dict[value] = None

    def discard(self, value: int) -> None:
        """Remove an integer from the set if present.

        If the value is not in the set, this has no effect.

        Args:
            value: The integer to remove.
        """
        try:
            del self._dict[value]
        except KeyError:
            pass

    def remove(self, value: int) -> None:
        """Remove an integer from the set.

        Raises KeyError if the value is not in the set.

        Args:
            value: The integer to remove.

        Raises:
            KeyError: If the value is not in the set.
        """
        del self._dict[value]
