"""Custom list-based collection type implementing a positive-integer-keyed mapping.

Type Parameters:
    V: Value type stored in the map.
"""
from copy import copy
from typing import Callable, Generic, Iterator, MutableMapping, TypeVar, cast

V = TypeVar('V')


class DenseIntDict(MutableMapping[int, V], Generic[V]):
    """A MutableMapping[int, V] backed by a list for access time optimization.

    Though we hoped that these would perform better than Python dicts in the case of
    integer-named states and integer input characters, this turned out not to be the case :(

    Type Parameters:
        V: Value type stored in the map.
    """

    def __init__(self, size: int = 0, init: Callable[[], V] | None = None) -> None:
        self._data: list[tuple[V] | None]
        if init is None:
            self._data = [None] * size
        else:
            self._data = [(init(),) for _ in range(size)]
        self._size = size

    def __copy__(self) -> 'DenseIntDict[V]':
        """Create a shallow copy of this DenseIntDict.

        Returns:
            A new DenseIntDict with the same keys and values.
        """
        new_dict: DenseIntDict[V] = DenseIntDict()
        new_dict._data = copy(self._data)
        new_dict._size = self._size
        return new_dict

    def __contains__(self, key: object) -> bool:
        """Check if a key exists in the map.

        Supports the 'in' operator for efficient key membership testing.
        """
        return (
            isinstance(key, int) and
            0 <= key and
            key < len(self._data) and
            self._data[key] is not None
        )

    def __getitem__(self, key: int) -> V:
        """Retrieve a value by integer key.

        Raises:
            KeyError: if the key is not present.
        """
        if key in self:
            (v,) = cast(tuple[V], self._data[key])
            return v
        else:
            raise KeyError(key)

    def __setitem__(self, key: int, value: V) -> None:
        """Set a value for an integer key.

        Automatically expands the underlying list as needed.

        Raises:
            KeyError: if the key is negative.
        """
        if key < 0:
            raise KeyError(key)
        if key >= len(self._data):
            self._data.extend([None] * (key - len(self._data) + 1))
        if self._data[key] is None:
            self._size += 1
        self._data[key] = (value,)

    def __delitem__(self, key: int) -> None:
        """Delete a value by integer key.

        Raises:
            KeyError: if the key is not present.
        """
        if key in self:
            self._data[key] = None
            self._size -= 1
        else:
            raise KeyError(key)

    def __iter__(self) -> Iterator[int]:
        """Iterate over keys that have values set."""
        for key, value in enumerate(self._data):
            if value is not None:
                yield key

    def __len__(self) -> int:
        """Return the number of entries in the map."""
        return self._size

    def __repr__(self) -> str:
        """Return a string representation of the map."""
        items = ', '.join(f'{k}: {v!r}' for k, v in self.items())
        return f'{self.__class__.__name__}({{{items}}})'
