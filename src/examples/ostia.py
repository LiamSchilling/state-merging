"""Demonstration of the OSTIA algorithm. The implementation is highly type-parametric,
so it can just as well be invoked for strings as for lists.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from itertools import count
from typing import Sequence, cast

from algorithms.ostia import ostia
from automata.SFST import SFST, assert_SFST, run

input_set: set[str] = {'a', 'b'}

dataset_list: list[tuple[list[str], list[int]]] = [
    (['a'], [1]),
    (['b'], [1]),
    (['a', 'a'], [0, 1]),
    (['a', 'a', 'a'], [0, 0, 1]),
    (['a', 'b', 'a', 'b'], [0, 1, 0, 1]),
    (['a', 'b'], [0, 1])
]

dataset_str: list[tuple[str, str]] = [
    ('a', '1'),
    ('b', '1'),
    ('aa', '01'),
    ('aaa', '001'),
    ('abab', '0101'),
    ('ab', '01')
]

if __name__ == "__main__":
    fst_list: SFST[int, str, Sequence[int]] = ostia(
        input_set=input_set,
        dataset=dataset_list,
        epsilon=[],
        concat=lambda v1, v2: cast(list[int], v1) + cast(list[int], v2),
        choose_transition=lambda _, trs: next(iter(trs)),
        search_iter=lambda _, qs: iter(qs),
        state_supply=count(),
        verbose=True
    )

    assert_SFST(fst_list)

    for input, output in dataset_list:
        match run(fst_list, input, cast(list[int], []), lambda v1, v2: v1 + cast(list[int], v2)):
            case None:
                assert False, \
                    f"learned FST rejected positive data {input, output}"
            case _, real_output:
                assert real_output == output, \
                    f"learned FST incorrectly output {real_output} on positive data {input, output}"

    print("\n" + "-" * 80 + "\n")

    fst_str: SFST[int, str, Sequence[str]] = ostia(
        input_set=input_set,
        dataset=dataset_str,
        epsilon='',
        concat=lambda v1, v2: cast(str, v1) + cast(str, v2),
        choose_transition=lambda _, trs: next(iter(trs)),
        search_iter=lambda _, qs: iter(qs),
        state_supply=count(),
        verbose=True
    )

    assert_SFST(fst_str)

    for input, output in dataset_str:
        match run(fst_str, input, '', lambda v1, v2: v1 + cast(str, v2)):
            case None:
                assert False, \
                    f"learned FST rejected positive data {input, output}"
            case _, real_output:
                assert real_output == output, \
                    f"learned FST incorrectly output {real_output} on positive data {input, output}"
