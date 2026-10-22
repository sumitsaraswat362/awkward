# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("stable", [True, False])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "keys",
    [
        ["a\x00c", "a\x00b"],
        [b"a\x00c", b"a\x00b"],
        ["a\x00b", "a\x00c", "a\x00a", "a\x00b\x00z", "a\x00b\x00a"],
        [b"a\x00b", b"a\x00c", b"a\x00a", b"a\x00b\x00z", b"a\x00b\x00a"],
        ["", "a", "a\x00", "a\x00a", "a\x00b", "b", "\x00", "\x00\x00"],
        [b"", b"a", b"a\x00", b"a\x00a", b"a\x00b", b"b", b"\x00", b"\x00\x00"],
    ],
)
def test_argsort_strings_embedded_nul(keys, nested, stable, ascending):
    array = ak.Array([keys, [], keys] if nested else keys)
    result_argsort = ak.argsort(array, ascending=ascending, stable=stable).to_list()
    result_sort = ak.sort(array, ascending=ascending, stable=stable).to_list()

    groups = [keys, [], keys] if nested else [keys]
    expected_argsort = result_argsort if nested else [result_argsort]
    expected_sort = result_sort if nested else [result_sort]

    for group, indices, sorted_vals in zip(
        groups, expected_argsort, expected_sort, strict=True
    ):
        assert sorted(indices) == list(range(len(group)))
        expected_sorted_group = sorted(group, reverse=not ascending)
        assert sorted_vals == expected_sorted_group
        assert [group[i] for i in indices] == expected_sorted_group
        if stable:
            assert indices == sorted(
                range(len(group)), key=group.__getitem__, reverse=not ascending
            )
