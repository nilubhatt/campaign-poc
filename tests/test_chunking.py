import chunking


def test_empty_units_yields_no_chunks():
    assert chunking.pack([]) == []
    assert chunking.pack(["", "   ", None]) == []  # type: ignore[list-item]


def test_small_units_merge_into_one_chunk():
    out = chunking.pack(["one", "two", "three"], max_chars=100)
    assert out == ["one\n\ntwo\n\nthree"]


def test_units_split_across_chunks_when_combined_too_big():
    a, b = "x" * 60, "y" * 60
    out = chunking.pack([a, b], max_chars=100)
    assert out == [a, b]
    assert all(len(c) <= 100 for c in out)


def test_oversized_single_unit_is_hard_split():
    unit = "z" * 250
    out = chunking.pack([unit], max_chars=100)
    assert len(out) == 3
    assert "".join(out) == unit
    assert all(len(c) <= 100 for c in out)


def test_order_preserved():
    out = chunking.pack(["first", "second", "third"], max_chars=8)
    assert out == ["first", "second", "third"]
