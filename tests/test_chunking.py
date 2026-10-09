"""source: chunk keeps the 0.1 window and overlap, and drops tiny texts."""

from corpussync.chunking import chunk


def test_chunk_window_and_overlap_match_expected_strings():
    """source: chunk window and overlap stay on the 0.1 word counts."""
    words = [f"w{i:02d}" for i in range(30)]
    got = chunk(" ".join(words), max_tokens=16, overlap=2)
    assert got == [
        "w00 w01 w02 w03 w04 w05 w06 w07 w08 w09 w10 w11",
        "w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21",
    ]


def test_chunk_ten_words_or_fewer_returns_empty():
    """source: chunk returns nothing for text of ten words or fewer."""
    assert chunk("one two three four five six seven eight nine ten") == []
    assert chunk("only nine words live in this short line here") == []
    assert chunk("") == []


def test_chunk_eleven_words_is_one_window():
    """source: a text just over ten words is one chunk under the default window."""
    text = "one two three four five six seven eight nine ten eleven"
    assert chunk(text) == ["one two three four five six seven eight nine ten eleven"]


def test_chunk_default_window_is_393_with_overlap_64():
    """source: the default window is int(512/1.3) words and the overlap is 64."""
    words = [f"w{i}" for i in range(800)]
    chunks = chunk(" ".join(words))
    assert int(512 / 1.3) == 393
    assert chunks[0].split() == words[:393]
    assert chunks[1].split()[:64] == words[393 - 64:393]
