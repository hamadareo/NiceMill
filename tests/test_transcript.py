from pfb_app.preprocess.transcript import (
    Segment,
    parse_pasted_transcript,
    slice_transcript,
)

SEGS = [
    Segment(0, 4, "はじめます"),
    Segment(4, 9, "ここを見てください"),
    Segment(65, 70, "おわりです"),
]


def test_slice_transcript_overlap():
    assert slice_transcript(SEGS, 5, 10) == "[00:04] ここを見てください"
    assert slice_transcript(SEGS, 0, 100) == (
        "[00:00] はじめます\n[00:04] ここを見てください\n[01:05] おわりです"
    )
    assert slice_transcript(SEGS, 20, 30) == ""
    # boundary: a segment ending exactly at start does not overlap
    assert slice_transcript(SEGS, 4, 5) == "[00:04] ここを見てください"


def test_parse_pasted_transcript_with_times():
    segs = parse_pasted_transcript("[00:05] こんにちは\n01:10 つぎです\n続きの行\n[1:02:03] 長い")
    assert [s.start for s in segs] == [5, 70, 3723]
    assert segs[0].end == 70
    assert segs[1].text == "つぎです\n続きの行"
    assert segs[2].text == "長い"


def test_parse_pasted_transcript_without_times():
    segs = parse_pasted_transcript("先生の話\nです")
    assert len(segs) == 1
    assert segs[0].start == 0
    assert segs[0].text == "先生の話\nです"
    assert slice_transcript(segs, 100, 200) == "[00:00] 先生の話\nです"


def test_parse_empty():
    assert parse_pasted_transcript("  \n") == []


def test_segment_dict_roundtrip():
    s = Segment(1.5, 3.0, "a")
    assert Segment.from_dict(s.to_dict()) == s
