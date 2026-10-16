from polysub.cues import Word

from moments.segments import build_segments, format_time

from .conftest import words_from


def test_windows_cover_every_word_and_overlap():
    words = words_from([f"sentence number {i} has six words" for i in range(60)])
    segments = build_segments(words, "en")
    assert len(segments) > 3
    assert segments[0].start == words[0].start
    assert segments[-1].end == words[-1].end
    for a, b in zip(segments, segments[1:]):
        assert b.start < a.end  # consecutive windows overlap
        assert b.start > a.start  # and always move forward
    assert [s.index for s in segments] == list(range(len(segments)))


def test_windows_end_at_a_pause_near_the_target_length():
    words = words_from([f"this is sentence {i} of the talk" for i in range(40)], gap=1.0)
    for segment in build_segments(words, "en")[:-1]:
        assert 40 <= segment.end - segment.start <= 55
        # Each window ends on the last word of a sentence, so no sentence is cut in half.
        assert segment.text.split()[-1] == "talk"


def test_long_speech_without_pauses_is_still_cut():
    words = [Word("word", i * 0.4, i * 0.4 + 0.39) for i in range(400)]  # 160 s, no pauses
    segments = build_segments(words, "en")
    assert all(s.end - s.start <= 55 for s in segments)


def test_languages_without_spaces_are_joined_without_spaces():
    words = [Word("試験", 0.0, 0.5), Word("は", 0.5, 0.7), Word("来週", 0.7, 1.2)]
    assert build_segments(words, "ja")[0].text == "試験は来週"


def test_empty_transcript_gives_no_windows():
    assert build_segments([], "en") == []
    assert build_segments([Word("  ", 0, 1)], "en") == []


def test_format_time():
    assert format_time(5) == "0:05"
    assert format_time(754.9) == "12:34"
    assert format_time(3725) == "1:02:05"
