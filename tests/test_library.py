import json

import pytest

from moments.library import LibraryError, script_of, tokens, video_id_for

from .conftest import transcript

LECTURE = (
    ["welcome to the course on soil mechanics"] * 3
    + ["the exam will be held in the last week of march"] * 3
    + ["we now look at how water moves through clay"] * 30
    + ["the course fee can be paid at the office before june"] * 3
)


def test_add_writes_transcript_windows_and_meta(library, media):
    video_id = library.add(media("Lecture 1.mp4"), transcript(LECTURE), title="Soil mechanics 1")
    meta = library.meta(video_id)
    assert meta["title"] == "Soil mechanics 1"
    assert meta["language"] == "en" and meta["segments"] == len(library.segments(video_id))
    folder = library.root / "videos" / video_id
    assert len(json.loads((folder / "transcript.json").read_text("utf-8"))) > 100
    assert library.collection.count() == meta["segments"]


def test_search_finds_the_right_moment(library, media):
    video_id = library.add(media("lecture.mp4"), transcript(LECTURE))
    first = library.search("when is the exam", video_id=video_id, k=3)[0]
    assert "exam" in first.segment.text
    last = library.search("how do I pay the course fee", k=3)[0]
    assert "fee" in last.segment.text
    assert last.segment.start > first.segment.start


def test_an_english_question_finds_a_malayalam_moment(library, media):
    talk = ["ഇന്ന് നമ്മൾ മണ്ണിനെക്കുറിച്ച് പഠിക്കും"] * 20 + ["പരീക്ഷ മാർച്ചിൽ"] * 3 + ["വെള്ളം കളിമണ്ണിലൂടെ"] * 20
    video_id = library.add(media("talk.mp4"), transcript(talk, language="ml"))
    hit = library.search("exam", video_id=video_id, k=1)[0]
    assert "പരീക്ഷ" in hit.segment.text
    assert hit.meaning_rank == 1 and hit.keyword_rank is None  # found by meaning, not by matching words


def test_keywords_find_exact_names_that_meaning_alone_misses(library, media):
    talk = ["general remarks about the project"] * 30 + ["contractor Zubair Al Mansoori signed in 2019"] * 3
    library.add(media("meeting.mp4"), transcript(talk))
    hit = library.search("Mansoori", k=1)[0]
    assert "Mansoori" in hit.segment.text and hit.keyword_rank == 1


def test_search_can_cover_one_video_or_all(library, media):
    a = library.add(media("a.mp4", b"a"), transcript(["the exam is in march"] * 10))
    b = library.add(media("b.mp4", b"b"), transcript(["the exam is in april"] * 10))
    assert {h.video_id for h in library.search("exam", k=10)} == {a, b}
    assert {h.video_id for h in library.search("exam", video_id=a, k=10)} == {a}


def test_adding_the_same_file_again_replaces_its_windows(library, media):
    path = media("lecture.mp4")
    library.add(path, transcript(LECTURE))
    count = library.collection.count()
    library.add(path, transcript(LECTURE))
    assert library.collection.count() == count
    assert len(library.videos()) == 1


def test_video_ids_are_readable_and_tell_files_apart(media):
    a, b = media("Week 1 Lecture.mp4", b"one"), media("week-1-lecture.MP4", b"two")
    assert video_id_for(a).startswith("week-1-lecture-")
    assert video_id_for(a) != video_id_for(b)
    assert video_id_for(a) == video_id_for(a)


def test_helpful_errors(library, media):
    with pytest.raises(LibraryError, match="empty"):
        library.search("anything")
    library.add(media("a.mp4"), transcript(["hello there"] * 5))
    with pytest.raises(LibraryError, match="Type a question"):
        library.search("   ")
    with pytest.raises(LibraryError, match="No video"):
        library.search("hello", video_id="missing")
    with pytest.raises(LibraryError, match="No speech"):
        library.add(media("silent.mp4", b"s"), transcript([""]))


def test_tokens_split_words_or_character_pairs():
    assert tokens("When is the EXAM?", "en") == ["when", "is", "the", "exam"]
    assert tokens("試験は来週", "ja") == ["試験", "験は", "は来", "来週"]


def test_search_modes_use_one_ranking_or_both(library, media):
    talk = ["general remarks about the project"] * 30 + ["contractor Zubair Al Mansoori signed in 2019"] * 3
    library.add(media("meeting.mp4"), transcript(talk))
    keyword_only = library.search("Mansoori", k=3, mode="keyword")
    assert all(h.meaning_rank is None for h in keyword_only) and keyword_only[0].keyword_rank == 1
    assert all(h.keyword_rank is None for h in library.search("Mansoori", k=3, mode="meaning"))
    with pytest.raises(ValueError):
        library.search("Mansoori", mode="fuzzy")


def test_script_of_text():
    assert script_of("When is the exam?") == "LATIN"
    assert script_of("試験はいつですか") == "CJK"  # kana and kanji count as one script
    assert script_of("പരീക്ഷ എപ്പോഴാണ്?") == "MALAYALAM"
    assert script_of("परीक्षा कब है?") == "DEVANAGARI"
    assert script_of("123 ?!") == ""


def test_keywords_do_not_count_across_scripts(library, media):
    # "MIT" appears in the Japanese transcript, and keyword search finds it, but an English question is
    # scored on meaning alone for a Japanese video.
    talk = ["今日 は 土 の 話 を します"] * 40 + ["MIT の 講義 です"] * 3 + ["水 と 粘土 の 話"] * 40
    ja = library.add(media("ja.mp4", b"j"), transcript(talk, language="ja"))
    for h in library.search("MIT lecture", video_id=ja, k=5):
        assert h.score == pytest.approx(1 / (60 + h.meaning_rank), abs=1e-5)
    same_script = library.search("MIT", video_id=ja, k=5, mode="keyword")
    assert same_script and "MIT" in same_script[0].segment.text
