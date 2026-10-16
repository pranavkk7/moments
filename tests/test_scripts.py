from polysub.cues import Word

from moments.scripts import repair_words, to_script


def test_kannada_and_gurmukhi_letters_move_to_malayalam():
    assert to_script("ಅಧಿಯಾಯಂ ನಾಲ", "MALAYALAM") == "അധിയായം നാല"  # "chapter four"
    assert to_script("ਪਲਾਹਾਰਂ", "MALAYALAM") == "പലാഹാരം"  # "snacks", from Gurmukhi


def test_text_already_in_the_right_script_or_not_indian_is_untouched():
    assert to_script("നാല് Creative Commons 4.0", "MALAYALAM") == "നാല് Creative Commons 4.0"


def test_letters_without_a_counterpart_are_kept():
    # Gurmukhi ੜ (U+0A5C, a letter) would land on U+0D5C, a Malayalam fraction sign: it stays as it is.
    assert to_script("ੜ", "MALAYALAM") == "ੜ"


def test_unassigned_code_points_from_whisper_move_too():
    # U+0BA7 is unassigned in Tamil; Whisper writes it where Malayalam ധ (U+0D27) was spoken.
    assert to_script("அ஧்யாயம்", "MALAYALAM") == "അധ്യായമ്"  # Tamil ம் is ma + virama


def test_repair_words_only_for_indian_languages():
    words = [Word("ಅಧಿಯಾಯಂ", 0.0, 0.5), Word("ನಾಲ", 0.5, 0.9)]
    fixed, repaired = repair_words(words, "ml")
    assert repaired and [w.text for w in fixed] == ["അധിയായം", "നാല"]
    assert fixed[1].start == 0.5  # timing untouched
    same, repaired = repair_words([Word("hello", 0, 1)], "en")
    assert not repaired and same[0].text == "hello"
