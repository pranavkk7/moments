"""Repairing transcripts that Whisper wrote in the wrong Indian script.

Whisper sometimes writes a language in the alphabet of a neighbouring one: on a Malayalam recording we
tried, large-v3 wrote the speech in Gurmukhi (Punjabi script) and large-v2 wrote it in Kannada. The sounds
are right; only the letters are wrong. The major Indian scripts share one layout in Unicode (inherited
from the ISCII standard): the same consonant or vowel sign sits at the same offset in each script's
block. So moving each letter to the same offset in the right block recovers readable text in the right
script, e.g. Kannada ನಾಲ -> Malayalam നാല ("four").

The repair is approximate where one script lacks a letter the other has, but it turns text that neither
search nor a reader can use into text that both can.
"""
import unicodedata

# The first code point of each script's block. All are 128 code points long.
BLOCKS = {
    "DEVANAGARI": 0x0900, "BENGALI": 0x0980, "GURMUKHI": 0x0A00, "GUJARATI": 0x0A80, "ORIYA": 0x0B00,
    "TAMIL": 0x0B80, "TELUGU": 0x0C00, "KANNADA": 0x0C80, "MALAYALAM": 0x0D00,
}
SCRIPT_OF_LANGUAGE = {
    "hi": "DEVANAGARI", "mr": "DEVANAGARI", "ne": "DEVANAGARI", "bn": "BENGALI", "pa": "GURMUKHI",
    "gu": "GUJARATI", "or": "ORIYA", "ta": "TAMIL", "te": "TELUGU", "kn": "KANNADA", "ml": "MALAYALAM",
}


def block_of(ch):
    code = ord(ch)
    return next((name for name, start in BLOCKS.items() if start <= code < start + 0x80), None)


def to_script(text, script):
    """Moves every letter from another Indian script's block to the same place in `script`'s block.
    A character only moves onto the same kind of character (letter to letter, vowel sign to vowel sign);
    one with no counterpart, such as Gurmukhi ੜ whose slot in Malayalam is a fraction sign, is kept.
    Code points that are unassigned in their own script move onto any letter or vowel sign."""
    target = BLOCKS[script]
    out = []
    for ch in text:
        source = block_of(ch)
        if source and source != script:
            moved = chr(ord(ch) - BLOCKS[source] + target)
            # Whisper also emits code points that are unassigned in the wrong script (Tamil has no ധ, yet
            # U+0BA7 appears where ധ was said); those move whenever their slot in the target is in use.
            kind = unicodedata.category(ch)[0]
            fits = unicodedata.category(moved)[0] == kind or (kind == "C" and unicodedata.category(moved)[0] in "LM")
            out.append(moved if unicodedata.name(moved, "") and fits else ch)
        else:
            out.append(ch)
    return "".join(out)


def repair_words(words, language):
    """Returns (words, repaired) with each word's text moved into the language's own script, when the
    language is one of the Indian scripts above. Times are untouched."""
    script = SCRIPT_OF_LANGUAGE.get(language)
    if not script:
        return words, False
    repaired = False
    for w in words:
        fixed = to_script(w.text, script)
        if fixed != w.text:
            w.text = fixed
            repaired = True
    return words, repaired
