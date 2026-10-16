"""Which combination of rankings finds the right moment most often? Compares search over the original
transcript only with adding the English translation (its meaning, its keywords, or both).
    python eval/compare_rankings.py [DATA_DIR]"""
import sys

from moments.evaluate import load_cases, summarise, first_hit_rank
from moments.library import Library

COMBOS = {
    "original (meaning + keywords)": ("meaning", "keyword"),
    "+ translation meaning": ("meaning", "keyword", "meaning_en"),
    "+ translation keywords": ("meaning", "keyword", "keyword_en"),
    "+ translation meaning + keywords": ("meaning", "keyword", "meaning_en", "keyword_en"),
    "translation only (meaning + keywords)": ("meaning_en", "keyword_en"),
}

library = Library(sys.argv[1] if len(sys.argv) > 1 else "data")
cases = [c for c in load_cases("eval/questions.json", library) if c["ranges"] or c["pages"]]
groups = {"all": lambda c: True,
          "English questions, other-language videos": lambda c: c["question_language"] == "en" and c["video_language"] != "en",
          "same-language questions": lambda c: c["question_language"] == c["video_language"]}
print("| Rankings | Group | n | First | Top 3 | MRR |\n|---|---|---|---|---|---|")
for name, rankings in COMBOS.items():
    ranks = {}
    for case in cases:
        hits = library.search(case["question"], video_id=case["video_id"], k=5, rankings=rankings)
        ranks[id(case)] = first_hit_rank(hits, case["ranges"], case["pages"])
    for group, keep in groups.items():
        s = summarise([ranks[id(c)] for c in cases if keep(c)])
        print(f"| {name} | {group} | {s['questions']} | {s['hit@1']:.0%} | {s['hit@3']:.0%} | {s['mrr']:.2f} |")
