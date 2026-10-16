"""How often search finds the right moment, per video and per search mode.

The test set (eval/questions.json) lists questions for each sample video, written in English and in the
video's own language, each with the time range(s) where the answer is spoken. Those ranges were checked
by reading the transcript and listening to the clip. A retrieved window counts as a hit when it overlaps
an answer range by at least MIN_OVERLAP seconds.

Reported for each mode (hybrid, meaning only, keyword only):
- hit@1 and hit@3: how often a correct window is the first result, or in the first three;
- MRR (mean reciprocal rank): 1 for a hit at rank 1, 1/2 at rank 2, ... averaged over questions.

    python -m moments.evaluate [--questions eval/questions.json] [-k 5]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from polysub import languages

from .library import Library

MIN_OVERLAP = 3.0  # seconds
MODES = ("hybrid", "meaning", "keyword")


def overlaps(segment, ranges, min_overlap=MIN_OVERLAP):
    return any(min(segment.end, b) - max(segment.start, a) >= min(min_overlap, b - a) for a, b in ranges)


def first_hit_rank(hits, ranges):
    """1-based rank of the first window that covers an answer range, or None."""
    return next((rank for rank, hit in enumerate(hits, start=1) if overlaps(hit.segment, ranges)), None)


def load_cases(path, library):
    """Questions joined to the library's video ids, matched by the sample's file name."""
    data = json.loads(Path(path).read_text("utf-8"))
    by_file = {Path(m["source"]).name: m for m in library.videos()}
    cases = []
    for entry in data["videos"]:
        meta = by_file.get(entry["file"])
        if meta is None:
            raise SystemExit(f"{entry['file']} is not in the library. Add it with: python -m moments add samples/{entry['file']}")
        for q in entry["questions"]:
            cases.append({"video_id": meta["id"], "video_language": meta["language"], "title": meta["title"],
                          "question": q["question"], "question_language": q["language"],
                          "ranges": [tuple(r) for r in q["answer"]]})
    return cases


def evaluate(library, cases, modes=MODES, k=5):
    """Scores per mode and per (video language, question language) group."""
    results = {}
    for mode in modes:
        groups = defaultdict(list)
        for case in cases:
            hits = library.search(case["question"], video_id=case["video_id"], k=k, mode=mode)
            rank = first_hit_rank(hits, case["ranges"])
            same = case["question_language"] == case["video_language"]
            groups[(case["video_language"], "same language" if same else "English question")].append(rank)
            groups[("all", "")].append(rank)
        results[mode] = {key: summarise(ranks) for key, ranks in groups.items()}
    return results


def summarise(ranks):
    n = len(ranks)
    return {
        "questions": n,
        "hit@1": sum(1 for r in ranks if r == 1) / n,
        "hit@3": sum(1 for r in ranks if r and r <= 3) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def table(results):
    lines = ["| Mode | Video | Question | n | hit@1 | hit@3 | MRR |", "|---|---|---|---|---|---|---|"]
    for mode, groups in results.items():
        for (video_language, kind), s in sorted(groups.items(), key=lambda kv: (kv[0][0] != "all", kv[0])):
            video = "**all**" if video_language == "all" else languages.name(video_language)
            lines.append(f"| {mode} | {video} | {kind} | {s['questions']} | {s['hit@1']:.0%} | {s['hit@3']:.0%} "
                         f"| {s['mrr']:.2f} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="moments.evaluate", description=__doc__.splitlines()[0])
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--data", default="data")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args(argv)
    library = Library(args.data)
    cases = load_cases(args.questions, library)
    print(f"{len(cases)} questions, embedding model {library.embedder.name}\n")
    print(table(evaluate(library, cases, k=args.k)))


if __name__ == "__main__":
    main()
