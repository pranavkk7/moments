"""How often search finds the right moment, per video and per search mode.

The test set (eval/questions.json) lists questions for each sample video, written in English and in the
video's own language, each with the time range(s) where the answer is spoken. Those ranges were checked
by reading the transcript and listening to the clip. A retrieved window counts as a hit when it overlaps
an answer range by at least MIN_OVERLAP seconds.

Reported for each mode (hybrid, meaning only, keyword only):
- hit@1 and hit@3: how often a correct window is the first result, or in the first three;
- MRR (mean reciprocal rank): 1 for a hit at rank 1, 1/2 at rank 2, ... averaged over questions.

With --answers it also asks the LLM (this costs API calls) and reports, for written answers:
- cited right: of the questions the video answers, how often a cited moment covers the answer;
- said not covered: of the questions the video does not answer (empty answer list), how often the
  answer says so instead of guessing.

    python -m moments.evaluate [--questions eval/questions.json] [-k 5] [--answers]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from polysub import languages

from .library import Library
from .segments import Segment

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
    cases = [c for c in cases if c["ranges"]]  # questions the video does not answer have nothing to find
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


def evaluate_answers(library, cases, llm=None, per_video=3, k=6):
    """Written answers for the first `per_video` answerable questions of each video plus every question it
    does not answer. Returns the two rates and one row per question for inspection."""
    from .answer import answer

    titles = {m["id"]: m["title"] for m in library.videos()}
    chosen, seen = [], defaultdict(int)
    for case in cases:
        if not case["ranges"] or seen[case["video_id"]] < per_video:
            chosen.append(case)
            seen[case["video_id"]] += bool(case["ranges"])
    rows = []
    for case in chosen:
        hits = library.search(case["question"], video_id=case["video_id"], k=k)
        result = answer(case["question"], hits, titles, llm=llm)
        cited_right = any(overlaps(Segment(0, c.start, c.end, ""), case["ranges"]) for c in result.citations)
        rows.append({"question": case["question"], "title": case["title"], "covered": bool(case["ranges"]),
                     "found": result.found, "cited_right": cited_right, "answer": result.text})
    answerable = [r for r in rows if r["covered"]]
    absent = [r for r in rows if not r["covered"]]
    return {
        "cited right": sum(r["cited_right"] for r in answerable) / max(1, len(answerable)),
        "said not covered": sum(not r["found"] for r in absent) / max(1, len(absent)),
        "answerable": len(answerable), "not covered": len(absent), "rows": rows,
    }


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
    parser.add_argument("--answers", action="store_true", help="also score written answers (uses the API)")
    args = parser.parse_args(argv)
    library = Library(args.data)
    cases = load_cases(args.questions, library)
    print(f"{len(cases)} questions, embedding model {library.embedder.name}\n")
    print(table(evaluate(library, cases, k=args.k)))
    if args.answers:
        from dotenv import load_dotenv

        load_dotenv()
        scores = evaluate_answers(library, cases)
        print(f"\nWritten answers: cited the right moment for {scores['cited right']:.0%} of "
              f"{scores['answerable']} answerable questions; said 'not covered' for "
              f"{scores['said not covered']:.0%} of {scores['not covered']} questions the video does not answer.")
        for r in scores["rows"]:
            if r["covered"]:
                mark = "ok   " if r["cited_right"] else "MISS "
            else:
                mark = "ok   " if not r["found"] else "GUESS"
            print(f"[{mark}] {r['title']}: {r['question']}\n        {r['answer'][:200]}")


if __name__ == "__main__":
    main()
