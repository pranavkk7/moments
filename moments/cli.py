"""Command line: add videos to the library and search them.

    python -m moments add lecture.mp4 [--language ml] [--title "..."]
    python -m moments list
    python -m moments search "when does the speaker talk about exams?" [--video ID] [-k 5]
    python -m moments ask "when is the exam?" [--video ID]
"""
import argparse
import sys

from .answer import AnswerError, answer
from .library import Library, LibraryError
from .segments import format_time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="moments", description="Ask any video, in any language.")
    parser.add_argument("--data", default="data", help="library folder (default: data)")
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add", help="transcribe a video or audio file and add it to the library")
    add.add_argument("file")
    add.add_argument("--language", help="spoken language, e.g. hi or Hindi (default: detect)")
    add.add_argument("--title")

    commands.add_parser("list", help="list the videos in the library")

    search = commands.add_parser("search", help="find the moments that answer a question")
    search.add_argument("question")
    search.add_argument("--video", help="search one video (id from `list`) instead of all")
    search.add_argument("-k", type=int, default=5, help="how many moments to show (default: 5)")

    ask = commands.add_parser("ask", help="get a written answer with the moments that support it")
    ask.add_argument("question")
    ask.add_argument("--video", help="ask one video (id from `list`) instead of all")
    ask.add_argument("-k", type=int, default=6, help="how many moments the answer may draw on (default: 6)")
    return parser.parse_args(argv)


def main(argv=None, library=None, pool=None, llm=None, out=sys.stdout):
    args = parse_args(argv)
    library = library or Library(args.data)
    try:
        if args.command == "add":
            stage = {"name": "Transcribing"}

            def progress(fraction):
                print(f"\r{stage['name']}... {fraction:.0%}", end="", file=out, flush=True)

            def on_stage(name):
                stage["name"] = name.capitalize()

            video_id = library.ingest(args.file, pool=pool, language=args.language, title=args.title,
                                      progress=progress, on_stage=on_stage)
            meta = library.meta(video_id)
            print(f"\nAdded {meta['title']} ({meta['language']}, {format_time(meta['duration'])}, "
                  f"{meta['segments']} moments) as {video_id}", file=out)
        elif args.command == "list":
            videos = library.videos()
            if not videos:
                print("The library is empty. Add a video with: python -m moments add FILE", file=out)
            for m in videos:
                print(f"{m['id']}  {m['title']}  [{m['language']}, {format_time(m['duration'])}]", file=out)
        elif args.command == "search":
            titles = {m["id"]: m["title"] for m in library.videos()}
            for hit in library.search(args.question, video_id=args.video, k=args.k):
                s = hit.segment
                print(f"{titles.get(hit.video_id, hit.video_id)}  {format_time(s.start)}-{format_time(s.end)}",
                      file=out)
                print(f"  {s.text[:300]}{'...' if len(s.text) > 300 else ''}", file=out)
                if s.english:
                    print(f"  English: {s.english[:300]}{'...' if len(s.english) > 300 else ''}", file=out)
                print(file=out)
        elif args.command == "ask":
            titles = {m["id"]: m["title"] for m in library.videos()}
            hits = library.search(args.question, video_id=args.video, k=args.k)
            result = answer(args.question, hits, titles, llm=llm)
            print(result.text, file=out)
            for c in result.citations:
                print(f"  > {titles.get(c.video_id, c.video_id)}  {format_time(c.start)}-{format_time(c.end)}",
                      file=out)
    except (LibraryError, AnswerError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
