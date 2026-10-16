"""Adds English translations to every video in a library that is not in English yet.
    python eval/translate_all.py [DATA_DIR]"""
import sys
import time

from moments.library import Library

library = Library(sys.argv[1] if len(sys.argv) > 1 else "data")
for meta in library.items():
    if meta["language"] != "en" and not meta.get("english"):
        start = time.time()
        library.translate(meta["id"])
        print(f"translated {meta['id']} in {time.time() - start:.0f}s", flush=True)
