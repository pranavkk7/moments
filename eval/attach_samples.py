"""Records, for each sample video in a library, the browser-friendly copy it can be streamed from and its
credit line (author, licence, source page), as the Creative Commons licences require.
    python eval/attach_samples.py [DATA_DIR]"""
import json
import sys
from pathlib import Path

from moments.library import Library

samples = json.loads(Path("eval/samples.json").read_text("utf-8"))
library = Library(sys.argv[1] if len(sys.argv) > 1 else "data")
for meta in library.videos():
    sample = samples.get(Path(meta["source"]).name)
    if sample:
        library.update_meta(meta["id"], media_url=sample["play_url"],
                            credit={"author": sample["author"], "license": sample["license"], "page": sample["page"]})
        print("attached", meta["id"])
