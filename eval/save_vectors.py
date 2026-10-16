"""Writes Moments' own copy of every video's vectors (vectors.npz) for libraries made before it kept one.
    python eval/save_vectors.py [DATA_DIR]"""
import sys

import numpy as np

from moments.library import Library

library = Library(sys.argv[1] if len(sys.argv) > 1 else "data")
for meta in library.items():
    segments = library.segments(meta["id"])
    translated = [s for s in segments if s.english]
    arrays = {"original": library.embedder.passages([s.text for s in segments]),
              "original_index": [s.index for s in segments]}
    if translated:
        arrays.update(english=library.embedder.passages([s.english for s in translated]),
                      english_index=[s.index for s in translated])
    np.savez(library.root / "videos" / meta["id"] / "vectors.npz", **arrays)
    print("saved", meta["id"], len(segments), "windows,", len(translated), "translated")
