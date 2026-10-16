"""Adds the Malayalam sample twice, once per Whisper version (both script-repaired), into separate
libraries, so search can be measured on each.   python eval/compare_ml.py"""
from moments.library import Library

for engine in ("whisper", "whisper-v2"):
    library = Library(f"data-ml-{engine}")
    video_id = library.ingest("samples/ml_nooru_simhasanangal_ch4.ogg", language="ml", engine_name=engine,
                              title="Nooru Simhasanangal ch. 4 (Malayalam)")
    meta = library.meta(video_id)
    print(engine, meta["engine"], "repaired:", meta["script_repaired"], meta["segments"], "windows", flush=True)
