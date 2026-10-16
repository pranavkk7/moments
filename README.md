# Moments

**Ask any video, in any language, and jump to the moment that answers.**

Upload a lecture, interview or meeting in Hindi, Arabic, Japanese, Spanish, English or about 100 other
languages. Ask a question in English or in the video's own language, and Moments takes you to the exact
moments that answer it, with the words that were said there translated into English: clicking one
plays the video from there. Everything runs on your own machine, with
no API key and no per-question cost.

Optionally, with an LLM API key, Moments also writes a short answer that cites those moments.

![An English question about the Japanese version of a film. Moments jumps to 1:47, where the narrator says Jeff Bezos recommended the novel; each moment shows its English translation above the original Japanese, and yellow marks on the timeline show the three moments found.](docs/screenshot.png)

## How it works

```
video ──► speech recognition ──► timed words ──► 40-second windows ──► vectors + keyword index
             (PolySub)                │              (overlapping)       (ChromaDB, BM25)
                                      └─► English translation (Whisper) ─┘  for both languages

question ──► hybrid search (meaning + keywords, language-aware) ──► top moments ──► (optional) LLM answer
                                                                                    with checked citations
```

1. **Transcribe.** Speech recognition comes from [PolySub](https://github.com/pranavkk7/polysub), which
   already picks the more accurate engine per language by measurement (Qwen3-ASR or Whisper large-v3).
   Every word keeps its start and end time, and a fixed random seed makes the same file always give the
   same transcript.
2. **Translate.** For a video in another language, Whisper also translates the speech into English, with
   its own timings, on the same GPU and without any API. Each window gets the English that was said
   inside it, which the page shows above the original words.
3. **Cut into windows.** About 40 seconds each, ending at a pause where one is near, overlapping by 10
   seconds so an idea that straddles a cut is whole in one window.
4. **Index.** Multilingual embeddings (BAAI/bge-m3) for meaning and BM25 for exact names and numbers,
   over both the original words and the English translation.
5. **Search.** Up to four rankings are merged with reciprocal rank fusion: meaning and keywords in the
   video's language, and meaning and keywords in English. A keyword match only counts when the question
   is in the same language as the text it matched, judged by which of the two texts more of its words
   appear in (measured below). Keywords ignore accents on Latin letters, so "púlsar" finds "pulsar".
6. **Answer (optional).** With an API key, an LLM sees only the numbered moments, replies in a fixed JSON
   schema, and the code checks that every cited moment is one it was actually given. This part is covered
   by unit tests; the results below measure search, which is what Moments does without a key.

## Results

67 questions over 5 videos in 5 languages, each with the time range where the answer is spoken, read
from the timed transcripts ([eval/questions.json](eval/questions.json)). A result counts when its window
overlaps the answer by at least 3 seconds. `python -m moments.evaluate` reproduces the tables and
`python eval/compare_rankings.py` the comparison of rankings.

| Search | Right moment first | In the top 3 | MRR |
|---|---|---|---|
| **Moments (four rankings, language-aware)** | **90%** | **100%** | **0.94** |
| Original language only (meaning + keywords) | 85% | 99% | 0.92 |
| Meaning only (bge-m3) | 81% | 99% | 0.90 |
| Keywords only (BM25) | 42% | 52% | 0.48 |

By group:

| Video | Questions in | n | First | Top 3 |
|---|---|---|---|---|
| English | English | 12 | 92% | 100% |
| Spanish | English | 12 | 100% | 100% |
| Spanish | Spanish | 3 | 100% | 100% |
| Japanese | English | 12 | 75% | 100% |
| Japanese | Japanese | 4 | 100% | 100% |
| Hindi | English | 8 | 75% | 100% |
| Hindi | Hindi | 3 | 100% | 100% |
| Arabic | English | 10 | 90% | 100% |
| Arabic | Arabic | 3 | 100% | 100% |

What the numbers showed, in the order the changes were made:

- **Keywords can't cross languages.** BM25 over the original words found 0% of the English-question
  moments in the Hindi and Arabic videos. Meaning-based search is what connects the languages.
- **But keywords help within a language.** Adding them lifted Arabic-on-Arabic from 67% to 100% first
  place and Japanese-on-Japanese from 75% to 100%.
- **So keyword matches must stay within one language.** At first they counted whenever the question and
  the transcript shared a script, which already helped (English questions on the Japanese video: 50% to
  58% first). Spanish showed the flaw: it shares the Latin alphabet with English, so English questions
  were keyword-matched against Spanish words, and the Spanish video scored 92% and 67% first for English
  and Spanish questions while meaning alone scored 100%. Judging the question's language from which text
  its words appear in fixed it: 100% for both.
- **Searching the English translation too was the biggest single gain.** For English questions on the
  four non-English videos it lifted first place from 79% to 86% and put the right moment in the top 3
  every time; Japanese went from 58% to 75%. Same-language questions were unchanged, and the
  translation alone would have been far worse for them (52% first), so Moments keeps both.

Small test sets: treat differences of one or two questions as noise.

### Lessons from a fifth language

An early version also had a Malayalam audiobook. Whisper wrote its Malayalam speech in the wrong
alphabet (Gurmukhi with large-v3, Kannada with large-v2), so Moments now repairs transcripts written in
another Indian script by moving each letter to the same place in the right script's Unicode block
([moments/scripts.py](moments/scripts.py)). Whisper also gave a different transcript on every run of that
file, which is why transcription now uses a fixed random seed. Malayalam was then dropped from the
samples, because Whisper still skipped whole sentences of it; a speech model trained for Malayalam would
be the way back.

## Run it

```bash
python -m venv --system-site-packages .venv     # reuses a CUDA build of PyTorch if installed globally
.venv/Scripts/pip install -e ../polysub -r requirements.txt fastapi "uvicorn[standard]" python-multipart python-dotenv anthropic
copy .env.example .env                           # optional: an API key for written answers

.venv/Scripts/python -m moments add lecture.mp4 --language ml
.venv/Scripts/python -m moments search "when is the exam?"
.venv/Scripts/python -m moments ask "when is the exam?"                  # needs the optional API key
.venv/Scripts/python -m uvicorn moments.api:app --port 8010     # the web page
```

`MOMENTS_DEMO=1` runs the page without uploads (for hosting without a GPU) over an existing library;
`python eval/attach_samples.py` points the sample videos at copies on Wikimedia Commons that every browser
plays, with their credit lines.

Transcription needs an NVIDIA GPU (tested on an RTX 4060, 8 GB). Search runs on the CPU too.

```bash
.venv/Scripts/python -m pytest       # tests use stand-ins: no GPU, model download or API key needed
```

## Sample videos

All from Wikimedia Commons, used under their licences:

| File | Language | Licence |
|---|---|---|
| [Reticulum Rex](https://commons.wikimedia.org/wiki/File:Reticulum_Rex_with_subtitles_-_Creative_Commons.ogv), Creative Commons | English | CC BY-SA 1.0 |
| [Reticulum Rex (in Japanese)](https://commons.wikimedia.org/wiki/File:Reticulum_Rex_(in_Japanese)_-_Creative_Commons..ogv), Creative Commons | Japanese | CC BY-SA 2.1 JP |
| [About The Sengol](https://commons.wikimedia.org/wiki/File:About_The_Sengol,_About_its_History_and_Significance.webm) | Hindi | CC BY 3.0 |
| [Ancient construction practices, Dura-Europos (Djuss)](https://commons.wikimedia.org/wiki/File:Ancient_construction_practices_Dura-Europos_(Djuss).webm) | Arabic | CC BY-SA 4.0 |
| ["Las estrellas"](https://commons.wikimedia.org/wiki/File:%22Las_estrellas%22_(Worldnet_TV)..webm), Worldnet TV documentary | Spanish | Public domain |

The files themselves are not in the repository (`samples/` is ignored); [eval/samples.json](eval/samples.json) lists
where each came from.

## Licence

The code is released under the [MIT licence](LICENSE). The sample videos keep their own licences (above).
