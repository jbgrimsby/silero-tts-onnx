# DEVELOPMENT.md — silero-onnx-tts internals

For contributors to this codebase. User-facing usage lives in `README.md`.

## 1. Pipeline architecture

```
text
 │ normalize()            # \s+ collapse, abbreviations (dict + context rules),
 │                        # numbers incl. minus/decimals -> words
 ▼
FullStress                 # homographs first, then plain words:
 │  OnnxStress             #   regex/phrases/dict fast path (pure Python);
 │  │                      #   neural fallback: BERT-tiny -> silero_stress.onnx
 │  │                      #   (input_ids + homo span ids -> variant logits)
 │  └─► OnnxNgram          #   char-ngram featurization (pure Python) ->
 │                         #   ngram_core.onnx (EmbeddingBag + 2 MLP heads) ->
 │                         #   exceptions/user-marks/1-syllable rules (pure Python)
 ▼  stressed text with '+' marks
tokenize() + sos/eos      # 47-symbol table -> ids; speaker name -> embedding id
 │                        # durs_rate / pitch_coefs = ones (plain-text prosody)
 ▼
acoustic_v5_ru.onnx        # ids/spk/rates/coefs -> mel (B,192,F) + durations (B,T)
 ▼
vocoder_v5_ru.onnx         # mel -> mag/x/y spectral parts (B,1201,F)
 ▼
istft_48k.onnx             # (mag,x,y) -> waveform; complex math is real-valued
                           # IDFT (float64 core) inside, no ComplexFloat on boundary
 ▼
WAV 48 kHz mono float32 (soundfile)
```

All five `.onnx` files run in shared `onnxruntime` CPU sessions owned by
`SileroTTS`. The only torch-dependent step in the whole system is *producing*
the models (one-off export, outside this repo).

## 2. Architecture decisions

- **Zero-PyTorch runtime.** No `import torch` in `silero_onnx/` or `tests/`;
  enforced by first/last assertions in both test files. Rationale: small
  install, no 2 GB dependency, embeddable deployments.
- **Static data ships in the wheel.** `silero_onnx/data/*.json` (homodict,
  phrases, tokenizer vocab, ngrams, exceptions, symbols, speakers, stress
  config) is declared as setuptools `package-data` and verified inside the
  built wheel. Nothing is downloaded at runtime.
- **Generated artifacts go to the user cache.** The BERT tokenizer directory
  (`vocab.txt` + maps, reconstructed from `tokenizer.json`) is written to
  `~/.cache/silero_onnx/tokdir` (`$SILERO_ONNX_CACHE` overrides), never into
  the package — installed wheels are often read-only. A marker file guards
  completeness; content is deterministic.
- **Dynamic shapes, fixed batch.** All ONNX graphs use dynamic `T`/`F` axes;
  batch is specialized to 1 everywhere (loops/pads that require it are
  single-utterance by construction, documented in code).
- **ORT strictness workarounds (all verified, none speculative).**
  Repetition counts are int64 (`Range` rejects mixed int32/int64);
  complex numbers never cross a graph boundary (split real/imag);
  string/dict control flow stays in Python.
- **Error style.** `SystemExit(2/3/4)` with actionable messages for usage /
  environment / inference failures; `ValueError` for domain errors (unknown
  speaker/symbols). No tracebacks for expected failures, no silent wrong output.

## 3. Files in `silero_onnx/`

- `__init__.py` — public facade only: `SileroTTS`, `main`.
- `engine.py` — `SileroTTS(model_dir, data_dir)`, ORT session loading with
  missing-file diagnostics, `synthesize()` / `synthesize_to_file()`, CLI.
  Model discovery: explicit `model_dir` > `SILERO_ONNX_MODELS` >
  `<repo>/onnx`; no other fallbacks (missing models fail loudly).
  Production must pass `model_dir` explicitly.
- `frontend.py` — `normalize()` (whitespace/abbr/numbers), `tokenize()` (+
  `errors="raise"|"filter"`), baked symbol/speaker tables, `TorchFreeFrontend`
  assembling numpy feeds. Unknown chars are *filtered* (matches the original
  pipeline's silent dropping, unlike the strict default of `tokenize()`).
- `homograph_solver.py` — `OnnxStress` (tokenizer + regex/phrases + ORT batch
  + string assembly) and `FullStress` (homographs-then-ngrams ordering with
  the original skip-list wiring).
- `ngram_accentor.py` — `OnnxNgram`: `<word>` n-grams 1–4 → ids (UNK
  fallback), offsets, ORT core, exceptions/user-marks/1-syllable rules.
- `bert_tokenizer.py` — verbatim vendored stdlib-only WordPiece tokenizer.
- `data/` — runtime JSON tables (see above); `tokdir/` is *generated*, not shipped.

## 4. Project structure

```
silero/                          # repo root
├── silero_onnx/                 # torch-free runtime package (ships in the wheel)
│   ├── __init__.py              # public facade: SileroTTS, main
│   ├── engine.py                # sessions, synthesize(), CLI
│   ├── frontend.py              # normalize/tokenize/TorchFreeFrontend
│   ├── homograph_solver.py      # OnnxStress + FullStress
│   ├── ngram_accentor.py        # OnnxNgram
│   ├── bert_tokenizer.py        # vendored stdlib-only WordPiece
│   └── data/*.json              # runtime tables (shipped as package-data)
├── scripts/                     # one-off build: torch allowed, dormant after build
│   ├── requirements-export.txt  # build-env installer (extra + CPU torch index)
│   ├── export_all.py            # orchestrator: fixtures, then the 5 exporters
│   ├── make_fixtures.py         # [gen] onnx/fixtures/*.pt
│   ├── export_acoustic.py       # [gen] onnx/acoustic_v5_ru.onnx (+ .data)
│   ├── export_vocoder.py        # [gen] onnx/vocoder_v5_ru.onnx
│   ├── export_istft.py          # [gen] onnx/istft_48k.onnx (+ .data)
│   ├── export_stress.py         # [gen] onnx/silero_stress.onnx
│   ├── export_ngram.py          # [gen] onnx/ngram_core.onnx
│   ├── check_models.py          # deployment check (torch-free, no artifacts)
│   ├── _common.py               # gates, L1/L2 helpers, unified result-log format
│   ├── acoustic.py, build_acoustic.py, encoder.py, decoder.py,
│   │   fftblock.py, slowmha.py, predictors.py, istft_model.py,
│   │   layernorm.py             # clean torch ports vendored with the exporters
│   ├── README.md                # build quickstart + per-script table
│   └── DEVELOPMENT.md           # exporter internals, thresholds, maintenance
├── tests/                       # torch-free suites (29 tests, no fixtures)
│   ├── test_production.py
│   └── test_extended_robustness.py
├── onnx/                        # [gen, git-ignored] build output, never committed
│   ├── *.onnx (+ .data)         # the five models
│   ├── fixtures/*.pt            # generated references
│   └── *_result.txt             # per-step gate logs (`# header` → `L1/L2` → `RESULT`)
├── pyproject.toml               # runtime deps (torch-free) + export/dev extras
├── README.md                    # user quickstart (zero → audio)
└── DEVELOPMENT.md               # this file
```

Legend: `[gen]` = generated locally by the named script, never committed
(`onnx/`, caches and packaging outputs are git-ignored); everything else
is versioned source. The per-file generator mapping stays maintainable
by convention, not by vigilance: each `export_<name>.py` writes
`onnx/<name>*.onnx`, and `check_models.py::EXPECTED` is the
machine-checked registry of every artifact the build must produce —
adding a model means one row here, one exporter, one `EXPECTED` entry.

## 5. Development principles

- **Evidence over reasoning.** Every component carries a numeric gate
  (bit-exact where achievable, e.g. `max_abs=0.0` on transplants; ≤1e-4
  end-to-end). A passing suite is the definition of done.
- **Separation of concerns.** Neural math (ONNX) vs text rules (Python) vs
  orchestration (`SileroTTS`) never mix in one module.
- **Thread safety.** ORT sessions are shared across threads (documented
  thread-safe for `run()`); frontend state is read-only after init; lazy
  singletons are warmed before fan-out. Covered by a 6-thread test.
- **Cache isolation.** Anything written at runtime goes to tempdirs or the
  user cache; the repo and the package stay read-clean (tests assert behavior,
  and the suite itself writes WAVs only to `tempfile`).
- **No silent degradation.** Ambiguous cases (boundary homographs, .5 duration
  rounding) are pinned as explicit accepted variants, not hidden tolerances.

## 6. Prospects (ordered, each independently shippable)

1. SSML `<break>` pauses: split the acoustic graph at the len_reg boundary,
   run pause ops (`get_support_ids`/`process_pauses`/`fx_pauses`) in numpy
   between two ONNX calls. All pieces are plain tensor code.
2. 8/24 kHz outputs via the PQMF downsampling path (currently 48 kHz only).
3. GPU execution (CUDA EP) + FP16 revisit: FP16 was measured on CPU and
   rejected (broken loads, destroyed vocoder quality, zero speedup).
4. Batch synthesis (B>1) by generalizing the three single-utterance fast paths.
5. Model regeneration: the export factory lives in `scripts/`
   (`python scripts/export_all.py` from the repo root, needs the torch
   build env); reference fixtures are generated, not versioned
   (`scripts/make_fixtures.py` step 0 → `onnx/fixtures/`, git-ignored).
