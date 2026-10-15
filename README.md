# silero-onnx-tts

100% PyTorch-independent Russian text-to-speech: Silero TTS v5 voices,
executed entirely in ONNX Runtime. Text in, WAV out — no `torch` anywhere
in the runtime process (asserted by the test suite via `sys.modules`).

## Quickstart: from zero to audible speech (copy-paste)

```bash
cd silero
# 1. One-off build: export the five ONNX models into onnx/ (needs torch,
#    hence a separate venv; the runtime itself stays torch-free).
py -3.12 -m venv .venv-build
.venv-build\Scripts\activate
pip install -r scripts/requirements-export.txt
$env:PYTHONUTF8 = '1'   # Windows only: torch prints ✅/❌, cp1252 console crashes
                        # without UTF-8 mode. Session-only: stays after
                        # deactivate until the shell closes (Remove-Item Env:PYTHONUTF8 to unset)
python scripts/export_all.py
python scripts/check_models.py   # expect "all 5 models + smoke test passed"

# 2. Runtime install (your normal env, no torch here).
deactivate
py -3.12 -m pip install -e .

# 3. Speak: text in, WAV out (onnx/ is found automatically).
silero-onnx "Привет! Меня зовут Ксения." -o out.wav --speaker xenia
```

Play `out.wav` in any player, or stream straight to speakers:

```python
import sounddevice as sd   # pip install sounddevice
from silero_onnx import SileroTTS
tts = SileroTTS()          # onnx/ picked up automatically
audio, _ = tts.synthesize("Замок был открыт.", speaker="xenia")
sd.play(audio, 48000); sd.wait()
```

Build details live in `scripts/README.md`, internals in `DEVELOPMENT.md`.

## Features

- 5 voices: `aidar`, `baya`, `kseniya`, `eugene`, `xenia`, 48 kHz, CPU.
- Runtime deps only: `onnxruntime`, `numpy`, `soundfile`, `num2words`.
- Fast: RTF ~25x on CPU (4.15 s of audio in ~0.18 s wall time) — on par
  with torch CPU on the same texts (31x/28x and 31x/23x measured), while
  dropping the ~2 GB torch dependency entirely.
- Smart frontend: abbreviations, numbers (incl. negative/fractional),
  automated stress, yo-letter and homograph disambiguation.
- Measured quality vs the original pipeline: max abs error ~1e-4 on waveforms
  (9.4e-05 measured torch-chain vs ONNX-chain, T32 reference).

## Install

```bash
py -3.12 -m pip install -e .     # runtime only: onnxruntime/numpy/soundfile/num2words
```

vs the build env (separate venv, torch allowed, one-off model export):

```bash
pip install -r scripts/requirements-export.txt   # torch/onnx/onnxscript/… + CPU torch index
```

| | `pip install -e .` | `pip install -r scripts/requirements-export.txt` |
|---|---|---|
| What | the **runtime** (this package) | the **factory** (model exporters) |
| torch | never (stays torch-free) | yes (export needs it) |
| Where | your normal env | a separate `.venv-build` |
| When | always (to use the voices) | once (to build `onnx/`, then dormant) |

Python 3.10–3.12.

## Models

Inference needs a directory with five self-contained ONNX models
(single files, weights inside — no `.data` sidecars):

- `acoustic_v5_ru.onnx` (~31 MB) — text tokens + speaker + prosody
  rates → mel-spectrogram (192 dims) + per-token durations;
  the voice itself lives here
- `vocoder_v5_ru.onnx` (~58 MB) — mel → split spectral parts
  (`mag`, `x`, `y`; 1201 dims), no complex numbers on the boundary
- `istft_48k.onnx` (~92 MB) — (`mag`, `x`, `y`) → waveform via a
  real-valued inverse STFT (float64 IDFT core inside)
- `silero_stress.onnx` (~117 MB, homosolver) — picks the right variant
  of homographs (`замок`: castle/lock) with a BERT-tiny over the
  sentence context
- `ngram_core.onnx` (~8 MB, plain-word stress) — EmbeddingBag over
  character n-grams + two MLP heads: where the stress falls and
  where `е` becomes `ё`

These are produced once by the export pipeline (see `DEVELOPMENT.md`);
they are data, not code, and are not part of the wheel. Pass the
directory explicitly or via the `SILERO_ONNX_MODELS` variable.

## Usage

Two modes — same engine underneath, pick the interface:

### A. Python library (embed in your code)

```python
from silero_onnx import SileroTTS
tts = SileroTTS(model_dir="path/to/models")   # or SILERO_ONNX_MODELS, or <repo>/onnx auto
audio = tts.synthesize("Замок был открыт.", speaker="xenia")  # float32 mono, 48 kHz
tts.synthesize_to_file("Привет, мир!", "out.wav", speaker="baya")
print(tts.speakers)   # ['aidar', 'baya', 'eugene', 'kseniya', 'xenia']
```

### B. CLI utility (one-liners, scripts, no Python code)

```bash
silero-onnx "Привет! Меня зовут Ксения." -o out.wav --speaker xenia
silero-onnx --speakers
silero-onnx --help   # --models, --output, --speaker, --ssml
silero-onnx --examples
```

## SSML (`--ssml` / `ssml=True`)

`<speak>` root is mandatory — without it the input is rejected with a
one-line error (no traceback). Plain text without `--ssml` never parses
tags: `<break/>` would be vocalized as words.

| Tag | What it does | Example |
|---|---|---|
| `<break time="500ms"/>` / `<break time="2s"/>` | exact pause (ms, whole seconds) | `…пауза. <break time='500ms'/> Вот…` |
| `<break strength="…"/>` | `x-weak` 25 ms … `x-strong` 1000 ms | `<break strength='weak'/>` |
| `<prosody rate="…">` | `x-slow` 0.5 … `x-fast` 1.5, or `150%` | `<prosody rate='fast'>Скорее!</prosody>` |
| `<prosody pitch="…">` | `x-low` 0.75 … `x-high` 1.25, `robot` 0, or `+12%`/`-10%` | `<prosody pitch='high'>Высоко.</prosody>` |
| `<p>` / `<s>` | paragraph/sentence breaks (`x-strong`/`strong`) | `<s>Первое.</s><s>Второе.</s>` |

How it works: prosody becomes per-token rate/pitch arrays (one acoustic
pass, no model changes); `<break>` pauses force exact durations
(500 ms → 40 frames) in a second pass plus mel silencing. User `+`
stress marks keep working inside SSML.

Limits, all measured: fractional seconds (`1.5s`) fail like upstream;
breaks on commas skip upstream's support re-prediction (the pause
itself stays exact); `silero-onnx --examples` lists runnable samples.

## Tests

```bash
py -3.12 -m pip install -e ".[dev]"
py -3.12 -m pytest tests/ -v   # 58 tests, ~25 s, torch-free guaranteed
```

## Notes

- Single utterance per call (batch = 1); very long inputs follow the
  original ~1000-symbol guidance.
- Plain text only (punctuation pauses `. , ! ?` work); no SSML tags.
- Output length can differ by ±1 hop-frame (12.5 ms, inaudible) between
  runs on texts whose durations land exactly on .5 rounding boundaries;
  content still matches to ~1e-4. Boundary homographs (e.g. мел/мёл)
  may resolve either valid way for the same reason.

## License

This project is licensed under **CC BY-NC-SA 4.0** (see `LICENSE`).
In short: non-commercial use only, credit the authors, share alike.

- Original author: the **Silero Team**
  (https://github.com/snakers4/silero-models).
- This product is a **derivative work** (Adapted Material) based on
  Silero TTS v5, used under CC BY-NC-SA 4.0. What was changed:
  weights transplanted into reimplemented modules and exported to
  ONNX (acoustic/vocoder/ISTFT), stress/ngram cores exported to ONNX,
  frontend and tooling rewritten torch-free.
- Stress/accentor models come from `silero-stress`
  (https://github.com/snakers4/silero-stress) under the MIT license,
  © Silero Team.
