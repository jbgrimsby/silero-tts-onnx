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

- 5 voices: `aidar baya kseniya eugene xenia`, 48 kHz, CPU.
- Runtime deps only: `onnxruntime`, `numpy`, `soundfile`, `num2words`.
- Fast: RTF ~21x on CPU (4.15 s of audio in ~0.20 s wall time).
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

Inference needs a directory with five ONNX models (plus `.data` sidecars
where noted). Pass it explicitly or via the `SILERO_ONNX_MODELS` variable:

- `acoustic_v5_ru.onnx` + `.onnx.data`
- `vocoder_v5_ru.onnx`
- `istft_48k.onnx` + `.onnx.data`
- `silero_stress.onnx` (homographs)
- `ngram_core.onnx` (plain-word stress)

These are produced once by the export pipeline (see `DEVELOPMENT.md`);
they are data, not code, and are not part of the wheel.

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
silero-onnx --help   # --models, --output, --speaker
```

## Tests

```bash
py -3.12 -m pip install -e ".[dev]"
py -3.12 -m pytest tests/ -v   # 29 tests, ~15 s, torch-free guaranteed
```

## Notes

- Single utterance per call (batch = 1); very long inputs follow the
  original ~1000-symbol guidance.
- Plain text only (punctuation pauses `. , ! ?` work); no SSML tags.
- Output length can differ by ±1 hop-frame (12.5 ms, inaudible) between
  runs on texts whose durations land exactly on .5 rounding boundaries;
  content still matches to ~1e-4. Boundary homographs (e.g. мел/мёл)
  may resolve either valid way for the same reason.
