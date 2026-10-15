"""Torch-free Silero TTS v5 RU inference core (text -> ONNX(x5) -> audio).

Pipeline: TorchFreeFrontend (normalize + homosolver/ngram stress via 2
ORT sessions) followed by three more ORT sessions (acoustic, vocoder,
ISTFT): five models total, zero torch.
Public entry point is :class:`SileroTTS`; nothing here imports torch
(verified: fresh subprocess + ``sys.modules`` check in CI-style tests).

Typical use:
  tts = SileroTTS(model_dir="path/to/onnx")
  audio = tts.synthesize("Привет, мир!", speaker="xenia")  # float32 mono, 48kHz
"""
import argparse
import os
import sys

import numpy as np

assert "torch" not in sys.modules, "torch must never be imported here"

ACOUSTIC = "acoustic_v5_ru.onnx"
VOCODER = "vocoder_v5_ru.onnx"
ISTFT = "istft_48k.onnx"
DATA_SUFFIX = ".data"
SAMPLE_RATE = 48000


def _dev_models_dir():
    """Locate the ONNX model directory without explicit configuration.

    Only two sources exist, in order: SILERO_ONNX_MODELS env var, then
    the local build output (<repo>/onnx, produced by
    scripts/export_all.py). There are no legacy fallbacks: if neither
    holds the models, callers fail loudly (see SileroTTS). Production
    code must pass model_dir explicitly; this is a developer
    convenience only.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo_onnx = os.path.normpath(os.path.join(here, "..", "onnx"))
    env = os.environ.get("SILERO_ONNX_MODELS") or ""
    if env:
        # Explicit configuration is authoritative: a wrong path fails
        # loudly instead of silently falling back to another directory.
        if os.path.isfile(os.path.join(env, ACOUSTIC)):
            return env
        return None
    if os.path.isfile(os.path.join(repo_onnx, ACOUSTIC)):
        return repo_onnx
    return None


MISSING_MODELS_MSG = (
    "silero-onnx: error: no ONNX models found. Build them once with "
    "the torch build env (python scripts/export_all.py, see "
    "scripts/README.md) into <repo>/onnx/, or point SILERO_ONNX_MODELS "
    "at a directory holding acoustic_v5_ru.onnx + vocoder_v5_ru.onnx + "
    "istft_48k.onnx (+ .data sidecars) + silero_stress.onnx + "
    "ngram_core.onnx")


def _dev_data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_sessions(models_dir):
    """Load the three ORT sessions; fail loudly on missing files."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3  # ERROR only: hide benign optimizer chatter
    # (e.g. ngram EmbeddingBag subgraph renames); load errors still raise.

    def sess(name, need_data):
        base = os.path.join(models_dir, name)
        if not os.path.isfile(base):
            raise SystemExit(f"silero-onnx: error: missing model file {base}")
        if need_data and not os.path.isfile(base + DATA_SUFFIX):
            raise SystemExit(f"silero-onnx: error: missing weights sidecar {base}.data")
        try:
            return ort.InferenceSession(base, sess_options=so,
                                        providers=["CPUExecutionProvider"])
        except Exception as e:
            raise SystemExit(f"silero-onnx: error: cannot load {base}: {e}")

    return (sess(ACOUSTIC, True), sess(VOCODER, False), sess(ISTFT, True))


class SileroTTS:
    """High-level torch-free Russian TTS (v5 voices, 48 kHz, CPU)."""

    def __init__(self, model_dir=None, data_dir=None):
        if model_dir is None:
            model_dir = _dev_models_dir()
        if model_dir is None or not os.path.isdir(model_dir):
            raise SystemExit(
                MISSING_MODELS_MSG + " (SILERO_ONNX_MODELS=%r, "
                "model_dir=%r)" % (os.environ.get("SILERO_ONNX_MODELS"),
                                   model_dir))
        if data_dir is None:
            data_dir = _dev_data_dir()
        from .frontend import TorchFreeFrontend
        self._frontend = TorchFreeFrontend(models_dir=model_dir, data_dir=data_dir)
        try:
            import json
            with open(os.path.join(data_dir, "pause.json"),
                      encoding="utf-8") as f:
                self._sil_value = float(json.load(f)["sil_value"])
        except (OSError, ValueError, KeyError) as e:
            raise SystemExit(
                f"silero-onnx: error: cannot load pause data from "
                f"'{data_dir}': {e}")
        self._sessions = load_sessions(model_dir)
        self._model_dir = model_dir

    @property
    def speakers(self):
        """Available voice names (e.g. aidar baya kseniya eugene xenia)."""
        return sorted(self._frontend.speakers)

    def synthesize(self, text, speaker="xenia", sample_rate=SAMPLE_RATE,
                     ssml=False):
        """Russian text -> (mono float32 audio, durations array).

        ssml=True parses <speak> XML (break/prosody/p/s tags, see
        silero_onnx/ssml.py). Prosody needs one acoustic pass; <break>
        pauses need a second pass with forced durations (the graph has no
        symb_durs input) — see _synthesize_breaks.
        """
        if not text or not text.strip():
            raise SystemExit("silero-onnx: error: empty text: nothing to synthesize")
        if sample_rate != SAMPLE_RATE:
            raise SystemExit(
                f"silero-onnx: error: only sample_rate={SAMPLE_RATE} is supported "
                f"(got {sample_rate})")
        acoustic, vocoder, istft = self._sessions
        if ssml:
            feeds, breaks = self._frontend.prepare_ssml(text, speaker)
        else:
            feeds = self._frontend.prepare(text, speaker)
            breaks = []
        if breaks:
            mel, dur = self._synthesize_breaks(acoustic, feeds, breaks,
                                               self._sil_value)
        else:
            mel, dur = acoustic.run(None, feeds)
        mag, x, y = vocoder.run(None, {"mel": mel.astype(np.float32)})
        audio = istft.run(None, {"mag": mag.astype(np.float32),
                                 "x": x.astype(np.float32),
                                 "y": y.astype(np.float32)})[0]
        return np.ascontiguousarray(audio.squeeze()), dur

    @staticmethod
    def _synthesize_breaks(acoustic, feeds, breaks, sil_value):
        """Two-pass pause forcing + mel silencing (no graph changes needed).

        Pass 1 runs with the SSML rates to get base durations; pass 2
        re-runs with rates rewritten so round(dur/rate) == requested
        frames at each break token (mirrors the upstream in-graph
        dur[sid] = sdur override, which also lands pre-len_reg); then
        _apply_fx_pauses flattens the pause regions in mel (mirrors the
        upstream post-decoder fx_pauses).
        Known edge (documented, not silent): a base duration of 0
        cannot be lifted by rate scaling (0/x == 0) — that pause then
        undershoots. Boundary letters virtually never predict 0.
        """
        mel, dur = acoustic.run(None, feeds)
        rates = np.array(feeds["durs_rate"], dtype=np.float32)
        for pos, frames in breaks:
            base = float(dur[0, pos])
            rates[0, pos] = (base / frames) if (frames > 0 and base > 0) else 1.0
        feeds2 = dict(feeds)
        feeds2["durs_rate"] = rates
        mel, dur = acoustic.run(None, feeds2)
        mel = SileroTTS._apply_fx_pauses(mel, dur, breaks, sil_value)
        return mel, dur

    @staticmethod
    def _apply_fx_pauses(mel, dur, breaks, sil_value):
        """Numpy port of the upstream post-decoder fx_pauses mel surgery.

        For each (pos, frames): the sum of final durations before pos
        locates the pause in mel frames; the region is flattened to
        sil_value (upstream keeps the first 10 frames for pauses over
        10 frames). Runs on the (1,192,F) mel between the acoustic and
        vocoder sessions.
        Documented deviation: breaks on comma positions additionally
        re-predict durations upstream (batch-2 support path with
        pause-swapped tokens); here they take the same override+silence
        as other positions. Measured parity impact is small (see the
        SSML parity probe); exact support-path port would need a graph
        re-export for batch-2 duration prediction.
        """
        mel = np.array(mel, dtype=np.float32, copy=True)
        row = np.asarray(dur[0], dtype=np.float64)
        for pos, sdur in breaks:
            start = int(row[:pos].sum())
            if sdur > 10:
                mel[:, :, start + 10:start + sdur] = sil_value
            else:
                mel[:, :, start:start + sdur] = sil_value
        return mel

    def synthesize_to_file(self, text, path, speaker="xenia",
                           sample_rate=SAMPLE_RATE, ssml=False):
        """Like synthesize(), but writes a WAV file; returns its path."""
        audio, _dur = self.synthesize(text, speaker, sample_rate, ssml=ssml)
        outdir = os.path.dirname(os.path.abspath(path))
        try:
            os.makedirs(outdir or ".", exist_ok=True)
        except Exception as e:
            raise SystemExit(f"silero-onnx: error: cannot create '{outdir}': {e}")
        try:
            import soundfile as sf
        except ImportError:
            raise SystemExit("silero-onnx: error: soundfile is required "
                             "(pip install soundfile)")
        try:
            sf.write(path, audio, sample_rate)
        except Exception as e:
            raise SystemExit(f"silero-onnx: error: cannot write '{path}': {e}")
        return path


def _use_color():
    """ANSI colors only for live terminals that allow them."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _paint(code, s):
    return f"\033[{code}m{s}\033[0m" if _use_color() else s


class ColorHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """argparse help with section/flag coloring (tty only, via _paint).

    --speakers output stays plain on purpose (machine-readable list).
    """

    def format_help(self):
        import re
        text = super().format_help()
        if not _use_color():
            return text
        text = re.sub(r"(?m)^usage:", "\033[33musage:\033[0m", text)
        text = re.sub(r"(?m)^(positional arguments|options|examples):",
                      "\033[33m\\1:\033[0m", text)
        text = re.sub(r"(--[a-z-]+)", "\033[32m\\1\033[0m", text)
        text = re.sub(r"\"[^\"]*\"", lambda m: "\033[36m" + m.group(0)
                      + "\033[0m", text)
        return text


def examples_text():
    """Structured, color-aware usage examples (see --examples)."""
    c = _use_color()

    def H(s):  # yellow: section headers
        return f"\033[33m{s}\033[0m" if c else s

    def F(s):  # green: flags
        return f"\033[32m{s}\033[0m" if c else s

    def Q(s):  # cyan: quoted text
        return f"\033[36m{s}\033[0m" if c else s

    def G(s):  # gray: comments
        return f"\033[90m{s}\033[0m" if c else s

    def W(s):  # red: warnings
        return f"\033[31m{s}\033[0m" if c else s

    return "\n".join([
        H("📌 Basic"),
        f"  silero-onnx {Q('\"Привет! Меня зовут Ксения.\"')} "
        f"{F('-o')} out.wav",
        "",
        H("🎙️ Voices"),
        f"  silero-onnx {F('--speakers')}",
        f"  silero-onnx {Q('\"Замок был открыт.\"')} {F('-o')} castle.wav "
        f"{F('--speaker')} baya",
        "",
        H("✍️ Stress marks (+) go BEFORE the stressed vowel and win"),
        f"  {G('# castle vs lock: з+амок / зам+ок (system convention, see homodict)')}",
        f"  silero-onnx {Q('\"З+амок был открыт.\"')} {F('-o')} castle1.wav "
        f"{F('--speaker')} baya",
        f"  silero-onnx {Q('\"Зам+ок был открыт.\"')} {F('-o')} castle2.wav "
        f"{F('--speaker')} baya",
        "",
        H("⏱️ Punctuation shapes delivery (measured, not decorative)"),
        f"  silero-onnx {Q('\"Подожди... Продолжаю.\"')} {F('-o')} dots.wav "
        f"{G('# ... resculpts phrasing')}",
        f"  silero-onnx {Q('\"Он — молодец.\"')} {F('-o')} dash.wav "
        f"{G('# em-dash adds a beat')}",
        "",
        H("⚡ SSML (needs --ssml; <speak> root is mandatory)"),
        f"  {G('<break> — exact pauses (ms, whole seconds, strength):')}",
        f"  silero-onnx {F('--ssml')} "
        f"{Q('\"<speak>Сейчас будет пауза. <break time=\'500ms\'/> Вот она была.</speak>\"')} "
        f"{F('-o')} br.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q('\"<speak>Сейчас будет пауза. <break time=\'2s\'/> Вот она была.</speak>\"')} "
        f"{F('-o')} br2.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q('\"<speak>Сейчас будет пауза. <break strength=\'weak\'/> Вот она была.</speak>\"')} "
        f"{F('-o')} br3.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q('\"<speak>Первая часть. <break strength=\'x-strong\'/> Вторая часть после очень длинной паузы.</speak>\"')} "
        f"{F('-o')} br4.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q('\"<speak>Раз. <break time=\'250ms\'/> Два. <break strength=\'strong\'/> Три.</speak>\"')} "
        f"{F('-o')} br5.wav",
        f"  {G('<prosody> — the speech announces itself:')}",
        f"  silero-onnx {F('--ssml')} "
        f"{Q('\"<speak><prosody rate=\'fast\'>Я говорю быстро-быстро!</prosody> <prosody rate=\'slow\'>А теперь медленно и размеренно.</prosody>А теперь как обычно.</speak>\"')} "
        f"{F('-o')} pr.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q('\"<speak><prosody pitch=\'high\'>Здесь я говорю высоко.</prosody> <prosody pitch=\'low\'>А здесь низко.</prosody></speak>\"')} "
        f"{F('-o')} pi.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q('\"<speak><prosody rate=\'slow\' pitch=\'low\'>Говорю медленно и низко.</prosody></speak>\"')} "
        f"{F('-o')} pi2.wav",
        f"  silero-onnx {F('--ssml')} "
        f"{Q('\"<speak><prosody rate=\'slow\'>Говорю медленно, <prosody pitch=\'high\'>а это слово высоко</prosody> и снова ровно.</prosody></speak>\"')} "
        f"{F('-o')} pi3.wav {G('# nested prosody')}",
        f"  {G('<p> and <s> — paragraph/sentence breaks (x-strong/strong):')}",
        f"  silero-onnx {F('--ssml')} "
        f"{Q('\"<speak><s>Первое.</s><s>Второе.</s></speak>\"')} "
        f"{F('-o')} ps.wav",
        W("  NOTE: without --ssml, tags are plain text — <break .../> gets "
          "vocalized (numbers inside expand), it never pauses."),
        "",
        H("📁 Custom models"),
        f"  silero-onnx {F('--models')} D:\\models\\silero "
        f"{Q('\"Текст.\"')} {F('-o')} out.wav",
    ])


def main(argv=None):
    # Examples contain Cyrillic: force UTF-8 so --help never dies on a
    # cp1252 console (same hardening as scripts/_common.py).
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    del _s
    ap = argparse.ArgumentParser(
        prog="silero-onnx",
        usage="%(prog)s [options] [text]",
        description="Torch-free Silero v5 RU TTS (48 kHz)",
        epilog="run `silero-onnx --examples` for usage examples",
        formatter_class=ColorHelpFormatter)
    ap.add_argument("text", nargs="?", help="Russian text to synthesize")
    ap.add_argument("-o", "--output", default="out.wav", help="output WAV path")
    ap.add_argument("--speaker", metavar="SPEAKER_NAME", default="xenia",
                      help="voice name (see --speakers)")
    ap.add_argument("--models", metavar="MODELS_DIR", default=None,
                      help="dir with the 5 .onnx files (+ .data sidecars)")
    ap.add_argument("--speakers", action="store_true", help="list voices and exit")
    ap.add_argument("--ssml", action="store_true",
                      help="parse text as SSML (<speak>, break/prosody/p/s)")
    ap.add_argument("--examples", action="store_true",
                      help="print usage examples and exit")
    a = ap.parse_args(argv)
    if a.examples:
        print(examples_text())
        return
    tts = SileroTTS(model_dir=a.models)
    if a.speakers:
        print(" ".join(tts.speakers))
        return
    if not a.text:
        raise SystemExit("silero-onnx: error: no text given (see --help)")
    try:
        tts.synthesize_to_file(a.text, a.output, a.speaker, ssml=a.ssml)
    except ValueError as e:
        # Expected input failures (bad SSML, unknown speaker/symbols):
        # clean one-line message, never a traceback.
        raise SystemExit(f"silero-onnx: error: {e}")
    import soundfile as sf
    info = sf.info(a.output)
    print(_paint("32", f"saved {a.output}: {info.duration:.2f}s, {a.speaker}"))
    assert "torch" not in sys.modules, "torch leaked into torch-free runtime!"


if __name__ == "__main__":
    main()
    assert "torch" not in sys.modules, "torch leaked!"
