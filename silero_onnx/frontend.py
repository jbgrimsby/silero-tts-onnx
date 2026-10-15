"""Clean Python frontend prototype for Silero TTS v5 RU (torch-free ready).

Stages: abbreviations -> numbers -> stress (silero-stress) -> token IDs.
Deliberately has NO torch dependency itself (only stdlib + num2words +
silero-stress at call time); tensor building is one line at the boundary.

Symbol table (47 symbols) and sos/eos convention ('|', '~') match the
upstream v5_ru model (47-symbol table verified at export). SOS/EOS
wrapping is left to the merge stage (as in the original pipeline);
see add_sos_eos().
"""
import os
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

ABBR = {
    # street addresses
    "ул.": "улица", "пр.": "проспект", "пер.": "переулок", "пл.": "площадь",
    "наб.": "набережная", "бул.": "бульвар", "ш.": "шоссе", "туп.": "тупик",
    "д.": "дом", "корп.": "корпус", "кв.": "квартира",
    "оф.": "офис", "эт.": "этаж", "под.": "подъезд",
    # geography / admin
    "с.": "село", "дер.": "деревня", "пос.": "посёлок",
    "р.": "река", "оз.": "озеро", "обл.": "область", "р-н": "район",
    "ст.": "станция",
    # misc frequent
    "т.д.": "так далее", "т.е.": "то есть", "т.к.": "так как",
    "т.н.": "так называемый", "др.": "другой", "им.": "имени",
    "гг.": "годы", "г.г.": "годы", "в.": "век", "вв.": "века",
    "рис.": "рисунок", "табл.": "таблица",
    "руб.": "рублей", "коп.": "копеек", "шт.": "штук",
    "кг.": "килограмм", "гр.": "грамм", "мл.": "миллилитров",
    "л.": "литров", "км.": "километров",
    "мм.": "миллиметров",
    "ч.": "часов", "мин.": "минут", "сек.": "секунд",
}
# Ambiguous ones, resolved by local context (documented heuristic):
#   "стр. 5" (digit follows) -> страница, else строение
#   "г. Москва" (capitalized word follows) -> город, else год
#   "м. Тверская" (known station context is undecidable locally) -> метро
#   only before capitalized words, else метров
#   "с." kept as село (page sense "с." is rare in speech input).
_AMBIG = (
    (re.compile(r"\bстр\.(?=\s*\d)", flags=re.IGNORECASE), "страница"),
    (re.compile(r"\bстр\.", flags=re.IGNORECASE), "строение"),
    (re.compile(r"\bг\.(?=\s*[А-ЯЁ])"), "город"),
    (re.compile(r"\bг\.", flags=re.IGNORECASE), "год"),
    (re.compile(r"\bм\.(?=\s*[А-ЯЁ])"), "метро"),
    (re.compile(r"\bм\.", flags=re.IGNORECASE), "метров"),
    # "См." (смотри) vs "5 см" (сантиметров): digits before -> unit
    (re.compile(r"(?<=\d\s)см\.", flags=re.IGNORECASE), "сантиметров"),
    (re.compile(r"\bсм\.", flags=re.IGNORECASE), "смотри"),
)
_ABBR_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, ABBR), key=len,
                                                  reverse=True)) + r")",
                      flags=re.IGNORECASE)

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
# Minus as numeric sign only (not hyphenation): not preceded by a
# digit/dot/comma/word char, followed by a digit. "5-10" keeps its hyphen.
_MINUS_RE = re.compile(r"(?<![\d.,\w])-(?=\d)")


def expand_abbreviations(text: str) -> str:
    """Replace known Russian abbreviations (case-insensitive, dot-aware).

    Ambiguous forms (стр./г./м.) are resolved by the following token
    (documented heuristic, see _AMBIG); case of the first letter kept.
    """
    def keep_case(m, val):
        return val[0].upper() + val[1:] if m.group(0)[:1].isupper() else val
    for rx, val in _AMBIG:
        text = rx.sub(lambda m: keep_case(m, val), text)

    def rep(m):
        key = m.group(0).lower()
        val = ABBR.get(key, m.group(0))
        return val[0].upper() + val[1:] if m.group(0)[:1].isupper() else val
    return _ABBR_RE.sub(rep, text)


def _num_to_words(token: str) -> str:
    # NOTE: masculine nominative cardinal only ("одна книга", prepositional
    # "в ... году" need gender/case agreement — a known hard problem, out of
    # prototype scope; output stays grammatical for the common cases).
    # Digits have NO symbols in the v5 table, so unexpanded numbers would
    # fail tokenize() — this stage is required, not optional.
    from num2words import num2words as n2w
    t = token.replace(",", ".")
    try:
        if "." in t:
            whole, frac = t.split(".")
            w = n2w(int(whole), lang="ru")
            digits = {"0": "ноль", "1": "один", "2": "два", "3": "три",
                      "4": "четыре", "5": "пять", "6": "шесть", "7": "семь",
                      "8": "восемь", "9": "девять"}
            return w + " целых " + " ".join(digits[d] for d in frac if d.isdigit())
        return n2w(int(t), lang="ru")
    except Exception:
        return token  # leave untouched on any failure (loud fallback below)


def expand_numbers(text: str) -> str:
    text = _MINUS_RE.sub("минус ", text)
    return _NUM_RE.sub(lambda m: _num_to_words(m.group(0)), text)


_accentor = None


def get_accentor():
    global _accentor
    if _accentor is None:
        from silero_stress import load_accentor
        _accentor = load_accentor()
    return _accentor


def apply_stress(text: str, **flags) -> str:
    """Stress + yo + homographs via official silero-stress package."""
    acc = get_accentor()
    kw = dict(put_stress=True, put_stress_homo=True, put_yo=True,
              put_yo_homo=True, stress_single_vowel=True)
    kw.update(flags)
    return acc(text, **kw)


def normalize(text: str) -> str:
    """Deterministic text cleanup: whitespace, abbreviations, numbers (no ML).

    Whitespace (incl. newlines/tabs) collapses to single spaces first, mirroring
    the original pipeline's ``\\s+`` handling; leading/trailing space stripped.
    """
    text = re.sub(r"\s+", " ", text).strip()
    return expand_numbers(expand_abbreviations(text))


def build_symbol_table():
    """Return (symbols, symbol_to_id) exactly as v5_ru.pt uses them.

    Needs torch + the hub model once (reads the live tables, no guessing).
    """
    import torch
    torch._C._jit_set_profiling_mode(False)
    model, _ = torch.hub.load(repo_or_dir="snakers4/silero-models",
                               model="silero_tts", language="ru",
                               speaker="v5_ru", trust_repo=True)
    part = model.packages[0]
    return list(part.symbols), dict(part.symbol_to_id)


def tokenize(text: str, symbol_to_id, sos_token="|", eos_token="~",
             add_sos_eos=False, errors="raise") -> list:
    """Map cleaned+stressed text to symbol ids.

    Space is a regular symbol and is mapped (not dropped). Unknown
    characters: errors="raise" (default, loud) or "filter" (drop silently,
    like the original prepare_text_input regex does).
    """
    seq = text.lower().replace("—", "–")
    ids, unknown = [], []
    for i, ch in enumerate(seq):
        if ch in symbol_to_id:
            ids.append(symbol_to_id[ch])
        elif errors == "filter":
            continue
        else:
            unknown.append((i, ch))
    if unknown:
        raise ValueError(f"unknown symbols at {unknown[:8]} (use errors='filter' "
                         f"for original-like silent dropping)")
    if add_sos_eos:
        ids = [symbol_to_id[sos_token]] + ids + [symbol_to_id[eos_token]]
    return ids


def add_sos_eos(ids: list, symbol_to_id, sos_token="|", eos_token="~") -> list:
    return [symbol_to_id[sos_token]] + list(ids) + [symbol_to_id[eos_token]]


def prepare(text: str, symbol_to_id, stress=True):
    """Full prototype chain: normalize -> stress -> token ids."""
    t = normalize(text)
    if stress:
        t = apply_stress(t)
    return tokenize(t, symbol_to_id)


def load_symbol_table(data_dir=DATA_DIR):
    """Torch-free symbol table (dumped once from the live model)."""
    import json
    with open(os.path.join(data_dir, "symbols.json"), encoding="utf-8") as f:
        d = json.load(f)
    return d["symbols"], d["symbol_to_id"]


def load_speaker_map(data_dir=DATA_DIR):
    """Torch-free speaker name -> embedding id (dumped once, verified)."""
    import json
    with open(os.path.join(data_dir, "speakers.json"), encoding="utf-8") as f:
        d = json.load(f)
    return {k: v["id"][0] for k, v in d.items()}


class TorchFreeFrontend:
    """End-to-end torch-free text frontend (numpy in / numpy out).

    normalize -> FullStress (homosolver+ngram ONNX) -> tokenize (+sos/eos)
    -> model feed dict for acoustic_v5_ru.onnx. No torch at import or run.
    """

    def __init__(self, models_dir=None, data_dir=DATA_DIR):
        from .homograph_solver import FullStress
        try:
            self.stress = FullStress(models_dir=models_dir, data_dir=data_dir)
            _, self.symbol_to_id = load_symbol_table(data_dir)
            self.speakers = load_speaker_map(data_dir)
        except (OSError, ValueError, KeyError) as e:
            raise SystemExit(
                f"silero-onnx: error: cannot load frontend data from "
                f"'{data_dir}': {e}")

    def prepare(self, text, speaker="xenia"):
        """Text -> dict of numpy arrays matching acoustic ONNX inputs."""
        import numpy as np
        if speaker not in self.speakers:
            raise ValueError(f"unknown speaker '{speaker}', want one of "
                             f"{sorted(self.speakers)}")
        t = normalize(text)
        t = self.stress(t)
        # errors="filter": drop unknown chars silently, exactly like the
        # original prepare_text_input regex (emoji/latin/specials never crash).
        ids = add_sos_eos(tokenize(t, self.symbol_to_id, errors="filter"),
                          self.symbol_to_id)
        seq = np.array([ids], dtype=np.int64)
        n = seq.shape[1]
        return {"sequence": seq,
                "speaker_ids": np.array([self.speakers[speaker]], dtype=np.int64),
                "durs_rate": np.ones((1, n), dtype=np.float32),
                "pitch_coefs": np.ones((1, n), dtype=np.float32)}
