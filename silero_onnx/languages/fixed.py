"""Literal checkpoint data; no sessions or model imports.

The opt-in ``fixed`` namespace leaves stress-aware Russian ``ru`` unchanged.
Hosts own number/letter expansion, romanization and non-Russian stress marks.
"""
import json
from pathlib import Path
import re
from .base import AbstractLanguage


def prepare_ids(data, text, speaker, lang=None):
    """Preserve the original multilingual checkpoint's tokenization rules."""
    if speaker not in data['speaker_to_id'] or speaker == 'random':
        raise ValueError('Unknown fixed voice: ' + speaker)
    if data['phons']:
        raise NotImplementedError('Phonetic input not ported')
    if data['modern']:
        text = text.strip()
        text = re.sub(r'\n', lambda m: ' ' if m.start() > 0 and text[m.start()-1] in ',;:.!?' else '. ', text)
    text = text.lower().replace('—', '–').replace('‑', '-')
    if data['modern']:
        ext = data['ext_alph'] or {}
        lang = lang or speaker.split('_')[0]
        letters = {c for c in text if c.isalpha()}
        if letters and not letters <= set(data['alphabet']) and lang in ext:
            text = ''.join(ext[lang].get(c, c) for c in text)
        text = re.sub(r'[^{}]'.format(data['symbols'][3:] + r'\^'), '', text)
        text = re.sub(r'\s+', ' ', text.replace('^', ' ^')).strip().replace('^ ', '^')
        text = re.sub(r'\^+', '^', text).strip()
        if text.endswith('^'):
            text = text[:-1].strip()
        clean = re.sub(r'[^а-я\- ]', '', text)
    else:
        text = re.sub(r'[^{}]'.format(data['symbols'][3:]), '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = ''.join(data['symb_ascii_dict'].get(c, c) for c in text)
        clean = re.sub(r'[^a-z1-9\- ]', '', text)
    if not clean.replace(' ', ''):
        raise ValueError('Original frontend has no supported text')
    ids = [data['symbol_to_id'][c] for c in data['sos'] + text + data['eos']]
    if len(ids) > 1002:
        raise ValueError('Input exceeds the exported 1000-symbol text limit; chunk it first')
    return ids, text


class Language_Fixed(AbstractLanguage):
    language_id = 'fixed'
    requires_stress = False  # Supplied marks are literal; no automatic stress.

    def __init__(self, models_dir=None, data_dir=None):
        if models_dir is None:
            raise ValueError('Fixed frontend requires an explicit model directory')
        self.data = json.loads((Path(models_dir) / 'frontend.json').read_text(encoding='utf8'))
        self.bundle_id = self.data['id']
        if self.data.get('sample_rate') != 48000:
            raise ValueError('This runtime supports only the validated 48000-Hz exports')
        if not self.get_speakers():
            raise ValueError('Fixed frontend requires a nonempty voice table')

    def normalize(self, text):
        return text.lower().replace('—', '–').replace('‑', '-')

    def get_symbol_table(self):
        return list(self.data['symbols']), dict(self.data['symbol_to_id'])

    def get_speakers(self):
        return {name: value for name, value in self.data['speaker_to_id'].items() if name != 'random'}
