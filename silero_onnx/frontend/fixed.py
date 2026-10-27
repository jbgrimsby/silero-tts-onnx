"""Bundle-defined literal frontend, injected through FrontendRegistry."""
import numpy as np
from .base import AbstractTextFrontend
from ..languages.fixed import prepare_ids


class TextFrontend_Fixed(AbstractTextFrontend):
    frontend_id = 'fixed'

    def __init__(self, language, data_dir=None, supports_pitch=True):
        super().__init__(language)

    @property
    def speakers(self):
        return self.language.get_speakers()

    @property
    def symbol_to_id(self):
        return self.language.get_symbol_table()[1]

    def text_to_ids(self, text):
        raise ValueError('Fixed frontend tokenization requires an explicit speaker')

    @staticmethod
    def prepare_data(data, text, speaker, lang=None):
        ids, _clean = prepare_ids(data, text, speaker, lang)
        sequence = np.array([ids], dtype=np.int64)
        return {'sequence': sequence,
                'speaker_ids': np.array([data['speaker_to_id'][speaker]], dtype=np.int64),
                'durs_rate': np.ones(sequence.shape, dtype=np.float32),
                'pitch_coefs': np.ones(sequence.shape, dtype=np.float32)}

    def prepare(self, text, speaker='xenia', lang=None):
        return self.prepare_data(self.language.data, text, speaker, lang)

    def prepare_ssml(self, ssml_text, speaker='xenia'):
        raise ValueError('SSML is not supported for fixed multilingual exports')
