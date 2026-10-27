"""Literal multilingual strategy; sessions/frontend/version are injected."""
import numpy as np
from .base import AbstractTTSStrategy, SAMPLE_RATE


class Strategy_Fixed(AbstractTTSStrategy):
    def synthesize(self, text, speaker='xenia', sample_rate=SAMPLE_RATE, ssml=False, lang=None):
        self._validate_request(text, speaker, sample_rate, ssml)
        return self._synthesize(text, speaker, ssml=ssml, lang=lang)

    def _synthesize(self, text, speaker, ssml=False, lang=None):
        feeds = self._feeds.acoustic_feeds(**self._frontend.prepare(text, speaker, lang))
        mel, durations = self._sessions['acoustic'].run(None, feeds)
        if 'istft' in self._sessions:
            audio = self._vocode(mel)
        else:
            audio = self._sessions['vocoder'].run(None, {'mel': mel})[0]
        audio = np.asarray(audio).reshape(-1)
        if not np.isfinite(audio).all() or not np.isfinite(durations).all():
            raise ValueError('Model returned non-finite audio or durations')
        return audio, durations
