"""Opt-in layered fixed-voice multilingual support; torch-free.

No automatic numbers/letter names, SSML, stress or Indic romanization.
Original ports: Vladimir Govorov; models: Silero Team. Multilingual adaptation:
Jim Grimsby Jr. with AI assistance. See docs/multilingual.md and LICENSE.
"""
from pathlib import Path
import threading
from .frontend import FrontendRegistry
from .frontend.fixed import TextFrontend_Fixed
from .languages import LanguageRegistry
from .languages.fixed import Language_Fixed
from .models import ModelRegistry
from .models.fixed import CHECKPOINTS, Version_Fixed
from .strategies import StrategyFactory
from .strategies.fixed import Strategy_Fixed

_REGISTRATION_LOCK = threading.RLock()


def register_multilingual():
    """Register additive layers without replacing Russian defaults or CLI IDs."""
    with _REGISTRATION_LOCK:
        for registry, key, expected in ((LanguageRegistry, 'fixed', Language_Fixed),
                                        (FrontendRegistry, 'fixed', TextFrontend_Fixed)):
            current = registry._factories.get(key)
            if current is not None and current is not expected:
                raise ValueError('Registration collision: ' + key)
        for key in CHECKPOINTS:
            if key in ModelRegistry.list() and not isinstance(ModelRegistry.get(key), Version_Fixed):
                raise ValueError('Model registration collision: ' + key)
            if key in StrategyFactory.list() and StrategyFactory.get(key) is not Strategy_Fixed:
                raise ValueError('Strategy registration collision: ' + key)
        if 'fixed' not in LanguageRegistry.list():
            LanguageRegistry.register('fixed', Language_Fixed)
        if 'fixed' not in FrontendRegistry.list():
            FrontendRegistry.register('fixed', TextFrontend_Fixed)
        for key in CHECKPOINTS:
            if key not in ModelRegistry.list():
                ModelRegistry.register(Version_Fixed(key))
            if key not in StrategyFactory.list():
                StrategyFactory.register(key, Strategy_Fixed)


class MultilingualRuntime:
    """Backward-compatible facade through the same EngineFactory as Russian."""
    prepare_data = staticmethod(TextFrontend_Fixed.prepare_data)

    def __init__(self, directory, lower_memory=False):
        from .engine import EngineFactory
        register_multilingual()
        self.directory = Path(directory)
        language = Language_Fixed(models_dir=self.directory)
        self._engine = EngineFactory.create_engine(language_id='fixed', version_id=language.bundle_id,
            model_dir=str(self.directory), lower_memory=lower_memory)
        self.data = self._engine.language.data
        self.acoustic = self._engine.sessions['acoustic']
        self.vocoder = self._engine.sessions['vocoder']
        self.istft = self._engine.sessions.get('istft')

    def synthesize(self, text, speaker, lang=None):
        return self._engine.synthesize(text, speaker=speaker, lang=lang)
