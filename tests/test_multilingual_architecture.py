"""Fixed exports use the new public layers, not a second session pipeline."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from silero_onnx.engine import EngineFactory, SileroTTS
from silero_onnx.frontend import FrontendRegistry
from silero_onnx.frontend.fixed import TextFrontend_Fixed
from silero_onnx.languages import LanguageRegistry
from silero_onnx.languages.fixed import Language_Fixed
from silero_onnx.models import ModelRegistry
from silero_onnx.models.fixed import CHECKPOINTS
from silero_onnx.multilingual import MultilingualRuntime, register_multilingual
from silero_onnx.strategies import StrategyFactory
from silero_onnx.strategies.fixed import Strategy_Fixed
from test_multilingual import data


@pytest.fixture(autouse=True)
def isolated_registries(monkeypatch):
    for registry, attribute in ((LanguageRegistry, '_factories'),
                               (FrontendRegistry, '_factories'),
                               (ModelRegistry, '_versions'),
                               (StrategyFactory, '_strategies')):
        monkeypatch.setattr(registry, attribute, dict(getattr(registry, attribute)))


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    (tmp_path / 'frontend.json').write_text(json.dumps(data()), encoding='utf8')
    for name in ('acoustic.onnx', 'vocoder.onnx'):
        (tmp_path / name).touch()
    calls = []

    class Session:
        def __init__(self, path):
            self.name = Path(path).name

        def run(self, _outputs, feeds):
            calls.append((self.name, feeds))
            if self.name == 'acoustic.onnx':
                return [np.zeros((1, 2, 3), dtype=np.float32), np.ones((1, 3))]
            return [np.ones((1, 12), dtype=np.float32)]

    monkeypatch.setattr('silero_onnx.engine.create_session', Session)
    return tmp_path, calls


def test_registration_is_opt_in_idempotent_and_preserves_russian():
    original = {key: ModelRegistry.get(key) for key in ModelRegistry.list()}
    register_multilingual()
    register_multilingual()
    assert set(CHECKPOINTS) <= set(ModelRegistry.list())
    assert all(ModelRegistry.get(key) is value for key, value in original.items())
    from silero_onnx.versions import DEFAULT_VERSION, VERSIONS
    assert DEFAULT_VERSION == 'v5_6_ru'
    assert set(VERSIONS) == {'v5_ru', 'v5_6_ru'}


def test_registration_collision_is_atomic():
    LanguageRegistry.register('fixed', object)
    before = ModelRegistry.list()
    with pytest.raises(ValueError, match='collision'):
        register_multilingual()
    assert ModelRegistry.list() == before
    assert 'fixed' not in FrontendRegistry.list()


@pytest.mark.parametrize('checkpoint', CHECKPOINTS)
def test_every_checkpoint_has_registered_complete_contract(checkpoint):
    register_multilingual()
    version = ModelRegistry.get(checkpoint)
    assert version.frontend_ids == {'fixed'}
    assert version.acoustic_inputs() == ['sequence', 'speaker_ids', 'durs_rate', 'pitch_coefs']
    assert not version.supports_stress and not version.supports_ssml
    assert ('istft' in version.model_files()) == checkpoint.startswith('v5_')
    assert version.spec.model_set.basic_optimization_models == ({'istft'} if checkpoint.startswith('v5_') else set())


def test_real_factory_and_both_public_facades_use_layers(bundle):
    path, calls = bundle
    register_multilingual()
    direct = EngineFactory.create_engine('fixed', 'v3_en', model_dir=str(path))
    assert isinstance(direct.language, Language_Fixed)
    assert isinstance(direct.frontend, TextFrontend_Fixed)
    assert isinstance(direct.strategy, Strategy_Fixed)
    direct.synthesize('hello', speaker='en_0')
    SileroTTS(model_dir=str(path), language='fixed', version='v3_en').synthesize('hello', speaker='en_0')
    facade = MultilingualRuntime(path)
    facade.synthesize('hello', 'en_0', 'en')
    assert len(calls) == 6
    assert 'torch' not in sys.modules


def test_wrong_checkpoint_pair_fails_before_sessions(bundle, monkeypatch):
    path, _calls = bundle
    register_multilingual()
    monkeypatch.setattr('silero_onnx.engine.create_session', lambda *_a, **_kw: pytest.fail('session opened'))
    with pytest.raises(ValueError, match='does not match'):
        EngineFactory.create_engine('fixed', 'v4_indic', model_dir=str(path))
    with pytest.raises(ValueError, match='does not match'):
        EngineFactory.create_engine('fixed', 'v5_ru', model_dir=str(path))


@pytest.mark.parametrize('text,speaker,kwargs,error', [
    ('', 'en_0', {}, SystemExit), ('hello', 'missing', {}, ValueError),
    ('hello', 'en_0', {'ssml': True}, ValueError),
    ('hello', 'en_0', {'sample_rate': 24000}, SystemExit),
])
def test_shared_request_validation_before_pcm(bundle, text, speaker, kwargs, error):
    path, calls = bundle
    tts = MultilingualRuntime(path)
    with pytest.raises(error):
        tts._engine.synthesize(text, speaker=speaker, **kwargs)
    assert not calls


def test_language_hint_is_request_local(bundle):
    path, calls = bundle
    tts = MultilingualRuntime(path)
    tts.synthesize('hello', 'en_0', 'en')
    tts.synthesize('world', 'en_0')
    assert tts.data == data()
    assert len(calls) == 4
