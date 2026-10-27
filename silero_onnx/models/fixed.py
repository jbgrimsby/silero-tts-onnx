"""Opt-in declarative versions for the ten retained fixed-voice exports."""
from .spec import ModelSet, ModelSpec
from .versions import AbstractModelVersion

CHECKPOINTS = ('v3_en', 'v3_de', 'v3_es', 'v3_fr', 'v4_indic', 'v4_cyrillic',
               'v5_turkic', 'v5_caucasian', 'v5_cis_base_nostress', 'v5_cis_ext')


class Version_Fixed(AbstractModelVersion):
    frontend_ids = frozenset({'fixed'})

    def __init__(self, checkpoint):
        if checkpoint not in CHECKPOINTS:
            raise ValueError('Unknown fixed checkpoint: ' + checkpoint)
        self.version_id = checkpoint
        files = {'acoustic': 'acoustic.onnx', 'vocoder': 'vocoder.onnx'}
        if checkpoint.startswith('v5_'):
            files['istft'] = 'istft.onnx'
        self._spec = ModelSpec(checkpoint, 'Silero ' + checkpoint,
            ModelSet(files, tuple(files), ('sequence', 'speaker_ids', 'durs_rate', 'pitch_coefs'),
                     shared_models=frozenset(),
                     basic_optimization_models=frozenset({'istft'}) if 'istft' in files else frozenset()),
            supports_stress=False, supports_ssml=False, supports_pitch=True,
            has_pitch_coefs_input=True)

    @property
    def spec(self):
        return self._spec
