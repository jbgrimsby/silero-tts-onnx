"""All-bundle, two-memory-mode waveform gates through the layered public API.

Use previously exported fixtures; no conversions, downloads or audio playback.
Outputs a private report if requested. Never post unredacted machine paths.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import onnxruntime as ort
import soundfile as sf
from silero_onnx import SileroTTS
from silero_onnx.models.fixed import CHECKPOINTS
from silero_onnx.multilingual import MultilingualRuntime
from silero_onnx.frontend.fixed import TextFrontend_Fixed
from silero_onnx.strategies.fixed import Strategy_Fixed


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundles-root', type=Path, required=True)
    parser.add_argument('--fixtures-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier results')
    report = {'scope': 'Rebased layered API, all10 retained checkpoint bundles; not live NVDA/native review',
              'ort': ort.__version__, 'models': []}
    for checkpoint in CHECKPOINTS:
        directory = args.bundles_root / checkpoint
        fixtures = args.fixtures_root / checkpoint
        data = json.loads((directory / 'frontend.json').read_text(encoding='utf8'))
        cases = json.loads((fixtures / 'export-parity.json').read_text(encoding='utf8'))['cases']
        if not cases:
            raise ValueError('No original-waveform fixtures: ' + checkpoint)
        for name, identity in data['graphs'].items():
            path = directory / name
            if sha(path) != identity['sha256'] or path.stat().st_size != identity['bytes']:
                raise ValueError('Graph identity mismatch: ' + str(path))
        row = {'id': checkpoint, 'frontendSha256': sha(directory / 'frontend.json'),
               'graphs': data['graphs'], 'modes': []}
        report['models'].append(row)
        for low in (False, True):
            tts = MultilingualRuntime(directory, lower_memory=low)
            assert isinstance(tts._engine.frontend, TextFrontend_Fixed)
            assert isinstance(tts._engine.strategy, Strategy_Fixed)
            assert tts._engine.version.version_id == checkpoint
            assert set(tts._engine.speakers) == set(data['speaker_to_id']) - {'random'}
            results = []
            row['modes'].append({'lower_memory': low, 'cases': results})
            for index, case in enumerate(cases):
                started = time.perf_counter()
                audio, durations = tts.synthesize(case['text'], case['speaker'], case.get('lang'))
                path = fixtures / f'reference-{index}.wav'
                reference, rate = sf.read(path, dtype='float32')
                if rate != 48000 or reference.shape != audio.shape:
                    raise ValueError('Original waveform format/length mismatch')
                error = float(np.max(np.abs(audio - reference)))
                if error >= .005 or not np.isfinite(audio).all() or not np.isfinite(durations).all():
                    raise ValueError('Original waveform/finiteness gate failed: ' + checkpoint)
                results.append({'speaker': case['speaker'], 'lang': case.get('lang'),
                                'text': case['text'], 'samples': len(audio),
                                'referenceSha256': sha(path), 'maxWaveDifference': error,
                                'seconds': time.perf_counter() - started})
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
            print(checkpoint, 'lower_memory=' + str(low), len(results), 'original-waveform gates PASS', flush=True)
            del tts
            gc.collect()
        # The general public facade must resolve the same registered layers.
        direct = SileroTTS(model_dir=str(directory), language='fixed', version=checkpoint, lower_memory=True)
        case = cases[0]
        audio, _dur = direct.synthesize(case['text'], speaker=case['speaker'], lang=case.get('lang'))
        reference, _rate = sf.read(fixtures / 'reference-0.wav', dtype='float32')
        error = float(np.max(np.abs(audio - reference)))
        if error >= .005:
            raise ValueError('Public SileroTTS facade mismatch')
        row['publicSileroTtsMaxWaveDifference'] = error
        del direct
        gc.collect()
    report['torchImported'] = 'torch' in sys.modules
    report['passed'] = True
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
