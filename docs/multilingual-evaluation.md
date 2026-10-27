# Multilingual evaluation: scope and limitations

The retained conversion evaluation was performed on Windows with an Intel
Core i5-8400, two inference threads and one inter-op thread. Export used
PyTorch 2.14.0 CPU; the model runtime is torch-free. The lab used ONNX Runtime
1.29/1.30; the separate Sonata portable integration used ORT 1.23.2.

Ten checkpoints were converted: English V3, German V3, Spanish V3, French
V3, Indic V4, Cyrillic V4, Turkic V5, Caucasian V5, CIS base no-stress V5,
and CIS extended V5. This is not Turkish support: Turkic is a family and no
Turkish checkpoint was supplied in this set.

## Conversion evidence (2026-10-16)

The nine-model non-English sweep recorded 360 frontend comparisons, 72
complete original-checkpoint/ONNX language waveform comparisons (maximum
absolute difference 0.00052913), 221 fixed-voice smoke tests, and 464
identical-output comparisons between normal and lower-memory modes.
The English lab recorded five additional original/ONNX waveform comparisons.
These numerical checks establish fidelity for those samples, not universal
quality or language coverage. Source identities, measurements, ASR results
and indexed recordings are in the separate `silero-evaluation-20261016.zip`
handoff asset, not committed generated records.

ASR checked 33 non-English cases and skipped 39 unsupported cases. Only
three had an exact normalized transcript; script mismatches, limited
recognizer coverage and some errors were also present in original-model
audio. ASR is a sanity check, not proof of intelligibility for languages the
reviewer does not speak. No blanket native-language approval is claimed.

Representative paragraph generation rates were roughly 20-22 times real
time for spectral V5, 49-50 for V4 and 10 for Western V3 with normal memory
settings. Lower-memory settings traded throughput for reduced retained
memory. Consult per-checkpoint measurements rather than inferring one
universal rate. These are dated complete-utterance timings, not first PCM.

## Separate NVDA integration (2026-10-16)

The exact internal Sonata 2026.2.37 package was exercised in portable NVDA
2026.1.1, with actual model/voice identity checked before each test. Ten
checkpoint types and English plus fifteen foreign profiles emitted audio.
English words-only Say All, character/word/line navigation, cancellation and
recovery, voice switching and speech-control routing were exercised. These
are integration observations for that exact package, not certification of
the generic Python API or the current Sonata development version.

Limitations observed: unexpanded numeric-only chunks could stop Say All;
some ASR results were inconclusive; voice-manager UI acceptance was not
completed; installation/update/removal, current NVDA beta compatibility,
and all-language native listening were not established by that run. Sonata
text processing subsequently advanced, but this report does not relabel the
older run as a test of newer code. No raw personal NVDA logs are included.

## Contribution revalidation

Current-branch model-free tests, exporter smoke/rebuild checks and retained
model runtime comparisons are recorded separately in the handoff validation
record. They do not silently replace these original measurements.

### Layered architecture rebase (2026-10-26)

The contribution was locally rebased onto upstream architecture revision
`10e5698504e14b39ffd5f72dc2c7b02bd7893747`. Multilingual synthesis now resolves
through the public language/frontend/model/strategy registries and the same
`EngineFactory` as the Russian path. The old convenience API remains a facade.
Existing converted graph bytes and frontend rules were retained; no
reconversion was necessary.

All ten checkpoint bundles passed their 36 retained original-waveform fixtures
in both normal and lower-memory modes: 72 comparisons. Ten additional checks
used the general `SileroTTS(language="fixed", version=...)` API. Maximum absolute
waveform difference was 0.00060463, below the existing 0.005 gate. All declared
fixed speaker names remained available; this did not synthesize every speaker.
The runtime stayed torch-free.

The full suite passed 173 tests, zero skips/failures, with both Russian model
generations present from the official v0.5.0 release. The release ZIP and all
nine extracted graph digests matched the published SHA-256 identities. Tests
used ONNX Runtime 1.29.0 on Windows CPU, two intra-op and one inter-op thread.
The source before this rebase had 39 Russian model tests skipped when assets
were absent; those skips are not counted as this completed run.

This validates the adapted library/API and retained waveform fixtures, not a
new NVDA package, performance benchmark or new linguistic-quality assessment.
No publication, model-hosting change or live Sonata installation accompanied
the local rebase.
