# Multilingual contribution work

- Preserve the existing Russian strategies and source baseline. Multilingual
  fixed-voice support is additive and does not claim complete SSML/stress parity.
- Runtime and tests stay torch-free. PyTorch is export-time only.
- Model binaries, executable checkpoints, generated audio, logs and private
  NVDA configurations never enter Git. Release artifacts need explicit approval.
- Retain component-specific model licenses and attribution, hashes and sample
  provenance. Do not imply conversion parity is native-language approval.
- Serialize inference; two CPU threads and one inter-op for reproducibility.
- Public evidence must be path-redacted and distinguish retained observations
  from current-branch checks. No user voice banks or raw private logs.
- Publication needs explicit repository-owner approval for the exact artifacts
  and destination. Local validation and preservation commits are separate.
- MultilingualRuntime is a compatibility facade through EngineFactory. Keep
  literal tokenization/data in languages/fixed.py, tensor assembly in
  frontend/fixed.py, declarative topology in models/fixed.py and inference in
  strategies/fixed.py. Do not recreate a standalone session pipeline.
- Multilingual registrations are opt-in and idempotent. Preserve Russian ru,
  default v5_6_ru and the existing Russian CLI version surface. Bundle IDs must
  match selected versions before sessions open. The fixed namespace is not
  language detection or automatic numbering/stress/romanization.
- Session memory policy belongs to ort.py. BASIC optimization is declared only
  for the fixed spectral iSTFT artifact; do not downgrade acoustic/vocoder or
  change the existing Russian default policy. Preserve both memory modes.
- After layered changes run the full suite with --require-models when Russian
  assets are available, and check_multilingual_architecture.py against all ten
  retained exports and original waveform fixtures in both memory modes. Keep
  graph hashes; code adaptation alone does not justify reconversion.
