# Release artifacts

[Overview](../README.md) · [Model card](MODEL_CARD.md) · [Generation](GENERATION.md)

## v1.0.0

The [GitHub release](https://github.com/qianrong0709/PocketGPT/releases/tag/v1.0.0) pairs the generator source with `pocketgpt-v1.pt`, an inference-only export of the selected frozen-backbone fine-tuning checkpoint. The artifact embeds its vocabulary and architecture. `models/pocketgpt-v1.json` records the exact download URL, byte count, and SHA-256 checksum used by `scripts/download_model.py`.

The repository includes portable CPU/CUDA setup instructions, CLI and Python generation examples, condition schemas, frozen-backbone/early-stopping training support, and synthetic regression tests. Legacy research scripts remain in the [archive branch](https://github.com/qianrong0709/PocketGPT/tree/archive/legacy-before-refactor-20261106/src).

Only the inference model is distributed as a release asset; full run directories, optimizer/RNG state, server paths, logs, caches, and research datasets are not part of this artifact. The inference model is not an exact-resume checkpoint.

## Integrity and compatibility

```bash
python scripts/download_model.py
python scripts/download_model.py --verify-only
python -m pocketgpt inspect-checkpoint weights/pocketgpt-v1.pt
```

Keep the source revision, model file, and embedded vocabulary together. Review the [model card](MODEL_CARD.md), especially the historical pocket-descriptor provenance constraint. Third-party encoder source, weights, and preprocessing tools have their own licensing terms and are not bundled with this generator artifact.

No new code/model license is assigned by the packaging process. Contact the authors for permissions not explicitly granted by an applicable license. Research results and publication-level claims should be cited to the paper rather than inferred from an installation or generation smoke test.
