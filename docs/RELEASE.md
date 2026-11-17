# Model files

[Overview](../README.md) · [Model card](MODEL_CARD.md) · [Generation](GENERATION.md)

## v1.0.0

The trained model is included in the repository:

| File | Contents |
| --- | --- |
| [`models/pocketgpt-v1.pt`](../models/pocketgpt-v1.pt) | Inference weights, model configuration, and vocabulary |
| [`models/pocketgpt-v1.json`](../models/pocketgpt-v1.json) | Model metadata, file size, checksum, and backup download URL |
| [`models/SHA256SUMS`](../models/SHA256SUMS) | SHA-256 checksum for the weight file |

The same weights are also available in the [v1.0.0 release](https://github.com/qianrong0709/PocketGPT/releases/tag/v1.0.0). A separate download is unnecessary after cloning the current repository.

## Check the model

```bash
python scripts/download_model.py --verify-only
python -m pocketgpt inspect-checkpoint models/pocketgpt-v1.pt
```

If the model file is missing, restore it from the release:

```bash
python scripts/download_model.py
```

The script saves the file to `models/pocketgpt-v1.pt` and checks its SHA-256 checksum. An existing file is verified rather than overwritten.

See the [model card](MODEL_CARD.md) for architecture and pocket compatibility. For training continuation, use a full training checkpoint with optimizer and RNG state.

## Licensing

Contact the authors for code and model usage permissions not covered by an applicable license. Third-party encoder code, weights, and preprocessing tools retain their own licensing terms.
