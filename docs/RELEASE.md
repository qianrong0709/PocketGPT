# Release status and readiness

[Back to PocketGPT](../README.md)

## Current scope

The default branch contains the refactored development package in `pocketgpt/`, synthetic regression tests, portable configuration examples, usage documentation, and the unchanged legacy research scripts in `src/`. Newly trained checkpoints have not been released. No model-download link or one-command reproduction claim is provided before matching artifacts are available.

The code upload leaves all existing `src/` files unchanged. It does not alter repository visibility, choose a software license, publish training data, or modify the running training job. The generator core was copied unchanged from the training workbench; personal server launchers were replaced by an explicit, configurable Slurm example.

## Planned release contents

The current development snapshot includes the generator package, condition masks, six portable configurations, synthetic tests, installation instructions, and a configurable Slurm wrapper. A validated model release additionally needs:

- A versioned generator package with explicit condition selection and consistent missing-label masks.
- Portable training/inference configurations, with no machine-specific paths or credentials.
- Separate, documented generator and legacy pocket-encoder environments, including third-party license requirements.
- A small, redistribution-permitted example and a complete command from user inputs to generated SMILES.
- Checkpoint files, SHA256 checksums, exact code revision, vocabulary, model configuration, and feature/encoder provenance.
- A model card covering training stages, data provenance and availability, evaluation scope, supported conditions, and limitations.
- Reproducible evaluation commands and a clear distinction between paper results and newly measured checkpoint results.

## Validation gates

- [ ] Complete conditional pretraining and review train/validation learning curves and checkpoint integrity.
- [ ] Complete cached-vector fine-tuning with properties, a single fragment, and six-type IFP supervision.
- [ ] Evaluate generation validity, uniqueness, diversity, property control, fragment retention, and independently assessed pocket/IFP behavior.
- [ ] Measure the requested condition combinations; do not infer their quality from mask unit tests alone.
- [ ] Verify compatibility between newly encoded raw pockets and the cached descriptors used during training.
- [ ] Verify a real protein/pocket input through feature preparation, caching, checkpoint loading, and generation.
- [ ] Run a fresh-environment installation and inference test from the exact release revision.
- [ ] Label complex-grouped validation accurately; do not claim protein-held-out generalization without an appropriate split.
- [ ] Review the publication's supplementary methods before claiming exact architectural or hyperparameter reproduction.

Training and synthetic/regression tests alone do not complete these gates. The frozen encoder adapter is an automatic workflow, not joint differentiable PDB-to-generation training.

## Artifact publication checks

- [ ] Obtain the authors'/rights holders' approval for the chosen code and checkpoint licenses. Do not assign MIT, Apache, CC BY, or another license by assumption.
- [ ] Review source-dataset, third-party code, preprocessing-tool, and pretrained-encoder redistribution terms separately.
- [ ] Inspect the candidate release for API keys, tokens, credentials, personal information, internal paths, research logs, and unintended data files.
- [ ] Review tracked legacy files and relevant Git history before making the repository public. If a credential is found, arrange revocation/rotation; removing it from a later commit is not enough.
- [ ] Publish only explicitly approved code, documentation, examples, and checkpoint artifacts. Do not upload private CSVs, protein structures, experiment outputs, or environment/cache directories by default.
- [ ] Choose a supported distribution route for large checkpoints and publish integrity metadata; do not commit arbitrary training output directories into Git.
- [ ] Keep repository visibility unchanged until the owner explicitly authorizes a public release.

No external upload, model hosting, CI run on research data, or repository visibility change is authorized merely by this checklist.

## What users should expect at release

A user should be able to follow one version-matched installation guide, obtain one identified checkpoint, provide a selected pocket and optional constraints, and run a documented generation command. They should not need access to the authors' server paths or convert a training CSV to JSONL just to request molecules.

Until that path has been validated and released, the project page and paper describe the research, while this repository's status section describes software availability.
