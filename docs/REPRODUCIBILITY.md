# Reproducing and inspecting the project

## Report tables: ready to run

From a clean checkout, run:

```bash
python3 tools/reproduce_report_tables.py
python3 tools/verify_release.py
```

Both commands use the Python standard library. The first computes the three-seed Study-1 means from `primary.csv` and formats the saved Study-2 family estimates and cohort counts. It also checks the reported detection differences. It does not recompute confidence intervals or p-values from per-utterance scores. The second verifies the released file hashes and original-copy provenance.

## Calibration and metric checks

Install `requirements-analysis.txt` and run `python3 tools/check_metrics.py`. These small deterministic checks cover the finite-sample order statistic, insufficient enrollment, tied scores, speaker weighting, ranking direction, and Holm correction. They use the actual released `asvspoof5_metrics.py` and `selfcal_common.py` functions.

## Original experiment scripts

The code in `scripts/` is the implementation retained from the completed experiments. See [CODE_GUIDE.md](CODE_GUIDE.md) for entry points. With one documented exception, copied scripts are byte-identical to the working project versions. The narrative output strings in `validate_selfcal_confirmation.py` were translated into English for this release; its validation calculations were not changed. Original and released hashes are recorded in `provenance/copied_artifacts.json`.

These scripts are not a one-command fresh-training package. They deliberately check hashes, require historical input artifacts, and often create files exclusively to avoid replacing completed experiments. The included `results/` files are an aggregate subset, not the complete execution archive. Do not expect the historical pipeline to run against these summary files alone.

## Requirements for a full inference or training rerun

1. Obtain ASVspoof 2019 LA and the relevant ASVspoof 5 development/evaluation audio and protocols from the organizers. Raw audio is not included here.
2. Obtain AASIST and XLSR-LCNN from the pinned upstream repositories listed in the README. Place them under `external/aasist` and `external/w2v2lcnn`. Obtain their original checkpoints separately.
3. Obtain XLS-R-300M at the revision in `provenance/xlsr_frontend.json` and verify its model and feature-extractor hashes.
4. Restore the historical manifests, source-control checkpoints, feature-cache provenance, threshold records, preregistration, and scoring artifacts referenced by the scripts. The source controls and hash-bound experiment history are required inputs, not reconstructed by the released summary command. A full rerun requires access to the author's original execution archive or a separately specified new experiment.
5. Rebuild the GPU environment using `environment/requirements-gpu.txt` as the package-version reference. The original experiments used Python 3.10 and CUDA-enabled PyTorch. The historical scripts require CUDA and explicitly check model and input integrity.

The original Study-1 sequence is data preparation, audio audit, preflight, training, calibration scoring and threshold freezing, evaluation scoring, analysis, and validation. Study 2 selects previously unused speakers, freezes its rules, scores enrollment, freezes thresholds, then scores test data and runs analysis and validation. Do not remove the integrity checks to make an incomplete archive run.

## Scope of the released evidence

The included CSV and JSON files support inspection of the report's summary values. The portable reproduction command was run during release preparation. The full neural pipeline was not rerun for the course submission. The manuscript audit summary records the earlier 610/610 mapped numerical checks; it is not a fresh rerun, external replication, or publication decision.
