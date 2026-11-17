# CS 746 Project: Audio Anti-Spoofing with Genuine-Only Target Data

**Author:** Yilun Jin, Wichita State University  
**Course:** CS 746, Perspectives on Data Science, Fall 2026

This project asks what an audio deepfake detector can gain from genuine recordings in a new domain when no labeled target attacks are available. I compared model adaptation with threshold calibration, checked the cost on the source domain, and tested per-speaker calibration on a separate group of speakers.

The resulting paper, *When Only Bona Fide Speech Is Available: Controlled Adaptation and Per-Speaker Conformal Calibration for Audio Anti-Spoofing*, was submitted to **IEEE Access** on **October 12, 2026**. Manuscript **Access-2026-52407** is **Under Review** as of November 16, 2026.

## Start here

The main report tables can be reproduced from the included summary files with Python 3.10 or later. This step needs no GPU or third-party Python packages:

```bash
python3 tools/reproduce_report_tables.py
```

The command checks the expected results and prints the cohort table, the Study-1 model comparison, and the Study-2 calibration comparison. It aggregates saved results; it does not retrain the detectors or rerun inference.

To check the released files against their hashes:

```bash
python3 tools/verify_release.py
```

For small checks of the calibration and metric functions:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python tools/check_metrics.py
```

## Main findings

On the ASVspoof 5 unencoded evaluation subset, cross-entropy adaptation increased attack detection from **79.646% to 85.694%** for XLSR-LCNN and from **13.188% to 19.372%** for AASIST. The associated false-alarm changes differed between the two families.

In the separate 60-speaker confirmation study, per-speaker calibration improved AASIST detection by **5.99 percentage points** compared with a pooled threshold using the same enrollment recordings (95% CI **[1.84, 10.11]**, Holm-adjusted **p = 0.0064**). A gain was not confirmed for XLSR-LCNN. The false-alarm intervals include zero, leaving that cost unresolved.

These results are conditional on the corpus and systems studied. The calibration rule is an existing split-conformal order statistic. Follow-up split and multi-detector analyses are exploratory.

## Repository contents

| Location | Contents |
| --- | --- |
| `scripts/` | Study code for data preparation, adaptation, scoring, calibration, analysis, and validation |
| `results/` | Saved aggregate results used by the report and supporting analyses |
| `tools/` | Portable table reproduction, calibration checks, and release verification |
| `docs/` | Study protocol, code guide, and reproduction requirements |
| `provenance/` | Original file hashes, model provenance, and a summary of the archived manuscript audit |
| `environment/` | Versions used for the GPU experiments |

The original scripts retain the paths and integrity checks used in the experiments. A full training or inference rerun also needs the datasets, upstream model code, checkpoints, score archives, and historical artifacts described in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Those resources are not bundled here. The portable table command above works with the files in this repository.

## Data and upstream models

- [ASVspoof datasets and challenge resources](https://www.asvspoof.org/)
- [ASVspoof 5 resource description](https://arxiv.org/abs/2502.08857)
- [AASIST](https://github.com/clovaai/aasist), commit `a04c9863f63d44471dde8a6abcb3b082b07cd1d1`
- [XLS-R + LCNN baseline](https://github.com/xieyuankun/ADD-W2V2-LCNN-19LA0.6), commit `13a9a422889f599f35ce105f88c7a13615d1252a`
- [XLS-R-300M](https://huggingface.co/facebook/wav2vec2-xls-r-300m), revision `1a640f32ac3e39899438a2931f9924c02f080a54`

Obtain data and model resources from their original providers under the applicable terms. This repository does not redistribute their audio, model weights, or upstream implementation files.

## Project reference

Yilun Jin. *When Only Bona Fide Speech Is Available: Controlled Adaptation and Per-Speaker Conformal Calibration for Audio Anti-Spoofing*. Submitted to IEEE Access, Access-2026-52407, October 12, 2026. Under review.
