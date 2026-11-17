# Code guide

| Stage | Main files |
| --- | --- |
| Study-1 data selection and audio checks | `prepare_asvspoof5_adaptation.py`, `audit_asvspoof5_adaptation_audio.py` |
| Model checks and adaptation | `preflight_asvspoof5_adaptation.py`, `train_asvspoof5_adaptation.py` |
| Calibration and inference | `score_asvspoof5_adaptation.py` |
| Study-1 metrics and validation | `asvspoof5_metrics.py`, `analyze_asvspoof5_adaptation.py`, `validate_asvspoof5_adaptation.py` |
| Confirmation cohort selection | `prepare_selfcal_confirmation.py` |
| Confirmation inference | `run_selfcal_confirmation.py`, `execute_selfcal_confirmation.py` |
| Confirmation analysis and checks | `selfcal_common.py`, `analyze_selfcal_confirmation.py`, `validate_selfcal_confirmation.py` |
| Follow-up comparisons | `run_selfcal_posthoc.py`, `validate_selfcal_posthoc.py` |
| Enrollment split sensitivity | `run_selfcal_stability.py`, `validate_stability.py` |
| Matched-FA diagnostic | `run_stability_matched_fa.py`, `validate_stability_matched_fa.py` |
| Speaker-level false-alarm distribution | `pooled_tail.py` |
| Exploratory detector extension | `multidetector_d3.py`, `multidetector_d5_check.py` |
| Cohort description and enrollment grid | `p1_eda.py`, `p1_enhancement_grid.py` |
| Standard metrics | `run_standard_metrics.py` |
| Shared model and distillation helpers | `aasist_replication_common.py`, `preservation_utils.py` |

All files above are under `scripts/`. `score_audio.py` is an early single-file AASIST scoring entry point. Its bona fide logit is distinct from the common logit margin used in the main studies; follow the scoring contracts in [STUDY_PROTOCOL.md](STUDY_PROTOCOL.md).

`run_standard_metrics.py` also depends on the official ASVspoof evaluation package. Exploratory analyses require the original score arrays, speaker manifests, and public-score files referenced in their source. Their aggregate outputs are included selectively under `results/`.
