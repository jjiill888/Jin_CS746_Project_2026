# Study protocol

## Research question

What can a deployer gain from genuine target-domain speech when labeled target attacks are unavailable for training and calibration?

## Data roles

The target data come from ASVspoof 5. Adaptation, threshold calibration, evaluation, and confirmation use disjoint speaker groups.

| Role | Speakers | Genuine utterances | Spoofed utterances |
| --- | ---: | ---: | ---: |
| Adaptation | 86 | 2,580 | 0 |
| Study-1 calibration | 40 | 1,200 | 0 |
| Study-1 unencoded evaluation | 60 | 1,800 | 9,600 |
| Study-2 confirmation | 60 | 3,000 | 4,800 |

The confirmation group has 20 genuine enrollment and 30 genuine test utterances per speaker, plus ten spoofed utterances for each of eight development attacks. Study 1 uses sixteen evaluation attacks. A separate codec analysis uses derivatives of the same Study-1 evaluation speakers.

## Study 1: adaptation

Two detector families are compared: AASIST and XLSR-LCNN. Each family has an original checkpoint, three CE-adapted checkpoints, three KD10-adapted checkpoints, and three source-only controls. The source-only controls reuse verified historical runs with the same initialization, source data, training budget, and seeds. The full pool contains 20 models.

Each adapted run has 10,320 training slots: 2,580 source genuine, 5,160 source spoofed, and 2,580 target genuine. Training uses eight epochs, batch size 32, 2,584 Adam updates, learning rate and weight decay of 0.0001, and seeds 20260912, 20260913, and 20260914. KD10 adds source-model distillation on source examples with weight 10 and temperature 2. The XLS-R encoder is frozen; AASIST or the LCNN back end is updated.

Audio is mono at 16 kHz. The first 64,600 samples are used, repeating shorter recordings to that length. No extra denoising or augmentation is used. XLS-R features are cached in float16 and converted to float32 for LCNN training.

The common score is the genuine logit minus the spoof logit. A recording is rejected when its score is strictly below the threshold. Study-1 thresholds use the 1,200 calibration recordings at an empirical 1% false-alarm target and are frozen before evaluation scoring.

Adaptation plus calibration consumes 3,780 target genuine recordings, while calibration alone uses 1,200. These are different total information budgets. The source-only comparison controls training updates; the Study-2 pooled comparison controls enrollment information.

## Study 2: per-speaker calibration

Sixty development speakers were selected without looking at their scores from a previously unused eligible pool. The selection rule, model hashes, enrollment/test allocation, and analysis rules were frozen locally before confirmation scoring. The preregistration was local, not a registration on a public platform.

At nominal level 0.05, the per-speaker threshold is order statistic k = floor((n + 1) * 0.05). With n = 20, k = 1, so the threshold is the minimum enrollment score. The primary comparator is one pooled threshold made from the same 1,200 enrollment recordings. The transferred Study-1 threshold is secondary.

The primary endpoint is the per-speaker-minus-pooled detection difference, averaged equally over six CE/KD models in each family. Analysis uses 10,000 speaker-paired bootstrap resamples and Holm correction over the two one-sided family-level detection tests. False-alarm differences receive descriptive 95% intervals; no non-inferiority margin was specified.

Under within-speaker exchangeability with the model fixed, the conformal false-rejection bound is marginal over enrollment and a future utterance, at most 1/21. It is not a guarantee for each realized test batch or for all speakers simultaneously.

## Interpretation

The 100-split sensitivity analysis and nine-checkpoint extension reuse exposed speakers. They are exploratory, not fresh replications. Source-domain retention, codec effects, and speaker-level false alarms must be considered alongside detection gains. The data cannot isolate speaker identity from associated recording conditions, and possible XLS-R pretraining exposure to related source audio remains unresolved.
