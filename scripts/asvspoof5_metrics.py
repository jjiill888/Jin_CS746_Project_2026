"""Frozen ASV5 metric contracts; genuine-high margins, strict < decisions.

Deterministic step pAUC is area under the attainable TPR ceiling / alpha,
not interpolated or McClish-standardized pAUC. Codec comparisons match roots.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


def vector(x):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Expected a nonempty finite vector')
    return a


def calibrate(margins, identities, alpha=.01):
    x = vector(margins); ids = np.asarray(identities)
    if ids.shape != x.shape or not 0 <= alpha < 1:
        raise ValueError('Invalid identities or alpha')
    _, inv, counts = np.unique(ids, return_inverse=True, return_counts=True)
    values, vi = np.unique(x, return_inverse=True)
    mass = np.bincount(vi, weights=1 / counts[inv] / len(counts))
    before = np.r_[0., np.cumsum(mass)[:-1]]
    return float(values[np.flatnonzero(before <= alpha + 1e-12)[-1]])


def ranking(bonafide, spoof, alpha=.05):
    b, s = vector(bonafide), vector(spoof)
    if not 0 < alpha <= 1:
        raise ValueError('Invalid pAUC limit')
    values = np.unique(np.r_[b, s])
    # Boundary k includes all scores <= values[k], achievable with nextafter.
    fa = np.r_[0., np.searchsorted(np.sort(b), values, side='right') / len(b)]
    tp = np.r_[0., np.searchsorted(np.sort(s), values, side='right') / len(s)]
    def ceiling(a):
        return float(tp[fa <= a + 1e-12].max())
    knots = np.unique(np.r_[0., fa[(fa > 0) & (fa < alpha)], alpha])
    area = sum((hi - lo) * ceiling(lo) for lo, hi in zip(knots[:-1], knots[1:]))
    ss = np.sort(s)
    auc = np.mean((np.searchsorted(ss, b, 'left') + np.searchsorted(ss, b, 'right')) / (2 * len(s)))
    return dict(auc=float(auc), step_pauc=float(area / alpha), pauc_limit=alpha,
                oracle={str(a): ceiling(a) for a in (0., .01, .05)},
                genuine_support=len(b), spoof_support=len(s))


def identity_rates(rows, margins, threshold, attacks):
    """One condition only; require every identity to have all fixed attacks."""
    x = vector(margins)
    if len(rows) != len(x) or not np.isfinite(threshold):
        raise ValueError('Invalid rows or threshold')
    if len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate records')
    if len({r['codec'] for r in rows}) != 1:
        raise ValueError('Select one codec condition first')
    expected = set(attacks)
    if not expected or len(expected) != len(attacks):
        raise ValueError('Invalid attack list')
    if {r['attack'] for r in rows if r['label'] == 'spoof'} != expected:
        raise ValueError('Attack coverage differs from frozen list')
    ids = sorted({r['speaker_id'] for r in rows})
    rates = np.empty((len(ids), 1 + len(attacks)))
    supports = np.zeros_like(rates, dtype=int)
    for j, ident in enumerate(ids):
        for k, attack in enumerate([None, *attacks]):
            mask = np.array([r['speaker_id'] == ident and
                (r['label'] == 'bonafide' if attack is None else
                 r['label'] == 'spoof' and r['attack'] == attack) for r in rows])
            if not mask.any():
                raise ValueError(f'Missing identity/class/attack: {ident}/{attack}')
            rates[j, k] = np.mean(x[mask] < threshold); supports[j, k] = mask.sum()
    return ids, rates, supports


def paired_bootstrap(rates, reference=None, rounds=10000, seed=20260919):
    """Resample identical identities for FA and all attacks and both models.

    Input rows are same ordered identities, columns FA followed by fixed attacks.
    Return observed point estimate separately from bootstrap percentile limits.
    """
    a = np.asarray(rates, dtype=float)
    if a.ndim != 2 or not len(a) or a.shape[1] < 2 or not np.isfinite(a).all():
        raise ValueError('Invalid rate matrix')
    if (a < 0).any() or (a > 1).any():
        raise ValueError('Rates outside [0,1]')
    if reference is not None:
        ref = np.asarray(reference, dtype=float)
        if ref.shape != a.shape or not np.isfinite(ref).all() or (ref < 0).any() or (ref > 1).any():
            raise ValueError('Invalid paired reference')
        a = a - ref
    if rounds < 1:
        raise ValueError('Invalid rounds')
    endpoints = np.column_stack([a[:, 0], a[:, 1:].mean(axis=1)])
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(a), np.full(len(a), 1 / len(a)), size=rounds)
    draws = counts @ endpoints / len(a)
    return dict(point=endpoints.mean(axis=0).tolist(),
                low=np.quantile(draws, .025, axis=0).tolist(),
                high=np.quantile(draws, .975, axis=0).tolist(),
                rounds=rounds, seed=seed, unit='paired_identity',
                columns=['macro_FA', 'attack_equal_macro_TPR'])


def codec_pairs(rows, margins, threshold, codec):
    """Return root-matched decisions; callers aggregate by identity and attack."""
    x = vector(margins)
    if len(x) != len(rows) or codec == '-':
        raise ValueError('Invalid pairing inputs')
    raw = {}
    for r, m in zip(rows, x):
        if r['codec'] == '-':
            if r['record_id'] in raw:
                raise ValueError('Duplicate raw record')
            raw[r['record_id']] = (r, m)
    out = []
    for r, m in zip(rows, x):
        if r['codec'] != codec:
            continue
        if r['parent_record_id'] not in raw:
            raise ValueError('Missing raw parent')
        parent, pm = raw[r['parent_record_id']]
        for k in ('speaker_id', 'label', 'attack'):
            if r[k] != parent[k]:
                raise ValueError('Parent metadata mismatch')
        out.append(dict(record_id=r['record_id'], parent_record_id=r['parent_record_id'],
                        speaker_id=r['speaker_id'], label=r['label'], attack=r['attack'],
                        codec=codec, raw_spoof=int(pm < threshold), codec_spoof=int(m < threshold),
                        delta=int(m < threshold) - int(pm < threshold)))
    if not out:
        raise ValueError('Empty codec comparison')
    return out


def freeze(path, threshold, rows, margins, model_sha256, alpha=.01):
    """Exclusive creation; provenance includes labels/roles/identity/root and scores."""
    if any(r['role'] != 'calibration' or r['label'] != 'bonafide' or r['codec'] != '-' for r in rows):
        raise ValueError('Calibration must be unencoded held-out genuine')
    x = vector(margins)
    if len(rows) != len(x) or len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('Invalid calibration records')
    expected = calibrate(x, [r['speaker_id'] for r in rows], alpha)
    if threshold != expected:
        raise ValueError('Threshold does not match frozen rule')
    payload = sorted([dict(r, margin=float(m)) for r, m in zip(rows, x)], key=lambda r:r['record_id'])
    record = dict(threshold=threshold, target_fa=alpha, rule='margin < threshold',
        frozen_at=datetime.now(timezone.utc).isoformat(), model_sha256=model_sha256,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        calibration_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        calibration_records=len(rows), calibration_identities=len({r['speaker_id'] for r in rows}))
    with Path(path).open('x') as f:
        json.dump(record, f, indent=2)
    return record
