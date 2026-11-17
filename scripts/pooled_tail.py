"""Post-hoc: per-speaker false-alarm dispersion under the pooled vs self thresholds (Study 2 cohort).

Exploratory, on the already-exposed 60 Study-2 identities. Reads only
results/selfcal_posthoc_v1/b1_arrays.npz; arm definitions are those of Study 2
(pooled = 60th smallest of the 1,200 enrollment scores; self = minimum of the
speaker's own 20). No p-values. The self arm is a contrast only: its spread is
expected by construction (Beta(1,20) realized FA plus binomial noise) and is
not a finding. The 20 models come from two families and are not independent.
"""
import csv, json, hashlib, pathlib
import numpy as np
R = pathlib.Path(__file__).resolve().parents[1]
I = R / 'results/selfcal_posthoc_v1'
O = R / 'results/pooled_tail_v1'
O.mkdir(exist_ok=False)  # refuse to overwrite
B, SEED = 10000, 20261007
z = np.load(I / 'b1_arrays.npz', allow_pickle=True)
en, tb = z['enrollment'], z['test_bona']  # (60, 20, M), (60, 30, M); reject iff s < t
U, NE, M = en.shape
N = tb.shape[1]
assert (U, NE, N) == (60, 20, 30)
arms = list(z['arms'])
assert np.array_equal(z['thresholds'][arms.index('self')], en.min(1))
assert np.array_equal(z['thresholds'][arms.index('pooled')][0], np.sort(en.reshape(-1, M), 0)[59])
names = {}
for r in csv.DictReader(open(I / 'b1_models.csv')):
    names[int(r['index'])] = (r['family'], r['model'])
assert sorted(names) == list(range(M))

def fa(e, t, arm):
    """Per-speaker realized FA, shape (speakers, models)."""
    thr = e.min(1)[:, None, :] if arm == 'self' else np.sort(e.reshape(-1, e.shape[2]), 0)[int(0.05 * e.shape[0] * NE) - 1][None, None, :]
    return (t < thr).mean(1)

def stats(f):
    p = f.mean(0)
    sd = f.std(0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        bsd = np.sqrt(p * (1 - p) / N)
        ratio = sd / bsd
    return dict(mean_FA=p, sd_obs=sd, sd_binomial=bsd, sd_ratio=ratio, frac_gt10=(f > 0.10).mean(0), worst_FA=f.max(0))

rng = np.random.default_rng(SEED)
idx = rng.integers(0, U, size=(B, U))  # identity bootstrap; enrollment and test resampled jointly
rows = []
for arm in ('pooled', 'self'):
    f = fa(en, tb, arm)
    s = stats(f)
    q = np.quantile(f, [0, .25, .5, .75, 1], axis=0)
    boot = {k: np.empty((B, M)) for k in ('mean_FA', 'sd_ratio', 'frac_gt10')}
    for b in range(B):
        sb = stats(fa(en[idx[b]], tb[idx[b]], arm))  # pooled threshold is recomputed per draw
        for k in boot: boot[k][b] = sb[k]
    for m in range(M):
        row = dict(arm=arm, family=names[m][0], model=names[m][1], n_speakers=U, n_test_bona_per_speaker=N,
                   mean_FA=s['mean_FA'][m], min_FA=q[0, m], q1_FA=q[1, m], median_FA=q[2, m], q3_FA=q[3, m], max_FA=q[4, m],
                   sd_obs=s['sd_obs'][m], sd_binomial=s['sd_binomial'][m], sd_ratio=s['sd_ratio'][m],
                   frac_speakers_FA_gt10=s['frac_gt10'][m])
        for k in boot:
            lo, hi = np.nanquantile(boot[k][:, m], [.025, .975])
            row[f'{k}_lo'], row[f'{k}_hi'] = lo, hi
        rows.append(row)
with open(O / 'per_model.csv', 'w', newline='') as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# Reference for the self arm under ideal within-speaker exchangeability with continuous scores
x = rng.binomial(N, rng.beta(1, NE, size=(200000, U))) / N
ref = stats(x.T)
summary = {'n_speakers': U, 'n_enroll_per_speaker': NE, 'n_test_bona_per_speaker': N, 'n_models': M, 'bootstrap_draws': B, 'seed': SEED,
           'note': 'post-hoc, exploratory, already-exposed cohort; 20 models from 2 families are not independent; speaker is confounded with recording condition'}
for arm in ('pooled', 'self'):
    a = [r for r in rows if r['arm'] == arm]
    summary[arm] = {k: dict(min=float(np.min([r[k] for r in a])), median=float(np.median([r[k] for r in a])), max=float(np.max([r[k] for r in a])))
                    for k in ('mean_FA', 'sd_ratio', 'max_FA', 'frac_speakers_FA_gt10')}
    for fam in sorted({r['family'] for r in a}):
        summary[arm][fam] = {k: dict(min=float(np.min([r[k] for r in a if r['family'] == fam])), max=float(np.max([r[k] for r in a if r['family'] == fam])))
                             for k in ('mean_FA', 'sd_ratio', 'max_FA')}
summary['self_reference_exchangeable'] = {k: dict(zip(('q05', 'q50', 'q95'), map(float, np.nanquantile(ref[k], [.05, .5, .95]))))
                                          for k in ('mean_FA', 'sd_ratio', 'worst_FA', 'frac_gt10')}
json.dump(summary, open(O / 'summary.json', 'w'), indent=2)
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
json.dump({'script': 'scripts/pooled_tail.py', 'script_sha256': sha(__file__),
           'inputs': {str(p.relative_to(R)): sha(p) for p in (I / 'b1_arrays.npz', I / 'b1_models.csv')},
           'outputs': {p: sha(O / p) for p in ('per_model.csv', 'summary.json')}}, open(O / 'provenance.json', 'w'), indent=2)
print(json.dumps(summary, indent=2))
