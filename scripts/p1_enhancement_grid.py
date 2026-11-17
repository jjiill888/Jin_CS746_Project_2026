"""P1 enhancement items 2 and 3 (newdoc/p1_enhancement_spec_2026-10-12.md): enrollment-size x nominal-level grid and
break-even ratios. Post hoc, exploratory, existing scores only; rules in results/p1_enhancement_v1/execution_rules.md,
written before this script was run. Refuses to overwrite.

Grid: alpha in {0.02, 0.05, 0.10} x n in {10, 15, 20, 25, 30, 40} on the exposed Study 2 cohort (60 identities). One fixed
nested draw: positions 1-20 are the stored B5 enrollment permutation; each identity's 30 test bona fide are permuted once,
the first 20 extend the enrollment order to 40 and the last 10 are the test bona fide shared by every cell. Threshold =
k-th smallest with k = floor((m+1)*alpha), reject iff margin < threshold; self m = n, pooled m = 60n; k = 0 is a vacuous
rule and is flagged, not reported as low FA. Family = six CE/KD models, equal weight; TPR attack-equal-weighted; identity
bootstrap with the stored B1 resampling counts, thresholds not recomputed within draws (as in B5).
Break-even: w* = dTPR/dFA (self - pooled) = c* x (bona fide : spoof prevalence ratio); no prior or cost is assumed.
"""
import csv, json, hashlib
from fractions import Fraction
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]; P = R / 'results/selfcal_posthoc_v1'; S = R / 'results/asvspoof5_selfcal_v1'
O = R / 'results/p1_enhancement_v1'; D3 = R / 'results/multidetector_d3_v1/per_system.csv'
ALPHAS, NS, NTEST, SEED = ('0.02', '0.05', '0.10'), (10, 15, 20, 25, 30, 40), 10, 20261012
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name, rows):
    with (O / name).open('x', newline='') as f: w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def kth(m, a): return int((m + 1) * Fraction(a))
def threshold(x, a):
    k = kth(len(x), a); return np.full(x.shape[1:], -np.inf) if k == 0 else np.sort(x, axis=0)[k - 1]
def rates(t, bona, fake):  # t (60, models) -> (models, 60, 1 + attacks): FA then per-attack TPR
    fa = (bona < t[:, None]).mean(axis=1); tp = (fake < t[:, None, None]).mean(axis=2)
    return np.concatenate([fa[:, None], tp], axis=1).transpose(2, 0, 1)
def interval(x, counts):
    lo, hi = np.quantile(counts @ x / len(x), [.025, .975]); return float(x.mean()), float(lo), float(hi)
def breakeven(dfa, dtp, counts):
    """dfa, dtp: per-identity self - pooled differences. Quadrant shares over draws; w* interval only if >=95% agree."""
    quad = lambda a, b: np.where((a > 0) & (b > 0), 'tradeoff_self_higher_FA_and_TPR', np.where((a < 0) & (b < 0), 'tradeoff_self_lower_FA_and_TPR',
                        np.where((a <= 0) & (b >= 0) & ((a != 0) | (b != 0)), 'self_not_worse_on_both', np.where((a >= 0) & (b <= 0) & ((a != 0) | (b != 0)), 'pooled_not_worse_on_both', 'identical'))))
    pa, pb = float(dfa.mean()), float(dtp.mean()); q = str(quad(np.array(pa), np.array(pb))); da, db = counts @ dfa / len(dfa), counts @ dtp / len(dtp); qs = quad(da, db)
    out = dict(dFA=pa, dTPR=pb, quadrant=q, w_star=pb / pa if q.startswith('tradeoff') else None, self_cheaper_when=('w < w_star' if pa > 0 else 'w > w_star') if q.startswith('tradeoff') else None,
               **{'share_' + k: float((qs == k).mean()) for k in ('tradeoff_self_higher_FA_and_TPR', 'tradeoff_self_lower_FA_and_TPR', 'self_not_worse_on_both', 'pooled_not_worse_on_both', 'identical')})
    same = qs == q; out['determined'] = bool(q.startswith('tradeoff') and same.mean() >= .95)
    out['w_star_lo'], out['w_star_hi'] = (map(float, np.quantile(db[same] / da[same], [.025, .975])) if out['determined'] else (None, None))
    return out
O.mkdir(exist_ok=True); assert not (O / 'grid_family.csv').exists()
models = json.loads((S / 'selfcal_preregistration.json').read_text())['models']
fam = {f: [j for j, m in enumerate(models) if m['family'] == f and m['method'] in ('full_ce', 'kd10')] for f in ('xlsr_lcnn', 'aasist')}
assert all(len(v) == 6 for v in fam.values())
d = np.load(P / 'b1_arrays.npz', allow_pickle=True); e, b, f, counts, ids = d['enrollment'], d['test_bona'], d['test_fake'], d['counts'], list(d['ids'])
b5 = np.load(P / 'b5_arrays.npz'); ep = b5['enrollment_permutations']; assert e.shape == (60, 20, 20) and b.shape == (60, 30, 20) and f.shape == (60, 8, 10, 20)
rng = np.random.default_rng(SEED); tp = np.stack([rng.permutation(30) for _ in ids])
I = np.arange(60)[:, None]; order = np.concatenate([e[I, ep], b[I, tp[:, :20]]], axis=1); test = b[I, tp[:, 20:]]; assert order.shape == (60, 40, 20) and test.shape == (60, NTEST, 20)
B5SET = [('0.05', 5), ('0.05', 10), ('0.05', 15), ('0.05', 20), ('0.10', 10), ('0.10', 15), ('0.10', 20)]
grid, be, cells, checked = [], [], {}, 0
for a in ALPHAS:
    for n in NS:
        en = order[:, :n]; t = dict(pooled=np.broadcast_to(threshold(en.reshape(-1, 20), a), (60, 20)), self=np.stack([threshold(x, a) for x in en]))
        rs = {arm: rates(t[arm], test, f) for arm in t}; ks, kp = kth(n, a), kth(60 * n, a)
        if (a, n) in B5SET:  # reproduce the stored B5 rates on the original 30 test bona fide
            ref = b5['rates'][B5SET.index((a, n))]; assert np.array_equal(rates(t['pooled'], b, f), ref[0]) and np.array_equal(rates(t['self'], b, f), ref[1]); checked += 1
        for name, ix in fam.items():
            z = {arm: rs[arm][ix].mean(axis=0) for arm in rs}; fa = {arm: z[arm][:, 0] for arm in z}; tpr = {arm: z[arm][:, 1:].mean(axis=1) for arm in z}
            row = dict(alpha=a, n=n, family=name, k_self=ks, effective_level_self=ks / (n + 1), self_vacuous=ks == 0, k_pooled=kp, n_test_bonafide_per_identity=NTEST)
            for arm in ('self', 'pooled'):
                for metric, x in (('FA', fa[arm]), ('TPR', tpr[arm])): row.update(zip((f'{metric}_{arm}', f'{metric}_{arm}_lo', f'{metric}_{arm}_hi'), interval(x, counts)))
            for metric, x in (('dFA', fa['self'] - fa['pooled']), ('dTPR', tpr['self'] - tpr['pooled'])): row.update(zip((metric, metric + '_lo', metric + '_hi'), interval(x, counts)))
            row['worst_identity_FA_self'], row['worst_identity_FA_pooled'] = float(fa['self'].max()), float(fa['pooled'].max()); grid.append(row)
            if ks > 0: be.append(dict(source='grid', comparison=f'{name} alpha={a} n={n} (10 shared test bona fide)', **breakeven(fa['self'] - fa['pooled'], tpr['self'] - tpr['pooled'], counts)))
assert checked == 6
r1 = d['rates']; arms = list(d['arms']); pi, si = arms.index('pooled'), arms.index('self')  # Study 2 confirmation contrast as recorded
s2 = []
for name, ix in fam.items():
    z = r1[si][ix].mean(axis=0) - r1[pi][ix].mean(axis=0); s2.append(dict(source='study2', comparison=f'{name} alpha=0.05 n=20 (30 test bona fide, confirmation split)', **breakeven(z[:, 0], z[:, 1:].mean(axis=1), counts)))
d3 = []
for r in csv.DictReader(D3.open()):
    if r['main_nine'] != 'True': continue
    a, t = float(r['dFA']), float(r['dTPR']); trade = (a > 0 and t > 0) or (a < 0 and t < 0)
    q = ('tradeoff_self_higher_FA_and_TPR' if a > 0 else 'tradeoff_self_lower_FA_and_TPR') if trade else 'self_not_worse_on_both' if (a <= 0 and t >= 0 and (a or t)) else 'pooled_not_worse_on_both' if (a or t) else 'identical'
    det = trade and not (float(r['dFA_lo']) <= 0 <= float(r['dFA_hi'])) and not (float(r['dTPR_lo']) <= 0 <= float(r['dTPR_hi']))
    d3.append(dict(source='d3', comparison=f"{r['model']} ({r['source']}) alpha=0.05 n=20 (10 test bona fide, 16 eval attacks)", dFA=a, dTPR=t, quadrant=q, w_star=t / a if trade else None,
                   self_cheaper_when=('w < w_star' if a > 0 else 'w > w_star') if trade else None, dFA_lo=float(r['dFA_lo']), dFA_hi=float(r['dFA_hi']), dTPR_lo=float(r['dTPR_lo']), dTPR_hi=float(r['dTPR_hi']), determined=det))
assert len(d3) == 9
write('grid_family.csv', grid); write('breakeven_study2_and_grid.csv', s2 + be); write('breakeven_d3.csv', d3)
write('grid_allocation.csv', [dict(speaker_id=ids[i], test_bona_sorted_index=int(v), position=k + 1, role='enrollment_21_40' if k < 20 else 'shared_test') for i in range(60) for k, v in enumerate(tp[i])])
(O / 'completion.json').write_text(json.dumps(dict(posthoc=True, exploratory=True, seed=SEED, b5_cells_reproduced=checked, inputs={str(p.relative_to(R)): sha(p) for p in (P / 'b1_arrays.npz', P / 'b5_arrays.npz', S / 'selfcal_preregistration.json', D3, O / 'execution_rules.md', Path(__file__))},
    artifacts={p.name: sha(p) for p in sorted(O.glob('*.csv'))}), indent=1) + '\n')
print('P1 ENHANCEMENT GRID COMPLETE', len(grid), len(be), len(d3))
