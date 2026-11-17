"""D3: exploratory per-speaker calibration extension on the 60 already-exposed Study 1 eval speakers.

Design fixed in newdoc/d3_finalization_and_ack_fix_2026-10-06.md §2 before any D3 number was computed:
arms as in Study 2 (self = minimum of the speaker's 20 enrollment scores; pooled = 60th smallest of the
1,200 enrollment scores), 20/10 enrollment/test allocation by SHA256("speakercal_alloc_v1|"+record_id),
reject iff s < t, attack-equal-weighted TPR over 16 eval attacks, identity bootstrap intervals, no
significance testing. Not a confirmation: the cohort is exposed and the strength hypothesis was formed
after seeing Study 2 and the dev-204 checks. Public score files are filtered to these 60 speakers while
reading. Refuses to overwrite its output directory.
"""
import csv, json, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
R = Path(__file__).resolve().parents[1]; A = R / 'results/asvspoof5_adaptation_v1'; S = R / 'data/external_scores/sasb_asvspoof5'; O = R / 'results/multidetector_d3_v1'
B, NPART, SEED, NE, NT, ALPHA = 10000, 100, 20261007, 20, 10, 0.05
XLSR = {'XLSR-SLS', 'Nes2Net', 'W2V2-AASIST'}
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
def read(p):
    with Path(p).open() as f: return list(csv.DictReader(f))
def eer(b, s):
    t = np.unique(np.r_[b, s]); fa = np.searchsorted(np.sort(b), t, 'left') / len(b); miss = 1 - np.searchsorted(np.sort(s), t, 'left') / len(s); i = np.argmin(abs(fa - miss)); return float((fa[i] + miss[i]) / 2)

def arms(en, te, sp):
    """en (U,NE,S), te (U,NT,S), sp (U,16,10,S) -> per-identity FA (U,S) and per-identity-attack TPR (U,16,S) for self and pooled."""
    ts = en.min(1); k = int(ALPHA * en.shape[0] * NE)
    tp = np.sort(en.reshape(-1, en.shape[2]), 0)[k - 1]
    return ((te < ts[:, None]).mean(1), (sp < ts[:, None, None]).mean(2)), ((te < tp).mean(1), (sp < tp).mean(2))
def delta(en, te, sp):
    (fs, ts_), (fp, tp_) = arms(en, te, sp)
    f = lambda fa, tpr: (fa.mean(0), tpr.mean(0).mean(0))  # macro FA over identities; attack-equal-weighted TPR
    (a, b), (c, d) = f(fs, ts_), f(fp, tp_)
    return dict(FA_self=a, TPR_self=b, FA_pooled=c, TPR_pooled=d, dFA=a - c, dTPR=b - d), fs, fp, ts_.mean(1) - tp_.mean(1)

def main():
    O.mkdir(exist_ok=False)
    man = read(A / 'manifest.csv'); zz = np.load(A / 'evaluation_logits.npz'); rows = [man[i] for i in zz['indices']]
    keep = np.array([r['codec'] == '-' for r in rows]); rr = [r for r, k in zip(rows, keep) if k]; assert len(rr) == 11400
    split = read(R / 'results/speakercal_split_v1/split.csv'); ids = sorted({r['speaker_id'] for r in rr})
    assert len(ids) == 60 and not set(ids) & {r['speaker_id'] for r in split}
    primary = read(A / 'analysis/primary.csv')
    names = [('local', p['family'], p['model']) for p in primary]; cols = [(zz['logits'][:, :, 0] - zz['logits'][:, :, 1])[keep]]
    want = {r['record_id']: i for i, r in enumerate(rr)}; hashes = {}
    for path in sorted(S.glob('*.scores.txt')):
        x = np.full(len(rr), np.nan)
        with path.open() as f:
            for line in f:
                k = line[:12]
                if k in want: x[want[k]] = float(line.split()[1])
        assert np.isfinite(x).all(), path
        names.append(('public', 'public', path.name.replace('.scores.txt', ''))); cols.append(x[:, None]); hashes[path.name] = sha(path)
    V = np.concatenate(cols, 1); NS = V.shape[1]
    attacks = sorted({r['attack'] for r in rr if r['label'] == 'spoof'}); assert len(attacks) == 16
    key = lambda r: hashlib.sha256(('speakercal_alloc_v1|' + r['record_id']).encode()).hexdigest()
    bona = np.empty((60, NE + NT, NS)); spoof = np.empty((60, 16, 10, NS)); alloc = []
    for u, q in enumerate(ids):
        bi = sorted((i for i, r in enumerate(rr) if r['speaker_id'] == q and r['label'] == 'bonafide'), key=lambda i: key(rr[i])); assert len(bi) == NE + NT
        bona[u] = V[bi]; alloc += [dict(speaker_id=q, record_id=rr[i]['record_id'], role='enrollment' if j < NE else 'test') for j, i in enumerate(bi)]
        for a, att in enumerate(attacks):
            si = [i for i, r in enumerate(rr) if r['speaker_id'] == q and r['attack'] == att]; assert len(si) == 10
            spoof[u, a] = V[si]
    en, te = bona[:, :NE], bona[:, NE:]
    point, fs, fp, did = delta(en, te, spoof)
    rng = np.random.default_rng(SEED)
    boot = {k: np.empty((B, NS)) for k in ('dFA', 'dTPR')}
    for b in range(B):  # identity bootstrap; pooled threshold recomputed from the resampled identities
        i = rng.integers(0, 60, 60); d = delta(en[i], te[i], spoof[i])[0]
        for k in boot: boot[k][b] = d[k]
    part = np.empty((NPART, NS)); partfa = np.empty((NPART, NS))
    for p in range(NPART):  # stability range only: random 20/10 re-partitions; never a sample size, no p-values
        perm = np.argsort(rng.random((60, NE + NT)), 1); bb = np.take_along_axis(bona, perm[:, :, None], 1)
        d = delta(bb[:, :NE], bb[:, NE:], spoof)[0]; part[p] = d['dTPR']; partfa[p] = d['dFA']
    out = []
    for s, (src, fam, model) in enumerate(names):
        b_all, s_all = bona[:, :, s].ravel(), spoof[:, :, :, s].ravel()
        main_set = (src == 'public' and model != 'AASIST') or (src == 'local' and model == 'original')
        row = dict(source=src, family=fam, model=model, main_nine=main_set, front_end='XLS-R' if (fam == 'xlsr_lcnn' or model in XLSR) else 'non-SSL',
                   score='bonafide logit minus spoof logit' if src == 'local' else 'as published (system-specific)',
                   pooled_eer=eer(b_all, s_all), tied_bonafide_fraction=float(1 - len(np.unique(b_all)) / len(b_all)))
        row.update({k: float(v[s]) for k, v in point.items()})
        for k in boot: row[k + '_lo'], row[k + '_hi'] = map(float, np.quantile(boot[k][:, s], [.025, .975]))
        row.update(identities_gain=int((did[:, s] > 0).sum()), identities_loss=int((did[:, s] < 0).sum()),
                   worst_identity_FA_self=float(fs[:, s].max()), worst_identity_FA_pooled=float(fp[:, s].max()))
        for nm, arr in (('dTPR', part), ('dFA', partfa)):
            q = np.quantile(arr[:, s], [0, .25, .5, .75, 1]); row.update({f'repart_{nm}_{t}': float(v) for t, v in zip(('min', 'q1', 'median', 'q3', 'max'), q)})
        out.append(row)
    with (O / 'per_system.csv').open('x', newline='') as f: w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    with (O / 'allocation.csv').open('x', newline='') as f: w = csv.DictWriter(f, fieldnames=list(alloc[0])); w.writeheader(); w.writerows(alloc)
    nine = [r for r in out if r['main_nine']]; assert len(nine) == 9
    rho = spearmanr([r['pooled_eer'] for r in nine], [r['dTPR'] for r in nine])
    summ = dict(code_sha256=sha(__file__), scope='60 Study 1 eval speakers (E_), unencoded: 20 enrollment + 10 test bona fide and 16 attacks x 10 spoof per speaker',
                status='exploratory extension on an already-exposed cohort; not a confirmation; not comparable with Study 2 numbers (16 eval attacks here, 8 dev attacks there)',
                intersection_with_frozen_split=0, alpha=ALPHA, bootstrap_draws=B, repartitions=NPART, seed=SEED,
                spearman_pooled_eer_vs_dTPR_main_nine=dict(rho=float(rho.statistic), n=9, note='descriptive only; strength is collinear with front-end type and with possible XLS-R pretraining exposure (MLS)'),
                inputs={'evaluation_logits.npz': sha(A / 'evaluation_logits.npz'), 'manifest.csv': sha(A / 'manifest.csv'), **hashes})
    with (O / 'summary.json').open('x') as f: json.dump(summ, f, indent=2)
    for r in out:
        if r['main_nine'] or r['model'] == 'AASIST':
            print(f"{r['model'][:14]:14s} {r['family'][:9]:9s} {r['front_end']:7s} EER {100*r['pooled_eer']:5.1f} | TPR pooled {100*r['TPR_pooled']:5.1f} self {100*r['TPR_self']:5.1f} d {100*r['dTPR']:+6.2f} [{100*r['dTPR_lo']:+6.2f},{100*r['dTPR_hi']:+6.2f}] repart [{100*r['repart_dTPR_min']:+.1f},{100*r['repart_dTPR_median']:+.1f},{100*r['repart_dTPR_max']:+.1f}] | FA pooled {100*r['FA_pooled']:4.1f} self {100*r['FA_self']:4.1f} d {100*r['dFA']:+5.2f} [{100*r['dFA_lo']:+5.2f},{100*r['dFA_hi']:+5.2f}] | gain/loss {r['identities_gain']}/{r['identities_loss']} ties {r['tied_bonafide_fraction']:.3f}")
    print(json.dumps(summ['spearman_pooled_eer_vs_dTPR_main_nine']))
if __name__ == '__main__': main()
