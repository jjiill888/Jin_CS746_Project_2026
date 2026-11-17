"""Shared data/label contracts for the raw-waveform second-detector replication."""
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/aasist_replication_v1'
UPSTREAM = ROOT/'external/aasist'
BASE = UPSTREAM/'models/weights/AASIST.pth'
CONFIG = UPSTREAM/'config/AASIST.conf'
SEEDS = [20260912, 20260913, 20260914]
METHODS = ['full_ce', 'kd10', 'source_only']
POOLS = {
    'source_dev': ('results/asvspoof2019_la_audit/validation.csv', 'validation'),
    'l2_val': ('results/l2_protocol_v2/manifest.csv', 'validation'),
    'eval_subset': ('results/mitigation_b_v1/asvspoof_eval_subset.csv', 'test'),
    'l2_test': ('results/l2_protocol_v2/manifest.csv', 'test'),
    'external': ('results/replication_v1/external_manifest.csv', 'test'),
}
FIELDS = ['utterance_id','speaker_id','split','label','attack_id','native_language','duration_seconds','path','spoof_logit','bonafide_logit','bonafide_score']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def read(path):
    with Path(path).open() as f: return list(csv.DictReader(f))

def save(path, obj):
    with Path(path).open('x') as f: json.dump(obj, f, indent=2, ensure_ascii=False)

def utc(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def labels_for_aasist(values):
    a = np.asarray(values, dtype=np.int64)
    if not np.isin(a, [0, 1]).all(): raise ValueError('Invalid binary label')
    return 1-a

def preprocess(path):
    x, sr = sf.read(ROOT/path, dtype='float32', always_2d=True)
    if not x.size or not np.isfinite(x).all(): raise ValueError(path)
    x = x.mean(axis=1)
    if sr != 16000:
        g = math.gcd(sr, 16000); x = resample_poly(x, 16000//g, sr//g)
    if len(x) < 64600: x = np.tile(x, 64600//len(x)+1)
    return x[:64600].astype(np.float32)

def make_model(weight=BASE):
    sys.path.insert(0, str(UPSTREAM))
    from models.AASIST import Model
    m = Model(json.loads(CONFIG.read_text())['model_config'])
    m.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True), strict=True)
    return m

def setup_gpu():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; GPU access required')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

def training_lists():
    prior = ROOT/'results/replication_v1'
    adapt = read(prior/'adaptation_training.csv'); source = read(prior/'source_only_training.csv')
    assert len(adapt) == len(source) == 10320
    mapping = {r['path']: i for i, r in enumerate(adapt)}
    assert len(mapping) == 10320
    for a, s in zip(adapt, source):
        assert a['position'] == s['position'] and a['label_0bona_1spoof'] == s['label_0bona_1spoof']
        if a['label_0bona_1spoof'] == '1': assert a['path'] == s['path']
        assert adapt[mapping[s['path']]]['label_0bona_1spoof'] == s['label_0bona_1spoof']
    return adapt, source, np.array([mapping[r['path']] for r in source])

class WaveRows(torch.utils.data.Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return torch.from_numpy(preprocess(self.rows[i]['path'])), i
