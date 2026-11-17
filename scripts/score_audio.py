"""Score one recording with the unchanged official AASIST checkpoint."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external" / "aasist"
sys.path.insert(0, str(UPSTREAM))
from models.AASIST import Model
from data_utils import pad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--speaker", default="unknown")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "score.csv")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this process; check GPU access or explicitly use --device cpu.")
    if args.output.exists():
        parser.error("Output already exists; choose another --output to preserve previous results.")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    waveform, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    if not waveform.size or not np.isfinite(waveform).all():
        parser.error("Audio must be nonempty and finite.")
    channels = waveform.shape[1]
    duration = len(waveform) / sr
    waveform = waveform.mean(axis=1)
    if sr != 16000:
        divisor = math.gcd(sr, 16000)
        waveform = resample_poly(waveform, 16000 // divisor, sr // divisor)
    waveform = pad(waveform, 64600).astype(np.float32)
    config = json.loads((UPSTREAM / "config" / "AASIST.conf").read_text())
    checkpoint = UPSTREAM / "models" / "weights" / "AASIST.pth"
    model = Model(config["model_config"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    model.to(args.device)
    with torch.inference_mode():
        _, logits = model(torch.from_numpy(waveform).unsqueeze(0).to(args.device))
    spoof, bona = logits[0].tolist()
    if not math.isfinite(spoof) or not math.isfinite(bona):
        raise RuntimeError("Model returned nonfinite scores")
    row = dict(file=str(args.audio.resolve()), speaker=args.speaker, device=args.device,
               torch_version=torch.__version__,
               original_sample_rate=sr, original_channels=channels,
               duration_seconds=duration, model_sample_rate=16000,
               model_samples=64600, spoof_logit=spoof, bonafide_score=bona,
               checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}\nHigher bonafide_score means more bona fide; no calibrated decision threshold is applied.")


if __name__ == "__main__":
    main()
