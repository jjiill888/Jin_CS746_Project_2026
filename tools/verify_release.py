#!/usr/bin/env python3
"""Verify released files and copied-artifact provenance using SHA-256."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
    for item in manifest["files"]:
        target = (ROOT / item["path"]).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():
            raise ValueError(f"Invalid or missing release file: {item['path']}")
        if digest(target) != item["sha256"]:
            raise ValueError(f"Release hash mismatch: {item['path']}")
    copied = json.loads((ROOT / "provenance/copied_artifacts.json").read_text())
    for item in copied["files"]:
        expected = item.get("release_sha256", item["sha256"])
        if digest(ROOT / item["path"]) != expected:
            raise ValueError(f"Copied artifact mismatch: {item['path']}")
    print(f"Verified {len(manifest['files'])} released files and {len(copied['files'])} copied artifacts.")


if __name__ == "__main__":
    main()
