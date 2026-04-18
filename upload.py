"""
KyzenRT Model Upload Tool
Copyright (c) 2025 Kypzen (kypzen.com)
SPDX-License-Identifier: MIT

Splits a GGUF file into 64MB chunks, computes SHA-256 per chunk,
uploads to Cloudflare R2, and generates a manifest JSON entry.
Set env vars: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET
"""

import argparse
import hashlib
import json
import os
import boto3
from pathlib import Path
from tqdm import tqdm

CHUNK_SIZE = 64 * 1024 * 1024  # 64MB

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def split_and_upload(model_path: Path, model_id: str, version: str):
    r2 = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    )
    bucket = os.environ["R2_BUCKET"]
    chunks = []

    with open(model_path, "rb") as f:
        i = 0
        while True:
            data = f.read(CHUNK_SIZE)
            if not data:
                break
            chunk_hash = hashlib.sha256(data).hexdigest()
            chunk_key = f"models/{model_id}/v{version}/chunk_{i:04d}.bin"
            print(f"[KyzenRT] Uploading chunk {i} ({len(data)//1024//1024}MB)...")
            r2.put_object(Bucket=bucket, Key=chunk_key, Body=data)
            chunks.append({"index": i, "key": chunk_key, "sha256": chunk_hash, "size": len(data)})
            i += 1

    manifest_entry = {
        "id": model_id,
        "version": version,
        "format": "gguf",
        "total_size": model_path.stat().st_size,
        "chunks": chunks,
        "sha256_full": sha256_file(model_path),
    }

    manifest_path = Path(f"./manifests/{model_id}-v{version}.json")
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_entry, indent=2))
    print(f"[KyzenRT] Manifest written: {manifest_path}")
    return manifest_entry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KyzenRT model upload tool")
    parser.add_argument("--model", required=True, help="Model file path (.gguf)")
    parser.add_argument("--id", required=True, help="Model ID for the store e.g. phi-3.5-mini-q4")
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args()
    split_and_upload(Path(args.model), args.id, args.version)
