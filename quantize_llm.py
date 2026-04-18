"""
KyzenRT LLM Quantization Pipeline
Copyright (c) 2025 Kypzen (kypzen.com)
SPDX-License-Identifier: MIT

Downloads a model from HuggingFace, converts to GGUF, quantizes.
Requires llama.cpp to be built and available at LLAMA_CPP_PATH.
"""

import argparse
import os
import subprocess
from pathlib import Path

LLAMA_CPP_PATH = os.environ.get("LLAMA_CPP_PATH", "./llama.cpp")
OUTPUT_DIR = Path("./models")

SUPPORTED_QUANTS = ["Q4_K_M", "Q5_K_M", "Q8_0", "Q4_0"]

def quantize(model_id: str, quant: str, output_dir: Path):
    if quant not in SUPPORTED_QUANTS:
        raise ValueError(f"Unsupported quant: {quant}. Choose from {SUPPORTED_QUANTS}")

    model_name = model_id.split("/")[-1].lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[KyzenRT] Downloading {model_id}...")
    subprocess.run([
        "huggingface-cli", "download", model_id,
        "--local-dir", str(output_dir / f"{model_name}_hf")
    ], check=True)

    print(f"[KyzenRT] Converting to GGUF...")
    gguf_path = output_dir / f"{model_name}.gguf"
    subprocess.run([
        "python3", f"{LLAMA_CPP_PATH}/convert_hf_to_gguf.py",
        str(output_dir / f"{model_name}_hf"),
        "--outfile", str(gguf_path)
    ], check=True)

    print(f"[KyzenRT] Quantizing to {quant}...")
    quant_path = output_dir / f"{model_name}.{quant}.gguf"
    subprocess.run([
        f"{LLAMA_CPP_PATH}/llama-quantize",
        str(gguf_path), str(quant_path), quant
    ], check=True)

    print(f"[KyzenRT] Done: {quant_path}")
    return quant_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KyzenRT LLM quantization pipeline")
    parser.add_argument("--model", required=True, help="HuggingFace model ID e.g. microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--quant", default="Q4_K_M", choices=SUPPORTED_QUANTS)
    args = parser.parse_args()
    quantize(args.model, args.quant, OUTPUT_DIR)
