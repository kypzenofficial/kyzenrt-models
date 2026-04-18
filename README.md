# KyzenRT Models

Quantization pipeline and model upload tools for KyzenRT — by [Kypzen](https://kypzen.com)

## What this does

1. Downloads base LLM from HuggingFace (FP16 safetensors)
2. Converts to GGUF using llama.cpp
3. Quantizes to Q4_K_M / Q5_K_M / Q8_0
4. Splits into 64MB chunks with SHA-256 per chunk
5. Uploads to Cloudflare R2 with manifest JSON

## Usage

```bash
pip install -r requirements.txt

# Quantize an LLM
python quantize_llm.py --model microsoft/Phi-3.5-mini-instruct --quant Q4_K_M

# Upload to store
python upload.py --model phi-3.5-mini --version 1.0.0
```

## Contributors

| GitHub | Role |
|--------|------|
| [kypzenofficial](https://github.com/kypzenofficial) | Maintainer |
| [prkshtshrm4](https://github.com/prkshtshrm4) | Core contributor |

## Licence
MIT
