"""Download the comparison models and eval datasets into HF_HOME (CPU only)."""
from huggingface_hub import snapshot_download
from datasets import load_dataset

for repo in ["openai-community/gpt2", "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "HuggingFaceTB/SmolLM2-135M"]:
    p = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    print(repo, p, flush=True)

for args in [("Rowan/hellaswag",), ("allenai/ai2_arc", "ARC-Easy"), ("allenai/ai2_arc", "ARC-Challenge"),
             ("baber/piqa",), ("Salesforce/wikitext", "wikitext-103-raw-v1"),
             ("EleutherAI/wikitext_document_level", "wikitext-103-raw-v1")]:
    try:
        ds = load_dataset(*args)
        print(args, {k: len(v) for k, v in ds.items()}, flush=True)
    except Exception as e:
        print(args, "FAILED", repr(e)[:300], flush=True)
