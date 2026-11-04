from huggingface_hub import hf_hub_download, snapshot_download
import os
for m in ["google/vit-base-patch16-224","google/vit-base-patch16-384","microsoft/resnet-50","facebook/dinov2-base","facebook/dinov2-with-registers-base"]:
    p=snapshot_download(m, allow_patterns=["*.json","model.safetensors","README.md"])
    print(m, p, flush=True)
print(hf_hub_download("timm/oxford-iiit-pet","data/test-00000-of-00001.parquet",repo_type="dataset"),flush=True)
print(hf_hub_download("timm/oxford-iiit-pet","README.md",repo_type="dataset"),flush=True)
for s in ["train","test"]:
    print(hf_hub_download("uoft-cs/cifar10",f"plain_text/{s}-00000-of-00001.parquet",repo_type="dataset"),flush=True)
print(hf_hub_download("uoft-cs/cifar10","README.md",repo_type="dataset"),flush=True)
