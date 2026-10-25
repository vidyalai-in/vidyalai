# Replace a checkpoint's optimizer state with a fresh AdamW's (empty moments, step counts reset).
import sys, torch
p = sys.argv[1]
ck = torch.load(p, map_location="cpu", weights_only=False)
ck["optimizer"]["state"] = {}
torch.save(ck, p)
