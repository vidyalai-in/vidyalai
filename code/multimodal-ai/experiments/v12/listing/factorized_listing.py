import torch
import torch.nn.functional as F

def attend(x, layout, heads=6):        # x: (batch, frames, patches, width) of query/key/value inputs
    B, T, S, D = x.shape
    if layout == "full":                # one sequence of T*S tokens
        seq = x.reshape(B, T * S, D)
    elif layout == "space":             # each frame alone: B*T sequences of S tokens
        seq = x.reshape(B * T, S, D)
    else:                               # "time": each patch position alone: B*S sequences of T tokens
        seq = x.transpose(1, 2).reshape(B * S, T, D)
    b, n, _ = seq.shape
    q = k = v = seq.reshape(b, n, heads, D // heads).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, D)
    return out.reshape(B, S, T, D).transpose(1, 2) if layout == "time" else out.reshape(B, T, S, D)

x = torch.randn(2, 8, 144, 192)         # 2 clips, 8 frames, 12 x 12 patches, width 192
for layout in ("full", "space", "time"):
    print(layout, tuple(attend(x, layout).shape))
