import torch

conv = torch.nn.Conv2d(3, 768, kernel_size=16, stride=16)   # ViT-B/16's patch embedding
x = torch.randn(1, 3, 224, 224)                              # one RGB image

a = conv(x).flatten(2).transpose(1, 2)                       # (1, 196, 768)

p = x.unfold(2, 16, 16).unfold(3, 16, 16)                    # (1, 3, 14, 14, 16, 16)
p = p.permute(0, 2, 3, 1, 4, 5).reshape(1, 196, 3 * 16 * 16) # 196 flattened patches
b = p @ conv.weight.reshape(768, -1).T + conv.bias           # one matrix multiply

print(a.shape, (a - b).abs().max().item())
