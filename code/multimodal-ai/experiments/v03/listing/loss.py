import torch
import torch.nn.functional as F

# Cosine similarities measured with CLIP ViT-B/32: row = image, column = caption
cos = torch.tensor([[0.2956, 0.2475, 0.2364, 0.2291],   # red circle
                    [0.2268, 0.3353, 0.2537, 0.2271],   # blue square
                    [0.2221, 0.2544, 0.3352, 0.2338],   # green triangle
                    [0.2405, 0.2549, 0.2688, 0.3107]])  # yellow star

def clip_loss(cos, scale):
    logits = scale * cos
    labels = torch.arange(len(cos))          # image i belongs with caption i
    return (F.cross_entropy(logits, labels)      # each row: which caption?
            + F.cross_entropy(logits.T, labels)) / 2  # each column: which image?

def siglip_loss(cos, scale, bias):
    sign = 2 * torch.eye(len(cos)) - 1       # +1 on the diagonal, -1 elsewhere
    return -F.logsigmoid(sign * (scale * cos + bias)).sum() / len(cos)

for scale in (1, 14.3, 100):
    print(f"CLIP loss at scale {scale}: {clip_loss(cos, scale):.3f}")
print(f"SigLIP loss at scale 10, bias -10: {siglip_loss(cos, 10, -10):.3f}")
