"""Toy models of superposition (owner: Part 6).

Follows Elhage et al. 2022, "Toy Models of Superposition"
(transformer-circuits.pub/2022/toy_model) and its MIT-licensed notebook
(github.com/anthropics/toy-models-of-superposition): many small models are trained at once
as "instances" of one batched model, so a whole sparsity or importance sweep is one run.

Models (x: n features, h: m hidden dimensions, W: [instance, n, m]):
  linear      x' = W^T W x + b
  relu_out    x' = ReLU(W^T W x + b)                 the ReLU output model
  relu_hidden x' = ReLU(W^T ReLU(W x) + b)           hidden ReLU: a privileged basis ("neurons")
Data: each feature is 0 with probability S (sparsity), else uniform on [0, 1].
Loss: importance-weighted squared error, mean over the batch, summed over instances.

Analysis helpers:
  norms(W)            ||W_i||, how strongly each feature is represented
  interference(W)     sum_{j != i} (W_i_hat . W_j)^2, how much other features share feature i's direction
  dimensionality(W)   D_i = ||W_i||^2 / sum_j (W_i_hat . W_j)^2  (paper's per-feature "dimensionality")
  dims_per_feature(W) m / ||W||_F^2, the paper's D* ("dimensions per feature")
  neuron_features(W)  for relu_hidden models: how many features load on each neuron

Tested with torch 2.11 on an RTX 4060 Laptop GPU.
"""
from __future__ import annotations

import math
import time

import torch


class ToyModel(torch.nn.Module):
    def __init__(self, n_inst, n_feat, n_hidden, kind="relu_out", importance=None,
                 density=None, seed=0, device="cuda"):
        super().__init__()
        assert kind in ("linear", "relu_out", "relu_hidden")
        g = torch.Generator(device="cpu").manual_seed(seed)
        W = torch.empty(n_inst, n_feat, n_hidden)
        std = math.sqrt(2.0 / (n_feat + n_hidden))           # Xavier normal, as in the notebook
        W.normal_(0, std, generator=g)
        self.W = torch.nn.Parameter(W.to(device))
        self.b = torch.nn.Parameter(torch.zeros(n_inst, n_feat, device=device))
        self.kind, self.n_inst, self.n_feat, self.n_hidden = kind, n_inst, n_feat, n_hidden
        imp = torch.ones(n_inst, n_feat) if importance is None else torch.as_tensor(importance, dtype=torch.float32)
        den = torch.ones(n_inst, n_feat) if density is None else torch.as_tensor(density, dtype=torch.float32)
        self.importance = imp.expand(n_inst, n_feat).to(device)   # [inst, feat]
        self.density = den.expand(n_inst, n_feat).to(device)      # probability a feature is nonzero (1 - S)
        self.gen = torch.Generator(device=device).manual_seed(seed + 1)

    def forward(self, x):                                   # x [batch, inst, feat]
        h = torch.einsum("bif,ifh->bih", x, self.W)
        if self.kind == "relu_hidden":
            h = torch.relu(h)
        out = torch.einsum("bih,ifh->bif", h, self.W) + self.b
        return out if self.kind == "linear" else torch.relu(out)

    def batch(self, n):
        shape = (n, self.n_inst, self.n_feat)
        dev = self.W.device
        on = torch.rand(shape, device=dev, generator=self.gen) < self.density
        return torch.where(on, torch.rand(shape, device=dev, generator=self.gen), torch.zeros((), device=dev))

    def per_instance_loss(self, x):
        return (self.importance * (x - self(x)) ** 2).mean(0).sum(-1)   # [inst]


def train(model, steps=10_000, batch=1024, lr=1e-3, weight_decay=0.01, log_every=0):
    """AdamW at a constant learning rate, the notebook's defaults (torch's AdamW weight decay is 0.01)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    t0 = time.time()
    for step in range(steps):
        x = model.batch(batch)
        loss = model.per_instance_loss(x).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"step {step} loss/inst {loss.item() / model.n_inst:.5f} {time.time() - t0:.0f}s", flush=True)
    return time.time() - t0


@torch.no_grad()
def eval_loss(model, n=100_000, chunk=10_000):
    tot = 0
    for _ in range(n // chunk):
        tot = tot + model.per_instance_loss(model.batch(chunk))
    return (tot / (n // chunk)).cpu()


@torch.no_grad()
def norms(W):
    return W.norm(dim=-1)                                   # [inst, feat]


@torch.no_grad()
def _overlaps(W):
    unit = W / W.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.einsum("ifh,igh->ifg", unit, W) ** 2      # (W_i_hat . W_j)^2, [inst, feat, feat]


@torch.no_grad()
def interference(W):
    o = _overlaps(W)
    return o.sum(-1) - torch.diagonal(o, dim1=1, dim2=2)


@torch.no_grad()
def dimensionality(W):
    return norms(W) ** 2 / _overlaps(W).sum(-1).clamp_min(1e-12)


@torch.no_grad()
def dims_per_feature(W):
    return W.shape[-1] / (W ** 2).sum((-1, -2))             # m / ||W||_F^2 per instance


def classify(W, learned=0.5, shared=0.1):
    """Per feature: 0 = not learned (||W_i|| < learned), 1 = in superposition (learned and
    interference > shared), 2 = dedicated dimension (learned, nearly orthogonal to the rest)."""
    nrm, itf = norms(W), interference(W)
    c = torch.zeros_like(nrm, dtype=torch.long)
    c[(nrm >= learned) & (itf > shared)] = 1
    c[(nrm >= learned) & (itf <= shared)] = 2
    return c


@torch.no_grad()
def neuron_features(W, thresh=0.3):
    """For a relu_hidden model: number of features with |W_ij| >= thresh on each neuron j.
    0 = unused neuron, 1 = monosemantic, >= 2 = polysemantic. Returns [inst, hidden]."""
    return (W.abs() >= thresh).sum(1)
