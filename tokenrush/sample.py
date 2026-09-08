"""Sampling that runs inside the decode graph.

Everything is shape-static and branch-free so it records into the CUDA graph:
the candidate set is a fixed top-K slice of the vocabulary (K = 64 is more
than any practical top_k), temperature and top_p/top_k are device tensors
read at replay time, and greedy (temperature 0) is selected arithmetically
rather than by a Python branch. The random number comes from torch's CUDA
generator, whose state torch.cuda.graph captures and advances per replay.
"""
import torch

CANDIDATES = 64


class SamplingParams:
    """Device-resident parameters; change them between replays with .set()."""

    def __init__(self, device):
        self.temperature = torch.zeros(1, device=device, dtype=torch.float32)   # 0 -> greedy
        self.top_p = torch.ones(1, device=device, dtype=torch.float32)
        self.top_k = torch.full((1,), CANDIDATES, device=device, dtype=torch.long)

    def set(self, temperature=0.0, top_p=1.0, top_k=CANDIDATES):
        assert 1 <= top_k <= CANDIDATES
        self.temperature.fill_(temperature)
        self.top_p.fill_(top_p)
        self.top_k.fill_(top_k)


def sample(logits: torch.Tensor, params: SamplingParams, u: torch.Tensor = None) -> torch.Tensor:
    """logits [M, V] -> tokens [M] (long), one independent draw per row.
    u: uniform [M] in [0, 1), drawn if None."""
    M = logits.shape[0]
    vals, idx = logits.float().topk(CANDIDATES, dim=-1)                  # descending
    greedy = logits.argmax(-1)          # not idx[:, 0]: topk breaks ties arbitrarily, argmax by index,
                                        # and the verify step uses argmax; greedy must agree with it
    temp = torch.clamp(params.temperature, min=1e-6)
    p = torch.softmax(vals / temp, dim=-1)
    cum = p.cumsum(-1)
    rank = torch.arange(CANDIDATES, device=logits.device)
    # top-p: keep the smallest prefix whose mass reaches top_p (the first token always stays);
    # top-k: keep the first top_k ranks
    keep = ((cum - p) < params.top_p) & (rank[None, :] < params.top_k)
    p = torch.where(keep, p, torch.zeros_like(p))
    cum = p.cumsum(-1)
    total = cum[:, -1:]
    if u is None:
        u = torch.rand(M, device=logits.device)
    pick = (cum < u.view(M, 1) * total).sum(-1).clamp(max=CANDIDATES - 1)   # inverse CDF, per row
    sampled = idx.gather(1, pick[:, None])[:, 0]
    return torch.where(params.temperature > 0, sampled, greedy)
