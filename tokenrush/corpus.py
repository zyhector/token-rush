"""Token-id corpora: the calibration set a quantizer sees, and the chunks the
quality table is measured on.

Stored as compressed npz of int32 ids (under 1 MB together) and kept in git,
because they are part of a checkpoint's recipe and **cannot be rebuilt
bit-for-bit elsewhere**: the code corpus is read from the installed torch's
own sources, so a different torch version gives different text. The builders
(`bench/quality_corpus.py`, `bench/quality_calib.py`) are in git too, for the
record of how the ids were chosen; `docs/quantization.md` carries their
hashes.
"""
import numpy as np
import torch


def save_ids(path: str, tensors: dict, meta: dict = None):
    """tensors: name -> LongTensor of token ids. meta: json-able scalars/lists."""
    out = {k: v.cpu().numpy().astype(np.int32) for k, v in tensors.items()}
    for k, v in (meta or {}).items():
        out["meta_" + k] = np.array(v)
    np.savez_compressed(path, **out)


def load_ids(path: str):
    """-> (name -> LongTensor, meta dict). Accepts the .pt the builders used to write."""
    if path.endswith(".pt"):
        d = torch.load(path)
        return ({k: v for k, v in d.items() if torch.is_tensor(v)},
                {k: v for k, v in d.items() if not torch.is_tensor(v)})
    z = np.load(path, allow_pickle=False)
    tensors, meta = {}, {}
    for k in z.files:
        if k.startswith("meta_"):
            v = z[k]
            meta[k[5:]] = v.tolist() if v.ndim else v.item()
        else:
            tensors[k] = torch.from_numpy(z[k].astype(np.int64))
    return tensors, meta


def sha256(t: torch.Tensor) -> str:
    import hashlib
    return hashlib.sha256(t.cpu().numpy().astype(np.int32).tobytes()).hexdigest()[:16]
