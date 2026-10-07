"""Where the weights come from: a local directory or a Hub repo id. No torch here,
so the MLX backend can use it on a Mac."""
import os
import re

# The engine's checkpoint (int4 g128 GPTQ + MSE, 4.25 bpw, 17 GB; docs/quantization.md) and
# its draft, on the Hub. Downloading is the route to the weights: a rebuild is a statistical
# sibling of the published file, not the same file (scripts/quantize/build.sh).
DEFAULT_REPO = "zyhector/Qwen3.8-27B-TokenRush-int4g128"
DFLASH_REPO = "z-lab/Qwen3.8-27B-DFlash2"
_REPO_ID = re.compile(r"^[\w.-]+/[\w.-]+$")
PACK_META = "tokenrush.json"
COPY_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt", "chat_template.jinja")


def resolve_model(spec: str, download: bool = True) -> str:
    """A local directory, or a Hub repo id -> the local directory holding it.

    A repo id is served from the Hub cache (HF_HOME) when it is already there,
    otherwise downloaded into it after a one-line notice of the size — so a
    second run costs nothing. With download=False a missing repo is an error
    that names the download command instead."""
    if os.path.isdir(spec):
        return spec
    if not _REPO_ID.match(spec):
        raise SystemExit(f"{spec}: no such directory (a local checkpoint) and not a Hub repo id (owner/name)")
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        return snapshot_download(spec, local_files_only=True)
    except LocalEntryNotFoundError:
        pass
    if not download:
        raise SystemExit(f"{spec} is not in the Hub cache and --no-download is set; fetch it with:\n"
                         f"    hf download {spec}")
    try:
        gb = sum(f.size or 0 for f in HfApi().model_info(spec, files_metadata=True).siblings) / 1e9
        size = f"{gb:.1f} GB"
    except Exception:
        size = "size unknown"
    print(f"downloading {spec} ({size}) into the Hub cache at {os.environ.get('HF_HOME', '~/.cache/huggingface')}; "
          f"later runs read it from there", flush=True)
    return snapshot_download(spec)


def is_packed(path: str) -> bool:
    return os.path.exists(os.path.join(path, PACK_META))


def not_packed_message(path: str) -> str:
    return (f"{path} is not a packed checkpoint (no {PACK_META}). The engine's checkpoint is {DEFAULT_REPO}: "
            f"run without --model to use it, or  hf download {DEFAULT_REPO} --local-dir <dir>  and pass that")
