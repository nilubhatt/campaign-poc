"""
Fetch and verify the CLIP checkpoint that ships inside the installer (plan item 1.2).

Run by CI on all three platforms before packaging, so the built payload contains the
weights and a clean, air-gapped machine needs no network at all:

    python scripts/fetch_weights.py dist/campaign-intelligence/models

Verification is the substance here, not a formality. A filtered corporate network answers
with an HTML block page and a 200 - which, unverified, gets written to disk as if it were
the model, passes every build step, and ships an installer that is broken in exactly the
way this whole work item exists to fix. Hash first, ship second.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# open_clip resolves ViT-B-32-quickgelu/openai to this repo (verified via
# open_clip.pretrained.get_pretrained_cfg). Pinned to an immutable commit, not "main":
# /resolve/main/ tracks the branch head, so an upstream re-upload would break a release
# build (the hash check would catch it, but only by failing the build at a bad moment).
CHECKPOINT_NAME = "open_clip_model.safetensors"
REVISION = "a6f597a30f7b82c51704746581f9a4e41421e878"
URL = ("https://huggingface.co/timm/vit_base_patch32_clip_224.openai/resolve/"
       f"{REVISION}/open_clip_model.safetensors")
SHA256 = "e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31"
SIZE_BYTES = 605_143_284


class WeightsError(RuntimeError):
    """Anything that would otherwise ship a wrong or missing checkpoint to a customer."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, sha256: str = SHA256) -> None:
    if not path.is_file():
        raise WeightsError(f"weights not found at {path}")
    actual = sha256_of(path)
    if actual != sha256:
        raise WeightsError(
            f"sha256 mismatch for {path}: expected {sha256}, got {actual} "
            f"({path.stat().st_size} bytes). A filtered network returning a block page "
            f"looks exactly like this — do NOT package it."
        )


def _download(url: str, dest: Path) -> None:
    import urllib.error
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    try:
        # A timeout matters on a CI runner: without one, a stalled CDN connection hangs the
        # job to the six-hour limit instead of failing in a minute.
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
            declared = response.headers.get("Content-Length") or response.headers.get("x-linked-size")
            if declared and int(declared) != SIZE_BYTES:
                raise WeightsError(
                    f"{url} offered {declared} bytes, expected {SIZE_BYTES}. A filtered "
                    f"network returning a block page looks exactly like this."
                )
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
    except BaseException as exc:
        # Never leave a half-written .part behind to confuse the next run (or a human
        # looking at the directory wondering which file is real).
        partial.unlink(missing_ok=True)
        if isinstance(exc, urllib.error.URLError):
            raise WeightsError(f"could not fetch {url}: {exc}") from exc
        raise
    partial.replace(dest)  # only becomes the real name once the transfer finished


def ensure_weights(target_dir: Path, *, sha256: str = SHA256, url: str = URL) -> Path:
    """Put a verified checkpoint in target_dir and return its path.

    A valid copy is left alone (CI caches these between builds; re-pulling 600MB every run
    is pure waste). An invalid one — truncated, or a block page from a previous attempt —
    is replaced rather than trusted."""
    target_dir = Path(target_dir)
    target = target_dir / CHECKPOINT_NAME

    if target.is_file():
        try:
            verify(target, sha256)
            _write_sidecar(target, sha256)
            return target
        except WeightsError:
            target.unlink()

    _download(url, target)
    verify(target, sha256)
    _write_sidecar(target, sha256)
    return target


def _write_sidecar(target: Path, sha256: str) -> None:
    """A .sha256 next to the checkpoint, in the format `shasum -c` / `sha256sum -c` read.

    This is what lets the INSTALLER verify what it just copied. CI verifying before
    packaging proves the payload was right when it was built; it says nothing about a
    half-written copy on the customer's disk, which resolves as "present" and only fails
    later, at load."""
    target.with_suffix(target.suffix + ".sha256").write_text(f"{sha256}  {target.name}\n")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    target_dir = Path(sys.argv[1])
    print(f"fetching CLIP weights into {target_dir} ...", flush=True)
    try:
        path = ensure_weights(target_dir)
    except WeightsError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    print(f"verified {path} ({path.stat().st_size} bytes, sha256 ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
