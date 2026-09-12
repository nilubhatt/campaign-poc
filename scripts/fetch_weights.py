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
import os
import sys
from dataclasses import dataclass
from pathlib import Path

CHECKPOINT_NAME = "open_clip_model.safetensors"


@dataclass(frozen=True)
class Source:
    """Somewhere a verified checkpoint can come from. Each carries its OWN hash and size:
    the mirror is fp16 and upstream is fp32, so they are genuinely different files and a
    single pinned hash could not describe both."""
    url: str
    sha256: str
    size_bytes: int = 0   # 0 = unknown, skip the pre-download size check
    note: str = ""


# Our own release asset first. The whole premise of this work is that huggingface.co is
# blocked on the customer's network; depending on it to BUILD the thing that fixes that is
# a dependency worth removing too. fp16, half the size, ranking verified identical by
# scripts/convert_fp16.py.
MIRROR = Source(
    url=("https://github.com/nilubhatt/campaign-poc/releases/download/weights-v1/"
         "open_clip_model.safetensors"),
    sha256="cbd90e47b939016c1cb2e3dc63e3d5ee3d663da57dc929c6ebb3067eedd0c452",
    size_bytes=302_588_458,
    note="fp16 mirror (this repo)",
)

# Upstream, kept as a fallback so a deleted or renamed release cannot break the build.
# open_clip resolves ViT-B-32-quickgelu/openai here (verified via get_pretrained_cfg).
# Pinned to an immutable commit, not "main": /resolve/main/ tracks the branch head.
REVISION = "a6f597a30f7b82c51704746581f9a4e41421e878"
UPSTREAM = Source(
    url=("https://huggingface.co/timm/vit_base_patch32_clip_224.openai/resolve/"
         f"{REVISION}/open_clip_model.safetensors"),
    sha256="e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31",
    size_bytes=605_143_284,
    note="fp32 upstream (Hugging Face)",
)

SOURCES = [MIRROR, UPSTREAM]

# Back-compat aliases for anything importing the single-source names.
URL = MIRROR.url
SHA256 = MIRROR.sha256
SIZE_BYTES = MIRROR.size_bytes


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


def _download(url: str, dest: Path, expected_size: int = 0) -> None:
    import http.client
    import urllib.error
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    try:
        # A timeout matters on a CI runner: without one, a stalled CDN connection hangs the
        # job to the six-hour limit instead of failing in a minute.
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
            declared = response.headers.get("Content-Length") or response.headers.get("x-linked-size")
            if expected_size and declared and int(declared) != expected_size:
                raise WeightsError(
                    f"{url} offered {declared} bytes, expected {expected_size}. A filtered "
                    f"network returning a block page looks exactly like this."
                )
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
    except BaseException as exc:
        # Never leave a half-written .part behind to confuse the next run (or a human
        # looking at the directory wondering which file is real).
        partial.unlink(missing_ok=True)
        # Everything a network can do to a 300MB transfer becomes the same kind of error, so
        # the caller can try the next source. Converting only URLError (the first version)
        # meant a connect failure degraded gracefully while a STALLED OR CUT STREAM - by far
        # the likelier failure on a large download - escaped and killed the build instead.
        if isinstance(exc, (urllib.error.URLError, OSError, http.client.HTTPException)):
            raise WeightsError(f"could not fetch {url}: {exc!r}") from exc
        raise
    partial.replace(dest)  # only becomes the real name once the transfer finished


def ensure_weights(target_dir: Path, *, sha256: str = "", url: str = "",
                   require_mirror: bool = False) -> Path:
    """Put a verified checkpoint in target_dir and return its path.

    An existing copy matching ANY known source is left alone — a bundle may already hold
    either the fp16 mirror or an fp32 upstream copy, and both are valid. One that matches
    nothing (truncated, or a block page from a previous attempt) is replaced rather than
    trusted. Sources are tried in order so a blocked or deleted one degrades to the next
    instead of failing the build."""
    target_dir = Path(target_dir)
    target = target_dir / CHECKPOINT_NAME
    sidecar = target.with_suffix(target.suffix + ".sha256")
    # An explicit hash (with or without a URL) means "just this one" — used by tests and by
    # anyone pinning a specific build. require_mirror is for release builds: shipping a
    # KNOWN artifact matters more than degrading, since a silent fallback would change which
    # model ships (300MB larger, different vectors) with nothing going red.
    if sha256:
        sources = [Source(url or MIRROR.url, sha256)]
    elif require_mirror:
        sources = [MIRROR]
    else:
        sources = SOURCES

    if target.is_file():
        actual = sha256_of(target)   # hash once, not once per source
        for source in sources:
            if actual == source.sha256:
                _write_sidecar(target, source.sha256)
                return target
        _discard(target, sidecar)

    failures = []
    for source in sources:
        try:
            _download(source.url, target, source.size_bytes)
            verify(target, source.sha256)
        except WeightsError as exc:
            failures.append(f"{source.note or source.url}: {exc}")
            _discard(target, sidecar)   # never leave an unverified file for the next source
            continue
        _write_sidecar(target, source.sha256)
        print(f"  using: {source.note or source.url}", flush=True)
        return target

    raise WeightsError("could not obtain the CLIP weights from any source:\n  "
                       + "\n  ".join(failures))


def _discard(target: Path, sidecar: Path) -> None:
    """Remove a file we will not vouch for, AND the sidecar describing it — a hash file left
    next to a different payload is how someone inspecting the directory gets misled."""
    target.unlink(missing_ok=True)
    sidecar.unlink(missing_ok=True)


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
    require_mirror = os.getenv("CAMPAIGN_WEIGHTS_REQUIRE_MIRROR") == "1"
    print(f"fetching CLIP weights into {target_dir} ...", flush=True)
    for source in (([MIRROR] if require_mirror else SOURCES)):
        print(f"  candidate: {source.note or source.url}", flush=True)
    try:
        path = ensure_weights(target_dir, require_mirror=require_mirror)
    except WeightsError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    print(f"verified {path} ({path.stat().st_size} bytes, sha256 ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
