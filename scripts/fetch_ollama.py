"""
Fetch, verify and stage the Ollama runtime and the nomic-embed-text model that ship inside
the bundle (see bundled_ollama.py for why). Run by CI on each platform after PyInstaller:

    python scripts/fetch_ollama.py dist/campaign-intelligence/ollama --cache .ollama-cache

The customer's network blocks ollama.com and registry.ollama.ai, which is why this happens
at BUILD time, on a runner that can reach both, and never on the customer's machine.

Pinned twice over. The Ollama release is a fixed tag whose archive must match the SHA-256
Ollama published for it. The model is pulled by that very runtime (so the files are laid out
exactly as it reads them) and then checked: every blob must hash to its own name, and the
pinned registry manifest and model weights must be among them. A moved `latest` tag fails
the build instead of quietly shipping different vectors.

CPU only. The GPU libraries (CUDA, ROCm, Vulkan, MLX) are 1.3 GB of Ollama's 1.4 GB Linux
and Windows downloads and would put the bundle past GitHub's 2 GB asset limit; a 137M
parameter embedding model does not need them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

OLLAMA_VERSION = "v0.40.5"


@dataclass(frozen=True)
class Asset:
    name: str
    sha256: str          # from the release's own sha256sum.txt
    binary: str          # the executable's path inside the extracted runtime


ASSETS = {
    "linux": Asset("ollama-linux-amd64.tar.zst",
                   "dc5eabc292f9949546c90a97c820f924b035e05243b0de13c7f366182562169c",
                   "bin/ollama"),
    "win32": Asset("ollama-windows-amd64.zip",
                   "5134ce7e3bc8e94d0494721c799ec293b429450d0c52461b0ee3647dd3bb4bb7",
                   "ollama.exe"),
    "darwin": Asset("ollama-darwin.tgz",
                    "75f9907dafa2fd4c61b5967de2ebb81ccc17eb28da9172edbad0dbc717474dcf",
                    "ollama"),
}

# Ollama is MIT and its archives carry the third-party licences but not its own.
LICENSE_URL = f"https://raw.githubusercontent.com/ollama/ollama/{OLLAMA_VERSION}/LICENSE"
LICENSE_SHA256 = "5934ed2ce0d15154bcdb9c85203210abac0da4314af34081e36df4599f90b226"

MODEL = "nomic-embed-text"
# The registry manifest for nomic-embed-text:latest (Ollama keeps it as a blob) and the
# GGUF weights it names. These are what decide the vectors.
MODEL_MANIFEST_DIGEST = "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f"
MODEL_WEIGHTS_DIGEST = "970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6"
EMBED_DIM = 768

# Path components that are GPU back-ends. Anything under one is left out.
_GPU_DIRS = ("cuda_", "rocm", "vulkan", "mlx_")


def platform_key() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform in ("win32", "darwin"):
        return sys.platform
    raise SystemExit(f"no bundled Ollama for {sys.platform}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: Path, sha256: str) -> Path:
    if dest.is_file() and sha256_of(dest) == sha256:
        print(f"cached: {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    actual = sha256_of(partial)
    if actual != sha256:
        partial.unlink()
        raise SystemExit(f"{url}: SHA-256 {actual} is not the pinned {sha256}")
    partial.replace(dest)
    return dest


def is_gpu_path(member: str) -> bool:
    return any(part.startswith(_GPU_DIRS) for part in Path(member).parts)


def _extract(archive: Path, dest: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if not is_gpu_path(info.filename):
                    zf.extract(info, dest)
        return
    if archive.name.endswith(".tar.zst"):
        # Streamed: decompressed, the Linux archive is several GB of CUDA libraries that are
        # about to be skipped.
        zstd = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=zstd.stdout, mode="r|") as tf:
            for member in tf:
                if not is_gpu_path(member.name):
                    tf.extract(member, dest, filter="data")
        if zstd.wait() != 0:
            raise SystemExit(f"zstd could not decompress {archive}")
        return
    with tarfile.open(archive, "r:gz") as tf:
        members = [m for m in tf.getmembers() if not is_gpu_path(m.name)]
        tf.extractall(dest, members=members, filter="data")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _call(port: int, path: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _serve(binary: Path, models: Path, port: int) -> subprocess.Popen:
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("OLLAMA_")}
    env.update(OLLAMA_HOST=f"127.0.0.1:{port}", OLLAMA_MODELS=str(models), OLLAMA_NOPRUNE="1")
    kwargs = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
              else {"start_new_session": True})
    proc = subprocess.Popen([str(binary), "serve"], env=env, stdin=subprocess.DEVNULL,
                            cwd=str(binary.parent), **kwargs)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/version", timeout=1)
            return proc
        except OSError:
            if proc.poll() is not None:
                raise SystemExit(f"{binary} serve exited with {proc.returncode}")
            time.sleep(0.5)
    _stop(proc)
    raise SystemExit(f"{binary} serve never answered on port {port}")


def _stop(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        import signal
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def verify_models(models: Path) -> None:
    """Every blob hashes to its own name, the pinned manifest and weights are there, and
    every manifest names only blobs that exist."""
    blobs = models / "blobs"
    names = {p.name for p in blobs.iterdir()} if blobs.is_dir() else set()
    for name in sorted(names):
        expected = name.removeprefix("sha256-")
        actual = sha256_of(blobs / name)
        if actual != expected:
            raise SystemExit(f"model blob {name} hashes to {actual}")
    for pinned in (MODEL_MANIFEST_DIGEST, MODEL_WEIGHTS_DIGEST):
        if f"sha256-{pinned}" not in names:
            raise SystemExit(f"{MODEL} is not the pinned build: blob sha256-{pinned} is "
                             f"missing (have {sorted(names)})")
    manifests = list((models / "manifests").rglob("*"))
    manifests = [m for m in manifests if m.is_file()]
    if not manifests:
        raise SystemExit("no model manifest was written")
    for manifest in manifests:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for layer in [data["config"], *data["layers"]]:
            name = layer["digest"].replace(":", "-")
            if name not in names:
                raise SystemExit(f"{manifest} names {name}, which is not in {blobs}")


def stage(dest: Path, cache: Path) -> None:
    # Absolute: the runtime is started with its own directory as the working directory, and
    # a relative path (which is what CI passes) would then name a file that is not there.
    dest, cache = dest.resolve(), cache.resolve()
    asset = ASSETS[platform_key()]
    archive = _download(
        f"https://github.com/ollama/ollama/releases/download/{OLLAMA_VERSION}/{asset.name}",
        cache / OLLAMA_VERSION / asset.name, asset.sha256)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    _extract(archive, dest)
    _download(LICENSE_URL, dest / "OLLAMA_LICENSE", LICENSE_SHA256)
    binary = dest / asset.binary
    if not binary.is_file():
        raise SystemExit(f"the archive has no {asset.binary}")

    models = dest / "models"
    port = _free_port()
    proc = _serve(binary, models, port)
    try:
        print(f"pulling {MODEL} through the staged runtime")
        status = _call(port, "/api/pull", {"model": MODEL, "stream": False}, timeout=900)
        if status.get("status") != "success":
            raise SystemExit(f"pull did not succeed: {status}")
        # The staged runtime, on this platform, CPU only, actually embeds with it.
        vector = _call(port, "/api/embeddings", {"model": MODEL, "prompt": "staging check"},
                       timeout=300)["embedding"]
        if len(vector) != EMBED_DIM:
            raise SystemExit(f"the staged runtime returned {len(vector)} dimensions")
    finally:
        _stop(proc)
    verify_models(models)
    size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"staged Ollama {OLLAMA_VERSION} + {MODEL} in {dest} ({size / 1048576:.0f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dest", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(".ollama-cache"))
    args = parser.parse_args()
    stage(args.dest, args.cache)


if __name__ == "__main__":
    main()
