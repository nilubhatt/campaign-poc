"""
The Ollama this build ships with, started and stopped by the product itself.

The customer's network blocks ollama.com and the model registry, so the installers' "download
Ollama, then `ollama pull nomic-embed-text`" could never complete there, and without the
embedder there is no text search, no upload and no evaluation. A build now carries the
runtime (CPU only: the GPU libraries are 1.3 GB of a 1.4 GB download and an embedding model
of this size does not need them) and the model files, under <app>/ollama/:

    bin/ollama + lib/ollama (Linux), ollama.exe + lib/ollama (Windows), flat (macOS)
                                               the runtime, from a pinned Ollama release
    models/manifests, models/blobs             nomic-embed-text, as `ollama pull` lays it out

It runs on 127.0.0.1:<BUNDLED_OLLAMA_PORT> (11435) with OLLAMA_MODELS pointing at that
directory, so it never meets an Ollama the user runs on 11434 and never reads or writes
their models. Nothing registers it as a service: whichever process needs the embedder first
starts it (`ensure_running`), later ones find it answering and use it, and it outlives them
the way a service would, so the model stays loaded between Claude Desktop sessions. The
installers stop it (`stop`) before they replace or remove the directory it runs from.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import config

_lock = threading.Lock()


def runtime_dir() -> Path:
    return config.app_dir() / "ollama"


def binary() -> Path:
    return config.bundled_ollama_binary()


def models_dir() -> Path:
    return runtime_dir() / "models"


def url() -> str:
    return f"http://127.0.0.1:{config.BUNDLED_OLLAMA_PORT}"


def in_use() -> bool:
    """True when the embedder this process talks to is the bundled one."""
    return (config.EMBED_PROVIDER == "ollama" and config.OLLAMA_URL == url()
            and config._bundled_ollama_present())


def _pid_file() -> Path:
    return config.DATA_DIR / "ollama.pid"


def log_file() -> Path:
    return config.DATA_DIR / "ollama.log"


def _answering(timeout: float = 1.0) -> bool:
    import httpx

    try:
        # trust_env=False: a corporate HTTP_PROXY would otherwise carry a loopback request
        # to a proxy that cannot reach it.
        return httpx.get(f"{url()}/api/version", timeout=timeout,
                         trust_env=False).status_code == 200
    except Exception:                      # noqa: BLE001 - any failure means "not answering"
        return False


def _read_pid_file() -> Optional[dict]:
    try:
        info = json.loads(_pid_file().read_text(encoding="utf-8"))
        return info if isinstance(info, dict) and isinstance(info.get("pid"), int) else None
    except (OSError, ValueError):
        return None


def _is_ollama(pid: int) -> bool:
    """The recorded pid is still an Ollama process. A pid outlives nothing: after a reboot
    the number can belong to anything, and `stop` must never kill whatever that is."""
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True, timeout=10).stdout
            return "ollama" in out.lower()
        out = subprocess.run(["ps", "-p", str(pid), "-o", "comm="],
                             capture_output=True, text=True, timeout=10).stdout
        return "ollama" in out.lower()
    except (OSError, subprocess.SubprocessError):
        return False


def stop() -> bool:
    """Stop the bundled Ollama this product started, and its model runner with it.

    Only a process recorded in the pid file and still named ollama is touched. Returns
    whether one was stopped."""
    info = _read_pid_file()
    stopped = False
    if info and _is_ollama(info["pid"]):
        pid = info["pid"]
        try:
            if os.name == "nt":
                # /T: the model runner is a child process holding the runtime's DLLs open,
                # and a leftover one is what makes an upgrade fail to replace them.
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               capture_output=True, timeout=30)
            else:
                # Started with start_new_session, so the pid is its process group and the
                # runner is in it.
                os.killpg(pid, signal.SIGTERM)
                for _ in range(50):
                    if not _is_ollama(pid):
                        break
                    time.sleep(0.1)
                else:
                    os.killpg(pid, signal.SIGKILL)
            stopped = True
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        _pid_file().unlink()
    except OSError:
        pass
    return stopped


def _spawn() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    # The user's own OLLAMA_* settings are for THEIR Ollama; inheriting OLLAMA_HOST or
    # OLLAMA_MODELS would put this one on their port or in their model store.
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("OLLAMA_")}
    env.update(OLLAMA_HOST=f"127.0.0.1:{config.BUNDLED_OLLAMA_PORT}",
               OLLAMA_MODELS=str(models_dir()),
               # The model store is read-only in spirit: nothing here should ever delete a
               # blob it does not recognise.
               OLLAMA_NOPRUNE="1")
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = (subprocess.CREATE_NO_WINDOW
                                   | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        kwargs["start_new_session"] = True
    with open(log_file(), "ab") as log:
        proc = subprocess.Popen([str(binary()), "serve"], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, env=env,
                                cwd=str(runtime_dir()), **kwargs)
    _pid_file().write_text(json.dumps({"pid": proc.pid, "binary": str(binary())}),
                           encoding="utf-8")


def ensure_running(timeout: float = 20.0) -> bool:
    """Make sure the bundled Ollama is answering, starting it if it is not.

    A no-op returning True when this process does not use the bundled embedder. Never
    raises: a failure to start leaves the embedder unreachable, which the callers already
    report with a remedy, and `log_file()` holds Ollama's own account of why."""
    if not in_use():
        return True
    with _lock:
        info = _read_pid_file()
        if _answering():
            # Answering, but from a different install directory: an upgrade or a moved
            # install left the previous copy running, serving from files that may no
            # longer exist. Replace it rather than talk to it.
            if info and info.get("binary") != str(binary()) and _is_ollama(info["pid"]):
                stop()
            else:
                return True
        try:
            _spawn()
        except OSError as exc:
            print(f"[campaign-intelligence] could not start the bundled embedder "
                  f"{binary()}: {exc}", file=sys.stderr, flush=True)
            return False
        deadline = time.monotonic() + max(0.5, timeout)
        while time.monotonic() < deadline:
            if _answering():
                return True
            time.sleep(0.25)
        return _answering()
