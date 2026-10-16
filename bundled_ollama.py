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
    VERSION                                    the Ollama version staged, e.g. 0.40.5

It runs on 127.0.0.1:<BUNDLED_OLLAMA_PORT> (11435) with OLLAMA_MODELS pointing at that
directory, so it never meets an Ollama the user runs on 11434 and never reads or writes
their models. Nothing registers it as a service: whichever process needs the embedder first
starts it (`ensure_running`), later ones find it answering and use it, and it outlives them
the way a service would, so the model stays loaded between Claude Desktop sessions. The
installers stop it (`stop`) before they replace or remove the directory it runs from.

WHICH server is ours is decided by the port, not by a pid file. The port is one per machine,
while a pid file lives in a data directory, and every process can have its own
(CAMPAIGN_POC_DATA, another user, an installer run as someone else). Deciding by pid file let a
start race, or a second data directory, hide a running server from `stop`, and the installer
then deleted files under it. So: whatever listens on the port is found, and it is ours when
its executable sits in this product's layout (`.../ollama/ollama`, `.../ollama/bin/ollama`,
`...\\ollama\\ollama.exe` in a directory holding campaign-intelligence, or one that has been
deleted). It is CURRENT when that executable is this install's and it reports the version this
install staged; anything else of ours is replaced. Anything not ours is never touched.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import config

_thread_lock = threading.Lock()
# A start that failed is not retried for this long: every embed of a deck would otherwise
# start a server that cannot bind and wait for it again.
_RETRY_AFTER_FAILURE_SECONDS = 10.0
_last_failure: dict = {}
# What went wrong last, in words, for the health check's remedy.
last_problem: Optional[str] = None


def runtime_dir() -> Path:
    return config.app_dir() / "ollama"


# Helper commands (powershell, taskkill) run without a console window: the server itself is a
# console program, and a window flashing up on every lookup is not something a marketer
# should see.
_HIDDEN = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def _left(deadline: Optional[float], default: float) -> float:
    return default if deadline is None else max(0.5, min(default, deadline - time.monotonic()))


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


def expected_version() -> Optional[str]:
    try:
        return (runtime_dir() / "VERSION").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _version(timeout: float = 1.0) -> Optional[str]:
    """The version whatever answers on the port reports, "" for an answer that is not
    Ollama's, or None when nothing answers."""
    import httpx

    try:
        # trust_env=False: a corporate HTTP_PROXY would otherwise carry a loopback request
        # to a proxy that cannot reach it.
        response = httpx.get(f"{url()}/api/version", timeout=timeout, trust_env=False)
    except Exception:                      # noqa: BLE001 - any failure means "not answering"
        return None
    try:
        return str(response.json().get("version") or "") if response.status_code == 200 else ""
    except ValueError:
        return ""


def _answering() -> bool:
    return _version() is not None


# ── who is listening, and what is it ───────────────────────────────────────────────────

def _listener_pid(deadline: Optional[float] = None) -> Optional[int]:
    """The pid listening on the bundled port, when this account can see it."""
    port = config.BUNDLED_OLLAMA_PORT
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 f"(Get-NetTCPConnection -LocalPort {port} -State Listen "
                 f"-ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess"],
                capture_output=True, text=True, timeout=_left(deadline, 20), **_HIDDEN
            ).stdout.strip()
            return int(out) if out.isdigit() else None
        if sys.platform == "darwin":
            out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                                 capture_output=True, text=True,
                                 timeout=_left(deadline, 10)).stdout.split()
            return int(out[0]) if out and out[0].isdigit() else None
        return _linux_listener_pid(port)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _linux_listener_pid(port: int) -> Optional[int]:
    """From /proc, so no `ss` or `lsof` is needed: the listening socket's inode, then the
    process holding it (only this account's processes are readable, which is the point)."""
    inodes = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) > 9 and fields[3] == "0A" and int(fields[1].split(":")[1], 16) == port:
                inodes.add(fields[9])
    if not inodes:
        return None
    for fd_dir in Path("/proc").glob("[0-9]*/fd"):
        try:
            for fd in fd_dir.iterdir():
                link = os.readlink(fd)
                if link.startswith("socket:[") and link[8:-1] in inodes:
                    return int(fd_dir.parent.name)
        except OSError:
            continue
    return None


def _exe_of(pid: int) -> Optional[str]:
    """The executable a process runs, or None when it is gone or cannot be read."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)   # QUERY_LIMITED_INFORMATION
            if not handle:
                return None
            try:
                size = wintypes.DWORD(32768)
                buf = ctypes.create_unicode_buffer(size.value)
                if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                    return None
                return buf.value
            finally:
                kernel32.CloseHandle(handle)
        if sys.platform.startswith("linux"):
            return re.sub(r" \(deleted\)$", "", os.readlink(f"/proc/{pid}/exe"))
        # macOS: `comm` is the path the process was started by (argv[0]), which for a server
        # this product starts is the absolute path of the runtime, since app_dir() resolves
        # sys.executable. Another account's process is not visible here at all.
        out = subprocess.run(["ps", "-p", str(pid), "-o", "comm="],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out or None
    except (OSError, subprocess.SubprocessError, AttributeError):
        return None


def looks_bundled(exe: str) -> bool:
    """An Ollama in this product's layout: never the user's own, wherever theirs lives
    (/usr/local/bin, Ollama.app, %LOCALAPPDATA%\\Programs\\Ollama)."""
    path = Path(exe)
    if path.name.lower() not in ("ollama", "ollama.exe"):
        return False
    runtime = path.parent.parent if path.parent.name == "bin" else path.parent
    if runtime.name != "ollama":
        return False
    app = runtime.parent
    if not app.exists():
        return True        # an install that has since been deleted or moved
    return any((app / name).exists()
               for name in ("campaign-intelligence", "campaign-intelligence.exe"))


def _same_file(a: str, b: Path) -> bool:
    try:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    except OSError:
        return False


def _listener(deadline: Optional[float] = None) -> tuple[Optional[int], Optional[str]]:
    """The listener's pid and executable, either None when it cannot be seen."""
    pid = _listener_pid(deadline)
    return (pid, _exe_of(pid) if pid is not None else None)


def _ours(deadline: Optional[float] = None) -> Optional[tuple[int, str]]:
    pid, exe = _listener(deadline)
    return (pid, exe) if pid is not None and exe and looks_bundled(exe) else None


def _alive(pid: int) -> bool:
    if os.name == "nt":
        return _exe_of(pid) is not None
    try:
        os.kill(pid, 0)
        if sys.platform.startswith("linux"):
            # Field 3 of /proc/<pid>/stat is the state; Z is a zombie. No `ps` needed.
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
        state = subprocess.run(["ps", "-p", str(pid), "-o", "stat="],
                               capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, IndexError, subprocess.SubprocessError):
        return False
    return bool(state) and not state.startswith("Z")


def _kill(pid: int, deadline: Optional[float] = None) -> bool:
    """Stop a server of ours and its model runner. True once it is gone. Bounded by
    `deadline` when one is given; the installers' stop has the default 15 seconds."""
    deadline = deadline if deadline is not None else time.monotonic() + 15

    def gone_by(until: float) -> bool:
        while time.monotonic() < until:
            if not _alive(pid):
                return True
            time.sleep(0.1)
        return not _alive(pid)

    try:
        if os.name == "nt":
            # /T: the model runner is a child process holding the runtime's DLLs open, and a
            # leftover one is what makes an upgrade fail to replace them.
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                           timeout=_left(deadline, 30), **_HIDDEN)
        else:
            # A server this product started leads its own process group (start_new_session)
            # with the runner in it; anything else gets a plain signal.
            group = os.getpgid(pid) == pid
            (os.killpg if group else os.kill)(pid, signal.SIGTERM)
            if gone_by(min(deadline, time.monotonic() + 5)):
                return True
            (os.killpg if group else os.kill)(pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    # taskkill and SIGKILL return before the process has gone; an installer about to copy
    # over its files needs it gone.
    return gone_by(max(deadline, time.monotonic() + 1))


def stop() -> bool:
    """Stop the bundled Ollama of this product that is running on the port, whichever install
    or data directory started it. Returns whether one was stopped. Never touches a process
    that is not in this product's layout."""
    pids = set()
    found = _ours()
    if found:
        pids.add(found[0])
    # The pid file as a hint: a server that is still starting has not bound the port yet.
    try:
        info = json.loads(_pid_file().read_text(encoding="utf-8"))
        hinted = int(info["pid"])
        exe = _exe_of(hinted)
        if exe and looks_bundled(exe):
            pids.add(hinted)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    results = [_kill(pid) for pid in pids]       # every one tried, not up to the first failure
    stopped = bool(pids) and all(results)
    if stopped or not pids:
        # Kept when a kill failed: it is the only hint to a server that has not bound yet.
        with contextlib.suppress(OSError):
            _pid_file().unlink()
    return stopped


# ── starting it ────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _machine_lock(deadline: float):
    """One starter at a time across this account's processes, so two that both find the port
    free do not both start a server. Per account: the temp directory is on macOS and Windows,
    and on Linux a lock file another account created cannot be opened (fs.protected_regular).
    A start that races another account's loses the bind, reports it, and the next call finds
    the winner. Bounded: past the deadline the caller carries on without it."""
    path = Path(tempfile.gettempdir()) / f"campaign-intelligence-embedder-{config.BUNDLED_OLLAMA_PORT}.lock"
    held = False
    try:
        handle = open(path, "a+b")
    except OSError:
        yield
        return
    try:
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.1)
        yield
    finally:
        if held:
            with contextlib.suppress(OSError):
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def _spawn() -> subprocess.Popen:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    # The user's own OLLAMA_* settings are for THEIR Ollama; inheriting OLLAMA_HOST or
    # OLLAMA_MODELS would put this one on their port or in their model store.
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("OLLAMA_")}
    env.update(OLLAMA_HOST=f"127.0.0.1:{config.BUNDLED_OLLAMA_PORT}",
               OLLAMA_MODELS=str(models_dir()),
               # The model store is read-only in spirit: nothing here should ever delete a
               # blob it does not recognise.
               OLLAMA_NOPRUNE="1")
    flags = {}
    if os.name != "nt":
        flags["start_new_session"] = True
    # One log per run of the server, so it cannot grow without bound.
    with open(log_file(), "wb") as log:
        args = dict(stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env,
                    cwd=str(runtime_dir()), **flags)
        if os.name != "nt":
            return subprocess.Popen([str(binary()), "serve"], **args)
        base = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            # Out of the launching app's job object where that is allowed, so closing Claude
            # Desktop does not take the embedder (and the loaded model) with it.
            return subprocess.Popen([str(binary()), "serve"],
                                    creationflags=base | 0x01000000, **args)  # BREAKAWAY_FROM_JOB
        except OSError:
            return subprocess.Popen([str(binary()), "serve"], creationflags=base, **args)


def _log_tail() -> str:
    try:
        lines = log_file().read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    return lines[-1][:300] if lines else ""


def _warm(deadline: float) -> None:
    """Load the model now. The server answers before the model is in memory, and the first
    embed paying the load is what made a 3-second health probe report a working embedder as
    slow. Best effort: the caller's check is what decides."""
    import httpx

    left = deadline - time.monotonic()
    if left <= 0.5:
        return
    with contextlib.suppress(Exception):
        httpx.post(f"{url()}/api/embeddings",
                   json={"model": config.OLLAMA_EMBED_MODEL, "prompt": "warm up",
                         "keep_alive": config.OLLAMA_KEEP_ALIVE},
                   timeout=left, trust_env=False)


def _fail(message: str) -> bool:
    global last_problem
    last_problem = message
    _last_failure[str(binary())] = time.monotonic()
    print(f"[campaign-intelligence] {message}", file=sys.stderr, flush=True)
    return False


def ensure_running(timeout: float = 20.0, warm: bool = True) -> bool:
    """Make sure the bundled Ollama of THIS install is answering, starting it if it is not,
    and replacing one left running by another install directory or version.

    A no-op returning True when this process does not use the bundled embedder. Never
    raises: a failure is reported through the embedder being unreachable, with
    `last_problem` and `log_file()` saying why. Everything, including stopping a stale
    server, happens inside `timeout`; the model is loaded after the lock is released, since
    loading needs no exclusivity and can take most of the budget."""
    if not in_use():
        return True
    deadline = time.monotonic() + max(0.5, timeout)
    with _thread_lock:
        with _machine_lock(deadline):
            ok = _start_locked(deadline)
    if ok and warm:
        _warm(deadline)
    return ok


def _start_locked(deadline: float) -> bool:
    global last_problem
    if time.monotonic() >= deadline:
        # The budget went on waiting for another starter. Every step below (a lookup, a kill,
        # a start) would run past it and could misread a slow lookup as "nobody owns this".
        return _answering() or _fail("ran out of time waiting for another process to start "
                                     "the bundled embedder; try again")
    version = _version()
    if version is not None:
        pid, exe = _listener(deadline)
        if pid is None:
            if version and version == expected_version():
                # Answering with our version, owner not visible (another account's process):
                # use it rather than fight it.
                last_problem = None
                return True
            if version:
                return _fail(
                    f"an Ollama {version} that this account cannot see is answering on port "
                    f"{config.BUNDLED_OLLAMA_PORT}, probably this product's embedder started by "
                    f"another account. Stop it from that account, or set "
                    f"CAMPAIGN_POC_BUNDLED_OLLAMA_PORT to a free port.")
        if pid is not None and exe and looks_bundled(exe):
            if _same_file(exe, binary()) and version == (expected_version() or version):
                last_problem = None
                return True
            # Another install's, or an older version: replace it. Not while a start failed
            # moments ago, which would leave nothing answering at all.
            if _recently_failed():
                return False
            if not _kill(pid, deadline):
                return _fail(f"could not stop the previous bundled embedder (pid {pid}, {exe}), "
                             f"so this install's cannot start")
        else:
            return _fail(
                f"port {config.BUNDLED_OLLAMA_PORT} is in use by another program, so the "
                f"bundled embedder cannot start. Set CAMPAIGN_POC_BUNDLED_OLLAMA_PORT to a free "
                f"port.")
    if _recently_failed():
        return False
    try:
        proc = _spawn()
    except OSError as exc:
        return _fail(f"could not start the bundled embedder {binary()}: {exc}")
    while time.monotonic() < deadline:
        if _answering():
            if proc.poll() is None:
                # Recorded only once it serves: a starter that lost a race must not leave its
                # dead pid where the live one's belongs.
                with contextlib.suppress(OSError):
                    _pid_file().write_text(
                        json.dumps({"pid": proc.pid, "binary": str(binary())}),
                        encoding="utf-8")
            last_problem = None
            return True
        if proc.poll() is not None:
            return _fail(f"the bundled embedder exited at start ({proc.returncode}): "
                         f"{_log_tail() or 'see ' + str(log_file())}")
        time.sleep(0.25)
    return _fail(f"the bundled embedder did not answer in time; see {log_file()}")


def _recently_failed() -> bool:
    failed_at = _last_failure.get(str(binary()))
    return failed_at is not None and time.monotonic() - failed_at < _RETRY_AFTER_FAILURE_SECONDS
