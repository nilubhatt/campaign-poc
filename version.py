"""
One place the build says what it is (§3.2, defect 10).

    "The client caches tool schemas at connect time. After the server was rebuilt
    mid-session, several parameters that were live and working were absent from the schemas
    in use, and had to be rediscovered by trial and error against a server that already
    supported them."

The first question support asks is "which build is this", and until now nothing could
answer it. `VERSION` is the release; `BUILD` adds the commit and the date when CI stamped
one in, because two builds of v0.2.7 from different commits are not the same thing to
anybody debugging one.

Deliberately importable with no dependencies and no I/O beyond one optional file read: the
version has to be reportable on a machine where the database is corrupt, the embedder is
down and the model is missing — which is exactly when somebody asks.
"""
from __future__ import annotations

import os
from pathlib import Path

# Bumped by hand at release, and the single source of truth — a second copy disagrees with
# the first the day somebody is in a hurry.
VERSION = "0.3.0"

# Written by CI into build_info.txt next to this file (one line: "<commit> <iso-date>").
# Absent in a source checkout, which is itself the useful answer: "not a released build".
_BUILD_INFO = Path(__file__).resolve().parent / "build_info.txt"


def _build() -> str:
    override = os.getenv("CAMPAIGN_POC_BUILD")
    if override:
        return override.strip()
    try:
        line = _BUILD_INFO.read_text(encoding="utf-8").strip()
    except OSError:
        return "source"
    return line.split()[0][:12] if line else "source"


BUILD = _build()
FULL = f"{VERSION}+{BUILD}" if BUILD != "source" else f"{VERSION} (source checkout)"
