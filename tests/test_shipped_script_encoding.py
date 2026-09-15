"""
Product review defect 11 (P2): "Shipped PowerShell must be ASCII or carry a UTF-8 BOM."

The reviewer did the decoding by hand and the diagnosis is exact:

    "A support script containing an em dash failed to parse on the target machine:
        Unexpected token 'the' in expression or statement.
        The Try statement is missing its Catch or Finally block.
    Windows PowerShell 5.1 decodes BOM-less files as Windows-1252. The em dash (E2 80 94)
    becomes three characters, the last of which is ” — which PowerShell accepts as a closing
    double quote. The string terminated mid-sentence and every brace after it mismatched."

    Fix: "Any .ps1 shipped to customers must be ASCII-only or saved UTF-8 with BOM, and CI
    should assert it. This passes on a developer machine — PowerShell 7 defaults to UTF-8."

That last sentence is why this is a test rather than a habit: the failure is invisible on
every machine the people writing the script use.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
UTF8_BOM = b"\xef\xbb\xbf"

# Files that get shipped to, or run on, a customer machine. .venv and .git are ours.
# Ours, or produced by a build. A venv's own Activate.ps1 failing this check would fail the
# suite for a reason that has nothing to do with what we ship.
SKIP_DIRS = {".git", ".venv", ".venv-build", "__pycache__", "node_modules", "build", "dist",
             ".weights-cache", "_verify", "Output", "site-packages"}


def _shipped(*suffixes):
    for path in sorted(ROOT.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def _offending_lines(raw: bytes):
    """Line number and text for every line carrying a byte outside ASCII."""
    bad = []
    for i, line in enumerate(raw.split(b"\n"), start=1):
        if any(byte > 0x7F for byte in line):
            bad.append((i, line.decode("utf-8", errors="replace").strip()))
    return bad


@pytest.mark.parametrize("script", list(_shipped(".ps1")), ids=lambda p: p.name)
def test_a_shipped_powershell_script_parses_on_windows_powershell_5(script):
    """ASCII-only, or a BOM so 5.1 knows it is UTF-8. Either is fine; neither is not."""
    raw = script.read_bytes()
    if raw.startswith(UTF8_BOM):
        return
    bad = _offending_lines(raw)
    assert not bad, (
        f"{script.relative_to(ROOT)} is BOM-less and contains non-ASCII bytes, so Windows "
        f"PowerShell 5.1 will decode it as Windows-1252: {bad}. Replace the characters with "
        f"ASCII, or save the file UTF-8 with BOM."
    )


@pytest.mark.parametrize("script", list(_shipped(".cmd", ".bat")), ids=lambda p: p.name)
def test_a_shipped_batch_file_is_ascii(script):
    """Same trap, older decoder: cmd.exe reads a .bat in the console's OEM code page, which
    on a machine in Europe or Asia is not the one the file was written in."""
    raw = script.read_bytes()
    assert not _offending_lines(raw), f"{script.relative_to(ROOT)} contains non-ASCII bytes"


@pytest.mark.parametrize("script", list(_shipped(".iss")), ids=lambda p: p.name)
def test_the_inno_setup_script_is_ascii(script):
    """Inno Setup reads a .iss as ANSI unless the file declares otherwise, so the same em
    dash that broke run.ps1 is a hazard here. It happens to sit in a comment today, which is
    harmless — and precisely the kind of thing that gets edited into a live string later."""
    raw = script.read_bytes()
    if raw.startswith(UTF8_BOM):
        return
    assert not _offending_lines(raw), f"{script.relative_to(ROOT)} contains non-ASCII bytes"


def test_the_check_would_have_caught_the_reported_failure(tmp_path):
    """The regression this exists to prevent, reconstructed byte for byte: an em dash inside
    a double-quoted string. Under Windows-1252 the E2 80 94 becomes three characters ending
    in a right curly quote, which PowerShell accepts as the closing double quote — the string
    ends mid-sentence and every brace after it mismatches."""
    script = tmp_path / "run.ps1"
    script.write_bytes('Write-Host "Ollama not found — downloading..."\n'.encode("utf-8"))

    raw = script.read_bytes()
    assert not raw.startswith(UTF8_BOM)
    bad = _offending_lines(raw)

    assert bad, "the check must fail on exactly the line that broke on the customer machine"
    assert "”" in raw.decode("cp1252"), (
        "and this is why: the third byte of the em dash decodes to a closing double quote"
    )


def test_there_is_at_least_one_shipped_script_to_check(tmp_path):
    """A parametrised test over an empty list passes silently, which would make this whole
    file a no-op the day somebody moves the scripts."""
    assert list(_shipped(".ps1")), "no .ps1 found — has the layout changed?"


# ── the same class of trap on the other platforms ───────────────────────────
#
# The plan said to audit the Unix scripts rather than assume this is a Windows problem. The
# encoding half genuinely is Windows-only — sh reads bytes and a UTF-8 locale handles the
# rest — but the shape of the defect is not: a shipped script that parses on the machine it
# was written on and not on the machine it runs on. On Unix that is line endings.

@pytest.mark.parametrize("script", list(_shipped(".sh")), ids=lambda p: p.name)
def test_a_shipped_shell_script_has_unix_line_endings(script):
    """A CRLF makes the shebang read as `/usr/bin/env bash\\r`, and the kernel reports "bad
    interpreter: no such file or directory" — naming a file that plainly exists. Invisible in
    every editor and on every machine where git checked the file out with LF."""
    raw = script.read_bytes()
    assert b"\r\n" not in raw, (
        f"{script.relative_to(ROOT)} has CRLF line endings; the shebang becomes "
        f"'/usr/bin/env bash\\r' and the script will not start"
    )


@pytest.mark.parametrize("script", list(_shipped(".sh")), ids=lambda p: p.name)
def test_a_shipped_shell_script_says_what_runs_it(script):
    assert script.read_bytes().startswith(b"#!"), (
        f"{script.relative_to(ROOT)} has no shebang, so what interprets it depends on who "
        f"invoked it"
    )


@pytest.mark.parametrize("script", list(_shipped(".ps1")), ids=lambda p: p.name)
def test_a_shipped_powershell_script_does_not_rely_on_the_line_ending(script):
    """PowerShell copes with either, but a mixed file is a sign of an editor that also
    rewrote the encoding — which is the thing that broke the customer's machine."""
    raw = script.read_bytes()
    mixed = b"\r\n" in raw and raw.replace(b"\r\n", b"").count(b"\n")
    assert not mixed, f"{script.relative_to(ROOT)} mixes CRLF and LF line endings"


@pytest.mark.parametrize("script", list(_shipped(".sh")), ids=lambda p: p.name)
def test_a_shipped_shell_script_actually_parses(script):
    """`bash -n` on everything we ship. Written after a stray quote left an installer
    unparseable and the whole suite stayed green: every other test here reads the file as
    text, and text that does not parse still contains the right words."""
    import subprocess

    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)

    assert result.returncode == 0, (
        f"{script.relative_to(ROOT)} does not parse:\n{result.stderr}"
    )
