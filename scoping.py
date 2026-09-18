"""
One call, one set of answers (§13.5/D114).

§13.3 needed this twice in one item — once so `prepare_evaluation` stopped embedding the same
proposal text four times, once so a save stopped reading the same cited record per finding —
and wrote it twice: a `ContextVar` defaulting to `None`, a context manager that installs a dict
only if none is installed, and a reader that passes straight through when none is. Both copies
were module globals to begin with, both were therefore shared between overlapping tool calls on
the MCP server's worker threads, and both needed the same `contextvars` fix. That is D114's
argument made concrete inside a single item, and §13.3 named it as debt owed here rather than
writing a third copy.

**Why a call and not a process.** What these memos remove is work repeated INSIDE one call —
the same string embedded four times, the same record read twice by two halves of one report.
Across calls the same memo is a cache, and a cache has to answer for staleness: §13.2 spent a
whole item establishing that a cached answer needs a key covering everything it depends on, and
the first attempt at one of these hid an embedder that had gone down between two calls. Nothing
writes to the thing being memoised during the call that memoises it, which is what makes the
narrow scope safe and the wide one not.

**Re-entrant by design.** A nested scope keeps the outer dict rather than shadowing it, so
`save_evaluation` opening both scopes and then reaching `prepare_evaluation`, which opens one
again, does not throw away what the outer block already paid for.
"""
from __future__ import annotations

import contextlib
import contextvars
from typing import Callable, Optional


def scoped_memo(name: str):
    """A `(scope, remembering)` pair: a context manager, and a reader that uses it.

    `scope()` opens a block in which `remembering(key, work)` calls `work` at most once per
    key. Outside a block, `remembering` calls `work` every time — which is what makes it safe
    to reach from code that does not know whether a scope is open.
    """
    holder: contextvars.ContextVar = contextvars.ContextVar(name, default=None)

    @contextlib.contextmanager
    def scope():
        if holder.get() is not None:
            # Already inside one: keep the outer dict. See the module docstring.
            yield
            return
        token = holder.set({})
        try:
            yield
        finally:
            holder.reset(token)

    def remembering(key, work: Callable):
        memo = holder.get()
        if memo is None:
            return work()
        if key not in memo:
            memo[key] = work()
        return memo[key]

    def is_open() -> bool:
        return holder.get() is not None

    scope.remembering = remembering        # type: ignore[attr-defined]
    scope.is_open = is_open                # type: ignore[attr-defined]
    return scope
