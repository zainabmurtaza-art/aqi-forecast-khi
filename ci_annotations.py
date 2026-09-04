# -*- coding: utf-8 -*-
"""
Tiny helper for surfacing partial-failure information in GitHub Actions.

Both scheduled pipelines isolate faults per city, so a run can legitimately
finish having done almost all of its work while a minority of units failed on
a transient blip. Exiting non-zero for that turns a self-healing hiccup into a
red X, but staying completely silent hides real degradation — so those cases
emit a workflow annotation instead: visible on the run summary page, without
failing the job.

Outside Actions (local runs, notebooks) these degrade to plain prints.
"""

import os
import sys

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def _emit(level: str, message: str) -> None:
    # Annotations must be single-line: newlines terminate the command, so any
    # multi-line detail would be dropped by the runner.
    flat = " ".join(str(message).split())
    if IN_GITHUB_ACTIONS:
        print(f"::{level}::{flat}", flush=True)
    else:
        print(f"[{level.upper()}] {flat}", flush=True)


def warn(message: str) -> None:
    """Marks the run with a warning annotation but leaves it green."""
    _emit("warning", message)


def notice(message: str) -> None:
    _emit("notice", message)


def write_summary(markdown: str) -> None:
    """Appends a Markdown block to the run's job summary panel, if available."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(markdown.rstrip() + "\n\n")
    except OSError as exc:  # never let reporting break the pipeline itself
        print(f"Could not write job summary: {exc}", file=sys.stderr)
