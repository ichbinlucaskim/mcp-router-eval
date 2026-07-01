#!/usr/bin/env python3
"""Fetch the ToolLinkOS dataset JSON into data/raw/ (idempotent).

Downloads regular_tools.json, core_tools.json, instances.json from the ToolLinkOS repo, pinned to a
specific commit for reproducibility (ADR 0001). Skips files already present, prints size + SHA-256
per file, and writes data/raw/SOURCE.md recording the exact source.

Uses only the standard library — no third-party deps required to read JSON.

Usage:
    python scripts/fetch_data.py            # fetch missing files
    python scripts/fetch_data.py --force    # re-download all
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Pinned source (github.com/EliasLumer/Graph-RAG-Tool-Fusion-ToolLinkOS, MIT license).
REPO = "EliasLumer/Graph-RAG-Tool-Fusion-ToolLinkOS"
COMMIT = "b630b98656e25c3b83a71ea0406572add38ae46d"  # main @ 2025-02-13
FILES = ("regular_tools.json", "core_tools.json", "instances.json")
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/toollinkos"

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(force: bool) -> list[tuple[str, int, str]]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, int, str]] = []
    for name in FILES:
        dest = RAW_DIR / name
        if dest.exists() and not force:
            print(f"[skip] {name} already present")
        else:
            url = f"{RAW_BASE}/{name}"
            print(f"[get ] {name} <- {url}")
            urllib.request.urlretrieve(url, dest)
        size = dest.stat().st_size
        digest = sha256(dest)
        results.append((name, size, digest))
        print(f"       {size:>10,} bytes  sha256:{digest}")
    return results


def write_source_md(results: list[tuple[str, int, str]]) -> None:
    lines = [
        "# ToolLinkOS data source",
        "",
        "Fetched by `scripts/fetch_data.py`. Not redistributed in this repo (gitignored); "
        "re-fetch with the script. License: MIT (see upstream repo).",
        "",
        f"- Repo: https://github.com/{REPO}",
        f"- Commit: `{COMMIT}`",
        f"- Path: `toollinkos/`",
        f"- Fetched (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "| File | Bytes | SHA-256 |",
        "| --- | ---: | --- |",
    ]
    lines += [f"| {n} | {s:,} | `{d}` |" for n, s, d in results]
    (RAW_DIR / "SOURCE.md").write_text("\n".join(lines) + "\n")
    print(f"[ok  ] wrote {RAW_DIR / 'SOURCE.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()
    results = fetch(args.force)
    write_source_md(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
