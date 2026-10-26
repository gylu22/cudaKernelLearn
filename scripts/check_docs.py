#!/usr/bin/env python3
"""Validate local Markdown links, code fences, and docs navigation."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LINK_PATTERN = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
FENCE_PATTERN = re.compile(r"^\s*(```|~~~)")


def markdown_files() -> list[Path]:
    files = [ROOT / "README.md", ROOT / "results" / "README.md"]
    files.extend(sorted(DOCS.rglob("*.md")))
    return [path for path in files if path.is_file()]


def link_target(raw: str) -> str:
    value = raw.strip()
    if value.startswith("<") and ">" in value:
        return value[1 : value.index(">")]
    return value.split(maxsplit=1)[0]


def local_markdown_targets(path: Path) -> set[Path]:
    targets = set()
    for match in LINK_PATTERN.finditer(path.read_text(encoding="utf-8")):
        target = link_target(match.group(1))
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        without_fragment = unquote(target.split("#", 1)[0])
        if not without_fragment:
            continue
        targets.add((path.parent / without_fragment).resolve())
    return targets


def check_links(files: list[Path]) -> list[str]:
    errors = []
    for path in files:
        for target in sorted(local_markdown_targets(path)):
            if not target.exists():
                errors.append(
                    f"{path.relative_to(ROOT)}: missing local target "
                    f"{target.relative_to(ROOT) if target.is_relative_to(ROOT) else target}"
                )
    return errors


def check_fences(files: list[Path]) -> list[str]:
    errors = []
    for path in files:
        opened: tuple[str, int] | None = None
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = FENCE_PATTERN.match(line)
            if not match:
                continue
            marker = match.group(1)
            if opened is None:
                opened = (marker, line_number)
            elif marker == opened[0]:
                opened = None
        if opened is not None:
            errors.append(
                f"{path.relative_to(ROOT)}:{opened[1]}: unclosed {opened[0]} fence"
            )
    return errors


def check_navigation() -> list[str]:
    index = DOCS / "index.md"
    linked = local_markdown_targets(index)
    active = {
        path.resolve()
        for path in DOCS.rglob("*.md")
        if "archive" not in path.relative_to(DOCS).parts and path != index
    }
    missing = sorted(active - linked)
    return [
        f"docs/index.md: active document is not in navigation: "
        f"{path.relative_to(ROOT)}"
        for path in missing
    ]


def main() -> int:
    files = markdown_files()
    errors = check_links(files) + check_fences(files) + check_navigation()
    if errors:
        print("Documentation checks failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(
        f"Documentation checks passed: {len(files)} Markdown files, "
        "local links, code fences, and navigation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

