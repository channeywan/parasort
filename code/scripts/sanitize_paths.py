#!/usr/bin/env python3
# sanitize_paths.py：清理历史追溯记录中的机器相关项目路径。
# 对应 DESIGN.md 第 6 节的实验记录与第 9 节的提交要求。
# 保留 CSV 实测、归档证据和报告正文，只改日志及状态中的路径。
"""Make existing command traces, logs, environment and status files portable."""

import argparse
import json
import os
from pathlib import Path
import tempfile

from portable_paths import portable_paths

CODE = Path(__file__).resolve().parents[1]


def sanitized_text(path, original, code_dir, old_roots=(), conda_prefix=None):
    def convert(value):
        return portable_paths(value, code_dir, old_roots, conda_prefix)

    if path.suffix == ".json":
        try:
            value = json.loads(original)
        except ValueError:
            return convert(original)
        result = convert(value)
        return (
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
            if result != value
            else original
        )
    if path.suffix == ".jsonl":
        lines = []
        for line in original.splitlines(keepends=True):
            try:
                value = json.loads(line)
            except ValueError:
                lines.append(convert(line))
                continue
            result = convert(value)
            lines.append(
                json.dumps(result, ensure_ascii=False)
                + ("\n" if line.endswith("\n") else "")
                if result != value
                else line
            )
        return "".join(lines)
    return convert(original)


def sanitize_tree(results, code_dir, old_roots=(), conda_prefix=None, dry_run=False):
    changed = []
    for path in sorted(Path(results).rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if any(
            part in {".mpl-cache", ".pending", "__pycache__"} for part in path.parts
        ):
            continue
        if path.suffix not in {".json", ".jsonl", ".log"} and path.name != "env.txt":
            continue
        original = path.read_text()
        replacement = sanitized_text(path, original, code_dir, old_roots, conda_prefix)
        if replacement == original:
            continue
        changed.append(path)
        if dry_run:
            continue
        descriptor, temporary = tempfile.mkstemp(
            prefix=path.name + ".", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w") as stream:
                stream.write(replacement)
            os.chmod(temporary, path.stat().st_mode)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, default=CODE)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument(
        "--old-root",
        type=Path,
        action="append",
        default=[],
        help="legacy project root; repeatable",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    results = args.results_dir or args.code_dir / "results"
    changed = sanitize_tree(results, args.code_dir, args.old_root, dry_run=args.dry_run)
    print(
        f"{'Would normalize' if args.dry_run else 'Normalized'} {len(changed)} metadata/log files."
    )
    for path in changed:
        print(path.relative_to(results))


if __name__ == "__main__":
    main()
