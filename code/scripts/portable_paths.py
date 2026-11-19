#!/usr/bin/env python3
# portable_paths.py：将实验追溯记录中的项目路径转换为可移植路径。
# 对应 DESIGN.md 第 6 节的实验记录与第 9 节的可复现要求。
# 只生成序列化副本，不修改执行命令或原始配置。
"""Normalize project and conda paths in JSON-compatible metadata copies."""

import os
from pathlib import Path
import re
import sys


def path_replacements(code_dir, old_roots=(), conda_prefix=None):
    """Return longest-first aliases, all expressed relative to code/."""
    code = Path(code_dir).expanduser().absolute()
    projects = {code.parent, code.resolve().parent}
    projects.update(Path(root).expanduser().absolute() for root in old_roots)
    prefix = Path(sys.prefix if conda_prefix is None else conda_prefix).resolve()
    is_conda = conda_prefix is not None or (prefix / "conda-meta").is_dir()
    if is_conda and prefix.name == ".conda":
        # A project-local conda symlink may still target the original checkout.
        projects.add(prefix.parent)
    replacements = {}
    for project in projects:
        for root in {project, project.resolve()}:
            replacements[str(root / "code")] = "."
            replacements[str(root)] = ".."
    replacements[str(code)] = "."
    replacements[str(code.resolve())] = "."
    if is_conda:
        replacements[str(prefix)] = "../.conda"
    replacements.pop(os.path.sep, None)
    return sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)


def portable_paths(value, code_dir, old_roots=(), conda_prefix=None):
    """Recursively copy metadata, replacing only complete project path prefixes.

    The caller retains the original argv and configuration for real execution.
    Unrelated absolute paths, values, and neighboring project names are preserved.
    """
    replacements = path_replacements(code_dir, old_roots, conda_prefix)
    patterns = [
        (
            re.compile(
                r"(?:(?<![\w./\\-])|(?<=-I)|(?<=-L))"
                + re.escape(prefix)
                + r"(?=$|[/\s'\";:,)\]}])"
            ),
            replacement,
        )
        for prefix, replacement in replacements
    ]

    def convert(item):
        if isinstance(item, Path):
            item = str(item)
        if isinstance(item, str):
            for pattern, replacement in patterns:
                item = pattern.sub(lambda match: replacement, item)
            return item
        if isinstance(item, dict):
            return {convert(key): convert(part) for key, part in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(part) for part in item]
        return item

    return convert(value)
