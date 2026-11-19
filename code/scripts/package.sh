#!/usr/bin/env bash
# Build the course submission from code and report, excluding runtime artifacts.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec "${PYTHON:-python3}" -B - "$@" <<'PY'
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

if len(sys.argv) != 3:
    raise SystemExit('用法：bash code/scripts/package.sh <学号> <姓名>')
student, name = sys.argv[1:]
for value in (student, name):
    if not value or value in ('.', '..') or any(ch in value for ch in '/\\\r\n\0'):
        raise SystemExit('学号和姓名必须是单个有效文件名片段。')
root = Path.cwd().resolve()
wrapper = f'{student}-{name}'
destination = root.parent / f'{wrapper}.zip'
excluded = {'bin', 'build', '.conda', '.mpl-cache', 'archive_pre_bindfix', '__pycache__', '.pending', '.git'}

def files(folder):
    for path in sorted(folder.rglob('*')):
        relative = path.relative_to(root)
        if any(part in excluded for part in relative.parts):
            continue
        if not path.is_file() or path.suffix in {'.o', '.pyc', '.zip', '.tmp'}:
            continue
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise SystemExit(f'拒绝打包仓库外的符号链接：{relative}')
        yield path, relative

selected = list(files(root / 'code'))
for folder in ('figures', 'tables'):
    selected.extend(files(root / 'report' / folder))
for filename in ('report.md', 'report.docx', 'report.pdf'):
    path = root / 'report' / filename
    if path.is_file():
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise SystemExit(f'拒绝打包仓库外的符号链接：report/{filename}')
        selected.append((path, path.relative_to(root)))
if not (root / 'report/report.md').is_file():
    raise SystemExit('缺少 report/report.md。')
zipinfo = shutil.which('zipinfo')
if not zipinfo:
    raise SystemExit('缺少 zipinfo；请使用 conda 安装 zip/unzip 工具。')
handle, temporary = tempfile.mkstemp(prefix=wrapper + '.', suffix='.zip', dir=root.parent)
os.close(handle)
try:
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr(wrapper + '/', '')
        archive.writestr(wrapper + '/code/', '')
        archive.writestr(wrapper + '/report/', '')
        archive.writestr(wrapper + '/report/figures/screenshots/', '')
        for path, relative in selected:
            archive.write(path, wrapper + '/' + relative.as_posix())
        # The two-directory submission also needs a portable environment recipe.
        if not (root / 'code/environment.yml').exists() and (root / 'environment.yml').exists():
            archive.write(root / 'environment.yml', wrapper + '/code/environment.yml')
    listing = subprocess.check_output([zipinfo, '-1', temporary], text=True)
    bad = [entry for entry in listing.splitlines() if any(word in entry.lower() for word in ('/home/', 'wcl', '.conda'))]
    if bad:
        raise SystemExit('压缩包路径检查失败：' + ', '.join(bad))
    with zipfile.ZipFile(temporary) as archive:
        for entry in archive.namelist():
            if not (entry == wrapper + '/' or entry.startswith(wrapper + '/code/') or entry.startswith(wrapper + '/report/')):
                raise SystemExit('出现 code/report 以外的条目：' + entry)
        corrupt = archive.testzip()
        if corrupt:
            raise SystemExit('ZIP 校验失败：' + corrupt)
    os.replace(temporary, destination)
finally:
    Path(temporary).unlink(missing_ok=True)
size = destination.stat().st_size / (1024 * 1024)
print(f'已生成：{destination}\n大小：{size:.2f} MiB；zipinfo 路径检查通过。')
if size > 50:
    print('警告：压缩包超过 50 MiB，请检查需要提交的原始数据与图片大小。')
PY
