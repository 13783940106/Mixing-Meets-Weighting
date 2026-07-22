#!/usr/bin/env python3
"""Read-only security and inventory audit for the V15.1 MS-CMR archive."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import stat
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")
EXECUTABLE_SUFFIXES = {
    ".exe", ".dll", ".com", ".bat", ".cmd", ".ps1", ".sh", ".bash",
    ".zsh", ".fish", ".py", ".pl", ".rb", ".jar", ".msi", ".scr",
}
ARCHIVE_SUFFIXES = {
    ".zip", ".rar", ".7z", ".tar", ".tgz", ".tbz", ".tbz2", ".gz",
    ".bz2", ".xz",
}


def classify_entry(info: zipfile.ZipInfo) -> tuple[str, int]:
    mode = (info.external_attr >> 16) & 0xFFFF
    if info.is_dir():
        return "directory", mode
    if stat.S_ISLNK(mode):
        return "symlink", mode
    if stat.S_ISCHR(mode):
        return "char_device", mode
    if stat.S_ISBLK(mode):
        return "block_device", mode
    if stat.S_ISFIFO(mode):
        return "fifo", mode
    if stat.S_ISSOCK(mode):
        return "socket", mode
    return "regular_file", mode


def archive_suffix(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    for suffix in sorted(ARCHIVE_SUFFIXES, key=len, reverse=True):
        if lower.endswith(suffix):
            return suffix
    return Path(lower).suffix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    exact_counts: Counter[str] = Counter()
    folded: dict[str, list[str]] = defaultdict(list)
    problems: list[dict[str, str]] = []
    rows: list[dict[str, object]] = []

    with zipfile.ZipFile(args.zip, "r") as archive:
        infos = archive.infolist()
        total_uncompressed = sum(item.file_size for item in infos)
        total_compressed = sum(item.compress_size for item in infos)

        for index, info in enumerate(infos, start=1):
            name = info.filename
            pure = PurePosixPath(name.replace("\\", "/"))
            entry_type, mode = classify_entry(info)
            exact_counts[name] += 1
            folded[name.casefold()].append(name)

            flags: list[str] = []
            if not name:
                flags.append("EMPTY_NAME")
            if name.startswith(("/", "\\")):
                flags.append("ABSOLUTE_PATH")
            if WINDOWS_DRIVE_RE.match(name):
                flags.append("WINDOWS_DRIVE_PATH")
            if ".." in pure.parts:
                flags.append("PATH_TRAVERSAL")
            if CONTROL_RE.search(name):
                flags.append("CONTROL_CHARACTER")
            if len(name.encode("utf-8", errors="replace")) > 4096:
                flags.append("PATH_TOO_LONG")
            if entry_type in {"symlink", "char_device", "block_device", "fifo", "socket"}:
                flags.append("DANGEROUS_ENTRY_TYPE")
            if info.flag_bits & 0x1:
                flags.append("ENCRYPTED")
            suffix = archive_suffix(name)
            suspicious_exec = suffix in EXECUTABLE_SUFFIXES
            if suspicious_exec:
                flags.append("EXECUTABLE_OR_SCRIPT")
            nested_archive = suffix in ARCHIVE_SUFFIXES and suffix != ".gz" and not name.lower().endswith(".nii.gz")
            if nested_archive:
                flags.append("NESTED_ARCHIVE")
            ratio = (info.file_size / info.compress_size) if info.compress_size else (0.0 if info.file_size == 0 else float("inf"))
            if info.file_size > 10 * 1024**3:
                flags.append("ENTRY_OVER_10_GIB")
            if ratio > 1000 and info.file_size > 100 * 1024**2:
                flags.append("EXTREME_EXPANSION_RATIO")

            for flag in flags:
                problems.append({"entry": name, "problem": flag})

            rows.append({
                "index": index,
                "path": name,
                "entry_type": entry_type,
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
                "crc32_hex": f"{info.CRC:08x}",
                "compression_method": info.compress_type,
                "encrypted": bool(info.flag_bits & 0x1),
                "unix_mode_octal": oct(mode),
                "path_bytes": len(name.encode("utf-8", errors="replace")),
                "nested_archive": nested_archive,
                "suspicious_executable": suspicious_exec,
                "flags": ",".join(flags) if flags else "NONE",
            })

        for name, count in exact_counts.items():
            if count > 1:
                problems.append({"entry": name, "problem": f"DUPLICATE_PATH_X{count}"})
        for names in folded.values():
            unique = sorted(set(names))
            if len(unique) > 1:
                problems.append({"entry": " | ".join(unique), "problem": "CASEFOLD_COLLISION"})

    dangerous = {
        "EMPTY_NAME", "ABSOLUTE_PATH", "WINDOWS_DRIVE_PATH", "PATH_TRAVERSAL",
        "CONTROL_CHARACTER", "PATH_TOO_LONG", "DANGEROUS_ENTRY_TYPE", "ENCRYPTED",
        "ENTRY_OVER_10_GIB", "EXTREME_EXPANSION_RATIO", "CASEFOLD_COLLISION",
    }
    blocking = [item for item in problems if item["problem"].split("_X", 1)[0] in dangerous or item["problem"].startswith("DUPLICATE_PATH")]
    summary = {
        "archive": str(args.zip.resolve()),
        "entry_count": len(rows),
        "file_count": sum(row["entry_type"] == "regular_file" for row in rows),
        "directory_count": sum(row["entry_type"] == "directory" for row in rows),
        "compressed_payload_bytes": total_compressed,
        "uncompressed_bytes": total_uncompressed,
        "expansion_ratio": round(total_uncompressed / total_compressed, 6) if total_compressed else None,
        "encrypted_entry_count": sum(bool(row["encrypted"]) for row in rows),
        "nested_archive_count": sum(bool(row["nested_archive"]) for row in rows),
        "suspicious_executable_count": sum(bool(row["suspicious_executable"]) for row in rows),
        "symlink_count": sum(row["entry_type"] == "symlink" for row in rows),
        "special_file_count": sum(row["entry_type"] in {"char_device", "block_device", "fifo", "socket"} for row in rows),
        "problem_count": len(problems),
        "blocking_problem_count": len(blocking),
        "security_decision": "PASS" if not blocking else "FAIL",
        "problems": problems,
    }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not blocking else 2


if __name__ == "__main__":
    sys.exit(main())
