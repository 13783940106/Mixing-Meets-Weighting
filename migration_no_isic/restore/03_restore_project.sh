#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 2 ]; then echo "usage: $0 <package_dir> <restore_parent_dir>"; exit 2; fi
PKG=$1; DEST=$2
mkdir -p "$DEST"
for p in "$PKG"/*.tar.zst; do echo "extract $p"; tar --zstd -xpf "$p" -C "$DEST"; done
