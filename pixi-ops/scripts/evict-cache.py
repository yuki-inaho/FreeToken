#!/usr/bin/env python3
"""指定ディレクトリ配下のモデルファイルの clean page cache を追い出す。

cgroup のメモリ上限がタイトな環境で、大きなモデルのロード中に
「読み込み済みページのキャッシュ」がピン留め済み RAM と合算で上限に達し、
リクレームスラッシングでシステムが固まるのを防ぐためのユーティリティ。

使い方: evict-cache.py <dir> [<dir> ...]
"""
import os
import sys

EXTS = (".safetensors", ".gguf", ".bin", ".part")


def main() -> int:
    dirs = sys.argv[1:] or [os.path.expanduser("~/Desktop/ft-models")]
    n = 0
    for d in dirs:
        for root, _subdirs, files in os.walk(d):
            for name in files:
                if not name.endswith(EXTS):
                    continue
                path = os.path.join(root, name)
                try:
                    fd = os.open(path, os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
                    n += 1
                except OSError:
                    pass
    print(f"evicted page cache for {n} files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
