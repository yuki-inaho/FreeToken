"""Compressed token map for Engram hashing, with a disk cache.

Engram hashes n-grams over a *compressed* id space in which tokens that normalize alike
(" The", "the", "THE") share one id. This is a verbatim port of the reference
``build_compressed_token_map``: the normalizer chain, the ``\\ufffd`` (partial byte) special
case and the insertion-order id assignment all have to match, because the compressed vocab
size seeds every hash multiplier and the ids themselves feed the hashes — a drift anywhere
rehashes the whole table.

Building it decodes every one of the ~129K vocab ids through the Rust tokenizer (seconds), so
the result is cached under ``~/.cache/freetoken/engram`` keyed by the bytes of the tokenizer
files; the sidecar records the ``tokenizers`` version, since its Unicode tables decide what
normalizes alike. ``load_or_build_token_map`` hard-asserts the expected compressed size when
the caller knows it (99092 for V4.1-Flash).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

FORMAT_VERSION = 1
CACHE_DIR_ENV = "FREETOKEN_ENGRAM_CACHE_DIR"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


def build_compressed_token_map(tokenizer) -> tuple[np.ndarray, int]:
    """``(lookup int32[vocab], compressed_vocab_size)``; ``tokenizer`` is a transformers fast
    tokenizer (``backend_tokenizer``) or a raw ``tokenizers.Tokenizer``."""
    from tokenizers import Regex, normalizers

    # a private-use char, so a token that is exactly one space survives Strip() instead of
    # collapsing to the empty string and merging with unrelated tokens
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # the raw Rust tokenizer, matching what training decodes with (no clean_up_tokenization_spaces)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        vocab_size = len(tokenizer)
    else:
        backend = tokenizer
        vocab_size = backend.get_vocab_size(with_added_tokens=True)
    key_to_new: dict[str, int] = {}
    lookup = np.zeros(vocab_size, dtype=np.int32)
    for token_id in range(vocab_size):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # a partial UTF-8 byte token: nothing to normalize, so key it by its raw form
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def cache_dir() -> str:
    return os.environ.get(CACHE_DIR_ENV) or os.path.join(os.path.expanduser("~"), ".cache", "freetoken", "engram")


def tokenizer_files_digest(model_path: str) -> str:
    """sha256 over the bytes of the tokenizer files present in ``model_path`` (name-prefixed, so a
    renamed or missing file changes the key too)."""
    h = hashlib.sha256()
    found = False
    for name in TOKENIZER_FILES:
        path = os.path.join(model_path, name)
        if not os.path.isfile(path):
            continue
        found = True
        h.update(name.encode() + b"\0")
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        h.update(b"\0")
    if not found:
        raise FileNotFoundError(f"no tokenizer.json under {model_path}")
    return h.hexdigest()


def cache_paths(model_path: str) -> tuple[str, str]:
    stem = os.path.join(cache_dir(), f"token_map-{tokenizer_files_digest(model_path)[:16]}")
    return stem + ".npy", stem + ".json"


def _tokenizers_version() -> str:
    import tokenizers

    return str(tokenizers.__version__)


def _load_tokenizer(model_path: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))


def load_or_build_token_map(
    model_path: str, expected_compressed_vocab: int | None = None, *, tokenizer=None, use_cache: bool = True
) -> tuple[np.ndarray, int]:
    """Cached ``build_compressed_token_map`` for the tokenizer in ``model_path``.

    The cache is trusted only when its sidecar carries this format version and the running
    ``tokenizers`` version; anything else rebuilds. ``expected_compressed_vocab`` (the args'
    ``engram_compressed_vocab_size``) is a hard assert: a different size means a different
    hash space, and serving with it would read garbage rows.
    """
    npy_path, json_path = cache_paths(model_path)
    if use_cache and os.path.isfile(npy_path) and os.path.isfile(json_path):
        try:
            with open(json_path) as f:
                meta = json.load(f)
            lookup = np.load(npy_path)
            ok = (
                meta.get("format_version") == FORMAT_VERSION
                and meta.get("tokenizers_version") == _tokenizers_version()
                and lookup.dtype == np.int32
                and lookup.ndim == 1
                and int(meta.get("vocab_size", -1)) == lookup.shape[0]
            )
        except (OSError, ValueError, json.JSONDecodeError):
            ok = False
        if ok:
            size = int(meta["compressed_vocab_size"])
            _check_expected(size, expected_compressed_vocab, model_path)
            return lookup, size

    if tokenizer is None:
        tokenizer = _load_tokenizer(model_path)
    lookup, size = build_compressed_token_map(tokenizer)
    _check_expected(size, expected_compressed_vocab, model_path)
    if use_cache:
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        # np.save appends ".npy" to a bare path, so write through a handle to keep the temp name
        tmp = npy_path + f".tmp{os.getpid()}"
        with open(tmp, "wb") as f:
            np.save(f, lookup)
        os.replace(tmp, npy_path)
        meta = {
            "format_version": FORMAT_VERSION,
            "compressed_vocab_size": size,
            "vocab_size": int(lookup.shape[0]),
            "tokenizers_version": _tokenizers_version(),
            "model_path": os.path.abspath(model_path),
        }
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=1)
    return lookup, size


def _check_expected(size: int, expected: int | None, model_path: str) -> None:
    if expected is not None and size != int(expected):
        raise AssertionError(
            f"engram compressed vocab is {size}, the checkpoint expects {expected} ({model_path}); "
            f"tokenizers {_tokenizers_version()} normalizes differently from the reference — "
            "refusing to hash with a different id space"
        )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m freetoken.engram.token_map", description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, help="checkpoint directory holding tokenizer.json")
    p.add_argument("--expected", type=int, default=None, help="expected compressed vocab size (hard assert)")
    p.add_argument("--no-cache", action="store_true", help="rebuild and do not touch the cache")
    a = p.parse_args(argv)
    lookup, size = load_or_build_token_map(a.model, a.expected, use_cache=not a.no_cache)
    npy_path, _ = cache_paths(a.model)
    print(f"vocab_size={lookup.shape[0]} compressed_vocab_size={size} tokenizers={_tokenizers_version()}")
    print(f"cache={npy_path if not a.no_cache else '(disabled)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "build_compressed_token_map",
    "cache_dir",
    "cache_paths",
    "load_or_build_token_map",
    "tokenizer_files_digest",
]
