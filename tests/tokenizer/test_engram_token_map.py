"""Engram compressed token map: collapsing semantics vs the reference builder, and the disk cache.

References: (1) hand-derived collapses on a tiny byte-level tokenizer built in-test (case,
leading space, accents, whitespace runs, a lone space, partial UTF-8 bytes); (2) a verbatim
copy of the reference ``build_compressed_token_map`` (list-returning) run over the same
tokenizer; (3) cache round-trip with the builder disabled, invalidation on changed tokenizer
bytes, and the hard assert on the expected compressed size.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

tokenizers = pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, decoders, models, pre_tokenizers  # noqa: E402

from freetoken.engram import token_map as tmod  # noqa: E402
from freetoken.engram.token_map import build_compressed_token_map, cache_paths, load_or_build_token_map  # noqa: E402

# byte-level alphabet: " " -> "Ġ", "é" (C3 A9) -> "Ã©", lone 0xC3 -> "Ã", lone 0xC2 -> "Â"
TOKENS = [
    "[UNK]",  # 0
    "The",  # 1
    "Ġthe",  # 2  " the"
    "THE",  # 3
    "ĠĠ",  # 4  "  "
    "Ġ",  # 5  " "
    "hello",  # 6
    "HÃ©llo",  # 7  "Héllo"
    "Ã",  # 8  partial byte 0xC3
    "Â",  # 9  partial byte 0xC2
    "world",  # 10
    "Ġworld",  # 11 " world"
    "ĠĠworld",  # 12 "  world"
    "Ċ",  # 13 "\n" -> whitespace run -> " "
]


def make_tokenizer(tokens=TOKENS) -> Tokenizer:
    vocab = {t: i for i, t in enumerate(tokens)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def ref_build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Verbatim reference (the tokenizer is a raw ``tokenizers.Tokenizer`` here, so
    ``backend_tokenizer``/``len`` are taken from it directly)."""
    from tokenizers import Regex, normalizers

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
    backend = tokenizer
    key_to_new: dict[str, int] = {}
    n = backend.get_vocab_size(with_added_tokens=True)
    lookup = [0] * n
    for token_id in range(n):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
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


def test_tokenizer_decodes_as_intended():
    tok = make_tokenizer()
    assert tok.decode([2], skip_special_tokens=False) == " the"
    assert tok.decode([4], skip_special_tokens=False) == "  "
    assert tok.decode([7], skip_special_tokens=False) == "Héllo"
    assert "\ufffd" in tok.decode([8], skip_special_tokens=False)
    assert tok.decode([13], skip_special_tokens=False) == "\n"


def test_collapses_match_hand_derivation_and_reference():
    tok = make_tokenizer()
    lookup, size = build_compressed_token_map(tok)
    assert lookup.dtype == np.int32 and lookup.shape == (len(TOKENS),)
    ref_lookup, ref_size = ref_build_compressed_token_map(tok)
    assert lookup.tolist() == ref_lookup and size == ref_size

    # hand-derived: case / leading space / accents collapse; whitespace runs and a lone space
    # collapse to " "; partial bytes stay distinct (keyed by their raw form)
    assert lookup[1] == lookup[2] == lookup[3]  # The / " the" / THE
    assert lookup[4] == lookup[5] == lookup[13]  # "  " / " " / "\n"
    assert lookup[6] == lookup[7]  # hello / Héllo
    assert lookup[10] == lookup[11] == lookup[12]  # world / " world" / "  world"
    assert lookup[8] != lookup[9]
    assert lookup[8] not in (lookup[1], lookup[4], lookup[6], lookup[10], lookup[0])
    # keys: [unk], the, " ", hello, Ã, Â, world -> 7 ids, assigned in first-occurrence order
    assert size == 7
    assert lookup.tolist() == [0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 2]


def _write_model(dir_path: str, tok: Tokenizer) -> str:
    os.makedirs(dir_path, exist_ok=True)
    tok.save(os.path.join(dir_path, "tokenizer.json"))
    with open(os.path.join(dir_path, "tokenizer_config.json"), "w") as f:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast"}, f)
    return dir_path


def test_cache_round_trip_and_invalidation(tmp_path, monkeypatch):
    monkeypatch.setenv(tmod.CACHE_DIR_ENV, str(tmp_path / "cache"))
    model = _write_model(str(tmp_path / "model"), make_tokenizer())

    lookup, size = load_or_build_token_map(model, 7)
    npy_path, json_path = cache_paths(model)
    assert os.path.isfile(npy_path) and os.path.isfile(json_path)
    with open(json_path) as f:
        meta = json.load(f)
    assert meta["compressed_vocab_size"] == 7 and meta["tokenizers_version"] == tokenizers.__version__
    assert meta["format_version"] == tmod.FORMAT_VERSION

    # second load comes from the cache: the builder must not run
    def boom(_):
        raise AssertionError("builder ran despite a valid cache")

    monkeypatch.setattr(tmod, "build_compressed_token_map", boom)
    lookup2, size2 = load_or_build_token_map(model, 7)
    assert size2 == 7 and lookup2.tolist() == lookup.tolist()
    # the expected-size assert fires even on a cache hit
    with pytest.raises(AssertionError, match="99092"):
        load_or_build_token_map(model, 99092)
    monkeypatch.undo()
    monkeypatch.setenv(tmod.CACHE_DIR_ENV, str(tmp_path / "cache"))

    # changed tokenizer bytes -> different key -> rebuilt, and the map reflects the new vocab
    model2 = _write_model(str(tmp_path / "model"), make_tokenizer(TOKENS + ["brand-new"]))
    npy2, _ = cache_paths(model2)
    assert npy2 != npy_path
    lookup3, size3 = load_or_build_token_map(model2, None)
    assert size3 == 8 and lookup3.shape == (len(TOKENS) + 1,)
    assert os.path.isfile(npy2)

    # a sidecar from another tokenizers version is not trusted
    with open(json_path) as f:
        meta = json.load(f)
    meta["tokenizers_version"] = "0.0.0"
    with open(json_path, "w") as f:
        json.dump(meta, f)
    calls = []
    real = tmod.build_compressed_token_map
    monkeypatch.setattr(tmod, "build_compressed_token_map", lambda t: (calls.append(1), real(t))[1])
    _write_model(str(tmp_path / "model"), make_tokenizer())
    load_or_build_token_map(model, 7)
    assert calls == [1]
    with open(json_path) as f:
        assert json.load(f)["tokenizers_version"] == tokenizers.__version__


def test_cli_prints_size_and_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(tmod.CACHE_DIR_ENV, str(tmp_path / "cache"))
    model = _write_model(str(tmp_path / "model"), make_tokenizer())
    assert tmod.main(["--model", model, "--expected", "7"]) == 0
    out = capsys.readouterr().out
    assert "compressed_vocab_size=7" in out and cache_paths(model)[0] in out
