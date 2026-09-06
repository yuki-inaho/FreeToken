import hashlib
import json

from benchmark.output_diagnostics import analyze, complete_rows, text_features


def test_features_are_descriptive_counts_with_defined_denominators():
    text = "第１２章\n第１２章\n\n第十三章"
    result = text_features(text)
    assert result["decimal_digits"] == 4
    assert result["decimal_digit_fraction"] == 4 / len(text)
    assert result["chapter_references"] == 3
    assert result["nonempty_lines"] == 3 and result["duplicate_line_occurrences"] == 1
    assert text_features("")["decimal_digit_fraction"] is None
    assert text_features("A" * 20)["repeated_16char_window_fraction"] == 4 / 5


def test_partial_append_is_ignored_without_parsing_incomplete_unicode(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_bytes(b'{"complete": 1}\n' + '{"text":"日'.encode()[:-1])
    rows, incomplete = complete_rows(path)
    assert rows == [{"complete": 1}] and incomplete
    assert path.read_bytes().endswith('日'.encode()[:2])


def test_capture_hash_is_verified_and_original_text_stays_unavailable(tmp_path):
    from tests.test_compare_decode import record

    row = record("baseline-followup", family="japanese")
    text = "第12章の料理について\n" * 3
    sha = hashlib.sha256(text.encode()).hexdigest()
    row["result"].update(output_artifact="outputs/one.json", output_chars=len(text), output_sha256=sha)
    (tmp_path / "outputs").mkdir()
    capture = tmp_path / "outputs/one.json"
    capture.write_text(json.dumps({"record_key": row["key"], "text": text, "output_sha256": sha}))
    (tmp_path / "runs.jsonl").write_text(json.dumps(row) + "\n")
    original = tmp_path / "original"
    original.mkdir()
    old = record("combined", family="japanese")
    (original / "runs.jsonl").write_text(json.dumps(old) + "\n")
    result = analyze(tmp_path, original=original)
    assert result["requests"][0]["text"]["chapter_references"] == 3
    assert result["original_combined_output_availability"]["records_with_capture_reference"] == 0
    capture.write_text(json.dumps({"record_key": row["key"], "text": text + "changed", "output_sha256": sha}))
    result = analyze(tmp_path)
    assert not result["requests"] and "hash mismatch" in result["unavailable"][0]["reason"]
