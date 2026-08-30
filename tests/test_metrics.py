"""Tests for the pass-rate criterion. Run: pytest -q"""
from src.eval.metrics import is_full_length, translate

GOOD = "ATG" + "GCT" * 466 + "TAA"          # 1,404 nt, in frame, clean stop


def test_accepts_clean_cds():
    assert len(GOOD) == 1404
    assert is_full_length(GOOD)


def test_rejects_internal_stop():
    bad = "ATG" + "GCT" * 200 + "TAA" + "GCT" * 265 + "TAA"
    assert not is_full_length(bad)


def test_rejects_missing_terminal_stop():
    assert not is_full_length("ATG" + "GCT" * 467)


def test_rejects_frameshift():
    assert not is_full_length(GOOD[:-1])


def test_rejects_out_of_range_length():
    assert not is_full_length("ATG" + "GCT" * 100 + "TAA")


def test_translation_stops_at_terminator():
    assert "*" not in translate(GOOD)
