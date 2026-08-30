"""Tests for the OpenGenome2 partition audit.

The load-bearing assertion is that apicoplasts are excluded from the
rbcL-bearing set, and that this is not the same as a naive plastid match.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.og2_audit import (
    accession_composition,
    classify,
    rbcl_exposure,
    summarise,
)


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Nicotiana tabacum chloroplast, complete genome", "photosynthetic plastid"),
        ("Brighamia insignis plastome, complete sequence", "photosynthetic plastid"),
        ("Cyanophora paradoxa cyanelle, complete genome", "photosynthetic plastid"),
        ("Zea mays chloroplast, complete genome", "photosynthetic plastid"),
        # Apicoplasts are plastids but have lost rbcL with photosynthesis.
        ("Toxoplasma gondii RH apicoplast, complete genome", "apicoplast (rbcL-less)"),
        (
            "Plasmodium relictum strain SGS1 genome assembly, organelle: plastid:apicoplast",
            "apicoplast (rbcL-less)",
        ),
        ("Homo sapiens mitochondrion, complete genome", "mitochondrion"),
        ("Leishmania tarentolae kinetoplast, complete genome", "mitochondrion"),
        ("Plasmodium yoelii genome assembly PY17X01, chromosome : MIT", "mitochondrion"),
        ("Trichoderma hamatum, complete genome", "unclassified"),
    ],
)
def test_classify(title, expected):
    assert classify(title) == expected


def test_apicoplast_precedence_over_plastid():
    """A title containing both words must classify as apicoplast, not plastid.

    Real OG2 records are labelled 'organelle: plastid:apicoplast'. Pattern
    order decides this, so it is asserted rather than left implicit.
    """
    t = "Plasmodium gallinaceum strain 8A genome assembly, organelle: plastid:apicoplast"
    assert "plastid" in t and "apicoplast" in t
    assert classify(t) == "apicoplast (rbcL-less)"


def test_naive_match_agrees_on_count_but_not_on_set():
    """Why the careful classifier exists.

    A plain chloroplast|plastid match wrongly admits apicoplasts labelled
    'plastid:apicoplast' and wrongly drops plastome/cyanelle records. In the
    real partition the two errors cancel exactly, so a matching count is not
    evidence of a matching set -- this fixture reproduces that trap.
    """
    df = pd.DataFrame(
        {
            "acc": ["NC_1", "NC_2", "NC_3", "NC_4"],
            "title": [
                "Plasmodium relictum genome assembly, organelle: plastid:apicoplast",
                "Plasmodium gallinaceum genome assembly, organelle: plastid:apicoplast",
                "Brighamia insignis plastome, complete sequence",
                "Cyanophora paradoxa cyanelle, complete genome",
            ],
            "slen": [30_000, 30_000, 150_000, 135_000],
        }
    )
    naive = df.title.str.contains("chloroplast|plastid", case=False)
    careful = df.title.map(classify) == "photosynthetic plastid"
    assert int(naive.sum()) == int(careful.sum()) == 2  # counts agree
    assert set(df.acc[naive]) != set(df.acc[careful])  # sets do not


def test_primary_submission_accessions_are_detected():
    """The audit's central claim is zero primary accessions -- so the detector
    must actually fire when one is present."""
    refseq = pd.DataFrame({"acc": ["NC_000932.1", "NW_001.1", "NT_002.1"]})
    assert accession_composition(refseq)["primary_submission"] == 0
    # Typical barcode rbcL submission accessions.
    barcode = pd.DataFrame({"acc": ["AY123456.1", "Z00044.1", "KF148613.1"]})
    assert accession_composition(barcode)["primary_submission"] == 3


def test_rbcl_exposure_excludes_apicoplasts():
    df = pd.DataFrame(
        {
            "acc": ["NC_1", "NC_2"],
            "title": [
                "Nicotiana tabacum chloroplast, complete genome",
                "Toxoplasma gondii apicoplast, complete genome",
            ],
            "slen": [155_000, 35_000],
        }
    )
    e = rbcl_exposure(df)
    assert e["plastid_records"] == 1
    assert e["plastid_nt"] == 155_000
    # rbcL is a small fraction of a plastid genome, and plastid a small
    # fraction of OG2 -- exposure must land well below 1e-5.
    assert 0 < e["rbcl_frac_of_og2"] < 1e-5


def test_summarise_partitions_all_records():
    df = pd.DataFrame(
        {
            "acc": [f"NC_{i}" for i in range(4)],
            "title": [
                "x chloroplast, complete genome",
                "y mitochondrion, complete genome",
                "z apicoplast, complete genome",
                "w, complete genome",
            ],
            "slen": [150_000, 16_000, 35_000, 30_000],
        }
    )
    s = summarise(df)
    assert int(s.n_records.sum()) == len(df)
    assert abs(s.pct_nt.sum() - 100.0) < 0.05
