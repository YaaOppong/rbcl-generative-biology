"""Tests for the BioNeMo/Megatron export.

The load-bearing property: `preprocess_evo2` splits its input itself, at random
and by record, via train_split/valid_split/test_split. If the export ever hands
it more than one split's worth of data, or asks it for a nonzero
valid/test_split, the species-disjoint design is silently discarded and the
leakage measured in DESIGN.md returns. Every config must therefore pin
train_split=1.0.
"""

from __future__ import annotations

import pytest

# torch is an optional extra: it needs a bf16-capable GPU for real use and
# is installed separately (see README). These modules exercise adapter
# injection and the loss on toy tensors, so they need torch but not a GPU.
# Skipping beats erroring: a fresh clone should get a clean run, not three
# collection failures that look like the suite is broken.
pytest.importorskip("torch")


import json

import pytest

from src.train.bionemo_export import export, lineage_for, write_fasta


def _rec(acc, clade, organism, seq="ATG" + "GCT" * 400):
    return {"accession": acc, "clade": clade, "organism": organism,
            "taxid": abs(hash(organism)) % 100000, "sequence": seq}


@pytest.fixture
def corpus(tmp_path):
    rows = []
    for i in range(30):
        rows.append(_rec(f"AC{i}.1", "Red algae", f"Porphyra sp{i}"))
        rows.append(_rec(f"AC{i}b.1", "Red algae", f"Porphyra sp{i}"))  # conspecific
    for i in range(20):
        rows.append(_rec(f"BD{i}.1", "Diatoms", f"Eunotia sp{i}"))
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_every_config_pins_train_split(corpus, tmp_path):
    """The whole point: BioNeMo must make no splitting decision."""
    out = tmp_path / "bn"
    export(corpus, out, seed=0)
    configs = sorted(out.glob("preproc_*.yaml"))
    assert len(configs) == 3
    for c in configs:
        cfg = json.loads(c.read_text())[0]
        assert cfg["train_split"] == 1.0, c.name
        assert cfg["valid_split"] == 0.0, c.name
        assert cfg["test_split"] == 0.0, c.name


def test_each_config_reads_exactly_one_fasta(corpus, tmp_path):
    """One datapath per config -- concatenating splits would remix them."""
    out = tmp_path / "bn"
    export(corpus, out, seed=0)
    for c in out.glob("preproc_*.yaml"):
        assert len(json.loads(c.read_text())[0]["datapaths"]) == 1


def test_export_raises_if_species_leak(corpus, tmp_path, monkeypatch):
    """The assertion travels with the data, not only the test suite."""
    import src.train.bionemo_export as mod

    def leaky(records, val_fraction, novel_share, seed):
        return records, [], records  # same records both sides
    monkeypatch.setattr(mod, "three_way_split", leaky, raising=False)
    monkeypatch.setattr(
        "src.train.lora_finetune.three_way_split", leaky, raising=False
    )
    with pytest.raises(RuntimeError, match="both train and val_novel"):
        export(corpus, tmp_path / "bad", seed=0)


def test_lineage_dropout_is_disabled(corpus, tmp_path):
    """Clade conditioning is the variable under test in Part B.

    preprocess_evo2 defaults random_lineage_dropout to 0.1; dropping lineage
    tags at random would degrade exactly the signal the arm manipulates.
    """
    out = tmp_path / "bn"
    export(corpus, out, seed=0)
    for c in out.glob("preproc_*.yaml"):
        assert json.loads(c.read_text())[0]["random_lineage_dropout"] == 0.0


def test_every_record_gets_a_lineage_tag(corpus, tmp_path):
    out = tmp_path / "bn"
    export(corpus, out, seed=0)
    for name in ("train", "val_seen", "val_novel"):
        fasta = (out / f"{name}.fasta").read_text()
        ids = [ln[1:].strip() for ln in fasta.splitlines() if ln.startswith(">")]
        tax = json.loads((out / f"preproc_{name}.yaml").read_text())[0]["taxonomy_data"]
        assert set(ids) == set(tax), name
        assert all(tax[i] for i in ids), f"{name} has empty lineage entries"


def test_lineage_uses_real_ranks_not_the_informal_clade():
    """'Red algae' is not a phylum name; Rhodophyta is."""
    e = lineage_for(_rec("X.1", "Red algae", "Porphyra umbilicalis"))
    assert e["phylum"] == "Rhodophyta"
    assert e["genus"] == "Porphyra" and e["species"] == "umbilicalis"
    assert "Red algae" not in e.values()


def test_unknown_clade_yields_partial_not_wrong_lineage():
    """An unmapped clade omits higher ranks rather than inventing them."""
    e = lineage_for(_rec("X.1", "Not a clade", "Genus species"))
    assert "phylum" not in e
    assert e["genus"] == "Genus"


def test_fasta_is_wrapped_and_uppercase(tmp_path):
    rows = [_rec("A.1", "Diatoms", "Eunotia sp1", seq="atgc" * 50)]
    tax = write_fasta(rows, tmp_path / "x.fasta")
    lines = (tmp_path / "x.fasta").read_text().splitlines()
    assert lines[0] == ">A.1"
    assert all(len(ln) <= 80 for ln in lines[1:])
    assert "".join(lines[1:]) == ("ATGC" * 50)
    assert tax["A.1"]["genus"] == "Eunotia"


def test_manifest_totals_match_the_corpus(corpus, tmp_path):
    out = tmp_path / "bn"
    man = export(corpus, out, seed=0)
    with open(corpus) as fh:
        n_in = sum(1 for _ in fh)
    assert sum(s["records"] for s in man["splits"].values()) == n_in
    assert man["leaked_species"] == 0