"""Tests for the species-disjoint, clade-stratified train/validation split.

The load-bearing property is leakage: rbcL is a barcode locus, so the corpus
holds many records per species (B1: mean 2.4, max 87). Under a record-level
split, 71% of B1 validation records had their species in the training set, and
validation loss measured memorisation of near-identical congeneric sequence
rather than generalisation.

The second property is that a dominant species must not invert the split. An
earlier greedy largest-first implementation sent an 80-of-100-record species to
validation wholesale, producing an 80% validation fraction against a 10% target.
"""

from __future__ import annotations

import pytest

# torch is an optional extra: it needs a bf16-capable GPU for real use and
# is installed separately (see README). These modules exercise adapter
# injection and the loss on toy tensors, so they need torch but not a GPU.
# Skipping beats erroring: a fresh clone should get a clean run, not three
# collection failures that look like the suite is broken.
pytest.importorskip("torch")


import collections

import pytest

from src.train.lora_finetune import (
    clade_stratified_split,
    species_key,
    three_way_split,
)


def _rec(clade, taxid, seq="ATG" + "GCT" * 400):
    return {"clade": clade, "taxid": taxid, "organism": f"Genus sp{taxid}",
            "sequence": seq}


def _corpus():
    """Two clades, uneven records-per-species, one dominant species."""
    recs = []
    for i in range(1, 21):                       # Alpha: 20 species x 2 records
        recs += [_rec("Alpha", i)] * 2
    recs += [_rec("Beta", 100)] * 40             # Beta: one dominant species
    for i in range(101, 141):                    # ...plus 40 singletons
        recs.append(_rec("Beta", i))
    return recs


def test_no_species_appears_on_both_sides():
    train, val = clade_stratified_split(_corpus(), 0.1, seed=0)
    assert {species_key(r) for r in train} & {species_key(r) for r in val} == set()


def test_dominant_species_does_not_invert_the_split():
    """One species holding 40/80 of a clade must not become the validation set."""
    _, val = clade_stratified_split(_corpus(), 0.1, seed=0)
    beta_val = [r for r in val if r["clade"] == "Beta"]
    assert len(beta_val) <= 16, f"Beta validation ran to {len(beta_val)}/80"
    assert 100 not in {r["taxid"] for r in beta_val}


def test_realised_fraction_is_close_to_target():
    recs = _corpus()
    for frac in (0.1, 0.2):
        _, val = clade_stratified_split(recs, frac, seed=0)
        realised = len(val) / len(recs)
        assert abs(realised - frac) < 0.5 * frac, f"{realised:.3f} vs {frac}"


def test_every_clade_is_represented_in_training():
    train, _ = clade_stratified_split(_corpus(), 0.1, seed=0)
    assert {r["clade"] for r in train} == {"Alpha", "Beta"}


def test_single_species_clade_stays_in_training():
    """A clade with one species must not be moved wholesale to validation."""
    recs = _corpus() + [_rec("Solo", 999)]
    train, val = clade_stratified_split(recs, 0.1, seed=0)
    assert [r for r in train if r["clade"] == "Solo"]
    assert not [r for r in val if r["clade"] == "Solo"]


def test_clade_proportions_preserved_in_validation():
    recs = _corpus()
    _, val = clade_stratified_split(recs, 0.2, seed=0)
    whole = collections.Counter(r["clade"] for r in recs)
    held = collections.Counter(r["clade"] for r in val)
    for clade, n in whole.items():
        assert held[clade] / n == pytest.approx(0.2, abs=0.12), clade


def test_split_is_deterministic_and_seed_dependent():
    recs = _corpus()
    a = sorted(species_key(r) for r in clade_stratified_split(recs, 0.1, 0)[1])
    b = sorted(species_key(r) for r in clade_stratified_split(recs, 0.1, 0)[1])
    assert a == b
    seeds = {tuple(sorted(species_key(r) for r in clade_stratified_split(recs, 0.1, s)[1]))
             for s in range(6)}
    assert len(seeds) > 1, "seed does not change which species are held out"


def test_no_record_is_lost_or_duplicated():
    recs = _corpus()
    train, val = clade_stratified_split(recs, 0.1, seed=0)
    assert len(train) + len(val) == len(recs)


def test_species_key_prefers_taxid_over_name():
    """Two records with one organism string but different taxids are distinct."""
    a = {"taxid": 1, "organism": "Genus species", "clade": "X"}
    b = {"taxid": 2, "organism": "Genus species", "clade": "X"}
    assert species_key(a) != species_key(b)


def test_three_way_never_exceeds_the_requested_total():
    """Realised hold-back is <= val_fraction, and nothing is lost.

    val_seen is capped by donor availability -- one record per multi-record
    species that stays in training -- so a clade whose records are mostly
    singletons yields less than the target. Under-holding is safe;
    over-holding would silently shrink the training set.
    """
    recs = _corpus()
    train, seen, novel = three_way_split(recs, val_fraction=0.2, seed=0)
    assert len(train) + len(seen) + len(novel) == len(recs)
    assert (len(seen) + len(novel)) / len(recs) <= 0.2 + 1e-9


def test_realised_holdback_matches_target_when_donors_exist():
    """With enough multi-record species, the 20% target is actually met."""
    recs = [_rec("Gamma", i) for i in range(1, 61) for _ in range(3)]
    _train, seen, novel = three_way_split(recs, val_fraction=0.2, seed=0)
    realised = (len(seen) + len(novel)) / len(recs)
    assert abs(realised - 0.2) < 0.03, realised
    assert len(seen) > 0 and len(novel) > 0


def test_val_novel_is_species_disjoint_from_training():
    """The early-stopping signal must contain no species seen in training."""
    train, _seen, novel = three_way_split(_corpus(), val_fraction=0.2, seed=0)
    assert not ({species_key(r) for r in train} & {species_key(r) for r in novel})


def test_val_seen_donors_keep_a_record_in_training():
    """val_seen is leaky BY DESIGN -- but only if the donor really is in training.

    If a donor's last record went to val_seen, that species would be absent from
    training and val_seen would silently become a second novel set, collapsing
    the very contrast the two sets exist to measure.
    """
    train, seen, _novel = three_way_split(_corpus(), val_fraction=0.2, seed=0)
    trained = {species_key(r) for r in train}
    for r in seen:
        assert species_key(r) in trained, species_key(r)


def test_singletons_are_not_stranded_out_of_training():
    """Most single-record species must still contribute gradient.

    A pure species-disjoint split removes ~20% of species from training; the
    three-way split exists partly to keep that number low.
    """
    recs = _corpus()
    train, _seen, _novel = three_way_split(recs, val_fraction=0.2, seed=0)
    all_species = {species_key(r) for r in recs}
    trained = {species_key(r) for r in train}
    assert len(trained) / len(all_species) > 0.75


def test_three_way_is_deterministic_and_seed_dependent():
    recs = _corpus()
    a = three_way_split(recs, 0.2, seed=0)
    b = three_way_split(recs, 0.2, seed=0)
    assert [len(x) for x in a] == [len(x) for x in b]
    novel_sets = {
        tuple(sorted(species_key(r) for r in three_way_split(recs, 0.2, seed=s)[2]))
        for s in range(6)
    }
    assert len(novel_sets) > 1, "seed does not change which species are novel"


def test_novel_share_controls_the_two_halves():
    recs = _corpus()
    _t, seen, novel = three_way_split(recs, 0.2, novel_share=1.0, seed=0)
    assert seen == []
    assert len(novel) > 0
    _t, seen2, novel2 = three_way_split(recs, 0.2, novel_share=0.0, seed=0)
    assert novel2 == []
    assert len(seen2) > 0


def test_species_key_falls_back_to_binomial():
    """Records built before taxid was carried through must still group."""
    a = {"organism": "Pinnularia borealis strain ABC", "clade": "X"}
    b = {"organism": "Pinnularia borealis isolate XYZ", "clade": "X"}
    assert species_key(a) == species_key(b)
    assert species_key({"taxid": 0, "organism": "Pinnularia borealis"}) == species_key(a)