"""Export the finetune corpus to BioNeMo/Megatron preprocessing inputs.

Why this module exists: `NVIDIA/bionemo-framework` ships a first-party LoRA
path for Evo 2 (`recipes/evo2_megatron`, `Evo2LoRA` on the Megatron Bridge PEFT
stack), which is what we train with. It consumes FASTA converted to Megatron
indexed-binary form by `preprocess_evo2`, not our JSONL.

THE LOAD-BEARING DETAIL. `preprocess_evo2` takes `train_split`/`valid_split`/
`test_split` and splits the input itself -- at random, by record. Handing it one
FASTA and asking for a 20% validation slice would silently discard our
species-disjoint design and reintroduce the leakage measured in DESIGN.md (71%
of validation records sharing a species with training; conspecific records are
median 99.26% identical). So we write each split as its OWN FASTA and
preprocess each with `train_split: 1.0`, giving BioNeMo no splitting decision
to make. The upstream config's own comment confirms this is the intended
pattern for manual splits.

Lineage tags: Evo 2 was pretrained with taxonomic lineage prefixes, and
`preprocess_evo2` injects them via a `taxonomy_data` map keyed on the FASTA
sequence id. We emit that map from the corpus so finetuning uses the same
conditioning format as pretraining rather than bare sequence.
"""

from __future__ import annotations

import json
from pathlib import Path

# Ranks preprocess_evo2's taxonomy_data accepts. 'clazz' is its spelling.
LINEAGE_FIELDS = ("kingdom", "phylum", "clazz", "order", "family", "genus", "species")

# Our clade labels are informal groupings, not formal ranks. Map each to the
# rank pair that places it, so a lineage tag is not silently wrong.
CLADE_TO_LINEAGE = {
    "Red algae": {"kingdom": "Eukaryota", "phylum": "Rhodophyta"},
    "Diatoms": {"kingdom": "Eukaryota", "phylum": "Bacillariophyta"},
    "Brown algae": {"kingdom": "Eukaryota", "phylum": "Ochrophyta"},
    "Green algae": {"kingdom": "Eukaryota", "phylum": "Chlorophyta"},
    "Mosses": {"kingdom": "Eukaryota", "phylum": "Bryophyta"},
    "Liverworts": {"kingdom": "Eukaryota", "phylum": "Marchantiophyta"},
    "Hornworts": {"kingdom": "Eukaryota", "phylum": "Anthocerotophyta"},
    "Ferns": {"kingdom": "Eukaryota", "phylum": "Polypodiophyta"},
    "Lycophytes": {"kingdom": "Eukaryota", "phylum": "Lycopodiophyta"},
    "Conifers": {"kingdom": "Eukaryota", "phylum": "Pinophyta"},
    "Cycads": {"kingdom": "Eukaryota", "phylum": "Cycadophyta"},
    "Ginkgo": {"kingdom": "Eukaryota", "phylum": "Ginkgophyta"},
    "Gnetophytes": {"kingdom": "Eukaryota", "phylum": "Gnetophyta"},
    "Charophytes": {"kingdom": "Eukaryota", "phylum": "Charophyta"},
    "Haptophytes": {"kingdom": "Eukaryota", "phylum": "Haptophyta"},
    "Dinoflagellates": {"kingdom": "Eukaryota", "phylum": "Myzozoa"},
}


def lineage_for(record: dict) -> dict:
    """Taxonomy entry for one record.

    Genus and species come from the organism binomial; higher ranks from the
    clade map. Ranks we cannot determine are omitted rather than guessed --
    preprocess_evo2 accepts a partial map (see its own test config, which
    supplies only kingdom/order/family for one accession).
    """
    entry = dict(CLADE_TO_LINEAGE.get(record.get("clade", ""), {}))
    tokens = str(record.get("organism", "")).split()
    if tokens:
        entry["genus"] = tokens[0]
    if len(tokens) > 1:
        entry["species"] = tokens[1]
    return {k: v for k, v in entry.items() if k in LINEAGE_FIELDS}


def write_fasta(records: list[dict], path: Path) -> dict[str, dict]:
    """Write records as FASTA; return the taxonomy_data map keyed on seq id."""
    taxonomy: dict[str, dict] = {}
    with open(path, "w") as fh:
        for rec in records:
            seq_id = str(rec["accession"])
            taxonomy[seq_id] = lineage_for(rec)
            fh.write(f">{seq_id}\n")
            seq = str(rec["sequence"]).upper()
            fh.writelines(seq[i : i + 80] + "\n" for i in range(0, len(seq), 80))
    return taxonomy


def preproc_config(
    fasta: Path, output_dir: Path, prefix: str, taxonomy: dict[str, dict], seed: int
) -> list[dict]:
    """A preprocess_evo2 config for ONE split.

    train_split is pinned to 1.0 and valid/test to 0.0 deliberately: the split
    has already been decided species-disjointly upstream, and letting
    preprocess_evo2 re-split would discard that. See the module docstring.
    """
    return [
        {
            "datapaths": [str(fasta)],
            "output_dir": str(output_dir),
            "output_prefix": prefix,
            "train_split": 1.0,
            "valid_split": 0.0,
            "test_split": 0.0,
            "overwrite": True,
            # Both strands are biologically real for a coding sequence, and Evo 2
            # was pretrained with reverse complements embedded.
            "embed_reverse_complement": True,
            "random_reverse_complement": 0.0,
            # No lineage dropout: the clade conditioning IS the variable under
            # test in Part B, so degrading it at random would confound the arm.
            "random_lineage_dropout": 0.0,
            "transcribe": None,
            "force_uppercase": True,
            "indexed_dataset_dtype": "uint8",
            "append_eod": True,
            "enforce_sample_length": None,
            "ftfy": False,
            "tokenizer_type": "Byte-Level",
            "vocab_file": None,
            "vocab_size": None,
            "merges_file": None,
            "tokenizer_model_name": None,
            "pretrained_tokenizer_model": None,
            "special_tokens": None,
            "fast_hf_tokenizer": True,
            "workers": 4,
            "preproc_concurrency": 100000,
            "chunksize": 25,
            "drop_empty_sequences": True,
            "nnn_filter": True,
            "seed": seed,
            "taxonomy_data": taxonomy,
        }
    ]


def export(
    jsonl: Path,
    out_dir: Path,
    val_fraction: float = 0.2,
    novel_share: float = 0.5,
    seed: int = 0,
) -> dict:
    """Write train/val_seen/val_novel FASTAs + one preproc config per split."""
    from src.train.lora_finetune import species_key, three_way_split

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(jsonl) as fh:
        records = [json.loads(line) for line in fh]
    train, seen, novel = three_way_split(records, val_fraction, novel_share, seed)

    manifest = {"source": str(jsonl), "seed": seed, "splits": {}}
    for name, rows in (("train", train), ("val_seen", seen), ("val_novel", novel)):
        fasta = out_dir / f"{name}.fasta"
        taxonomy = write_fasta(rows, fasta)
        cfg = preproc_config(fasta, out_dir, f"rbcl_{name}", taxonomy, seed)
        (out_dir / f"preproc_{name}.yaml").write_text(
            json.dumps(cfg, indent=2)  # valid YAML: YAML is a JSON superset
        )
        manifest["splits"][name] = {
            "records": len(rows),
            "species": len({species_key(r) for r in rows}),
            "nt": sum(len(r["sequence"]) for r in rows),
            "fasta": str(fasta),
            "config": str(out_dir / f"preproc_{name}.yaml"),
        }

    # Leakage assertion travels with the data, not just the test suite: if this
    # ever fires, the exported corpus is unusable for the Part B claim.
    tr_sp = {species_key(r) for r in train}
    nv_sp = {species_key(r) for r in novel}
    leaked = tr_sp & nv_sp
    if leaked:
        raise RuntimeError(
            f"{len(leaked)} species in both train and val_novel: {sorted(leaked)[:5]}"
        )
    manifest["leaked_species"] = 0
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--novel-share", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(export(Path(a.jsonl), Path(a.out_dir), a.val_fraction,
                            a.novel_share, a.seed), indent=2)[:1200])
