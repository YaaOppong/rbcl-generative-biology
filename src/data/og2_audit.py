"""OpenGenome2 partition audit: what plastid data did Evo 2 actually see?

Every headroom claim in this repo rests on Evo 2's rbcL exposure being limited
to whole plastid genomes, with no barcode-submission partition. This module
establishes that from the dataset itself rather than by assumption.

Two facts do the work:

1. OpenGenome2 has no barcode or marker-gene partition. Plastid sequence enters
   only via ``fasta/organelles/organelle_sequences.fasta.gz``.
2. That partition is RefSeq-only. Barcode rbcL submissions carry GenBank
   primary accessions exclusively, and there are zero such accessions present.

Classification is by record title, with one subtlety that a naive match gets
wrong: apicomplexan apicoplasts are plastids but non-photosynthetic and have
lost rbcL, so they must be excluded from the rbcL-bearing set. A plain
``chloroplast|plastid`` match returns the right *count* by cancelling two
errors against each other -- see ``test_partition_audit.py``.

Usage:
    python -m src.data.og2_audit --summary
    python -m src.data.og2_audit --verify-partition   # network: HF range read
"""
from __future__ import annotations

import argparse
import gzip
import io
import re
import urllib.request
from pathlib import Path

import pandas as pd

HF_ORGANELLE = (
    "https://huggingface.co/datasets/arcinstitute/opengenome2/resolve/main/"
    "fasta/organelles/organelle_sequences.fasta.gz"
)

# Plastid sensu lato: includes the relict plastids of apicomplexans and the
# cyanelle of glaucophytes, plus "plastome" which the obvious pattern misses.
PLASTID_RE = re.compile(r"chloroplast|plastid|plastome|apicoplast|cyanelle", re.IGNORECASE)
# Apicoplasts are plastids that have lost rbcL along with photosynthesis.
APICOPLAST_RE = re.compile(r"apicoplast", re.IGNORECASE)
MITO_RE = re.compile(r"mitochond|kinetoplast|chromosome ?: ?MIT", re.IGNORECASE)

# Card-reported figures for the organelle partition (arcinstitute/opengenome2).
ORGANELLE_TOKENS = 3.0e9
OG2_TOTAL_TOKENS = 9.3e12
RBCL_CDS_NT = 1438


def classify(title: str) -> str:
    """Assign one organelle class to a RefSeq record title."""
    if APICOPLAST_RE.search(title):
        return "apicoplast (rbcL-less)"
    if PLASTID_RE.search(title):
        return "photosynthetic plastid"
    if MITO_RE.search(title):
        return "mitochondrion"
    return "unclassified"


def classify_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["organelle_class"] = out["title"].fillna("").map(classify)
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    d = classify_frame(df)
    s = (
        d.groupby("organelle_class")
        .agg(n_records=("acc", "size"), total_nt=("slen", "sum"), mean_nt=("slen", "mean"))
        .assign(
            pct_records=lambda x: 100 * x.n_records / len(d),
            pct_nt=lambda x: 100 * x.total_nt / d.slen.sum(),
        )
        .sort_values("total_nt", ascending=False)
    )
    return s.round(2)


def rbcl_exposure(df: pd.DataFrame) -> dict:
    """Implied rbcL token exposure during Evo 2 pretraining."""
    d = classify_frame(df)
    photo = d[d.organelle_class == "photosynthetic plastid"]
    frac_nt = photo.slen.sum() / d.slen.sum()
    plastid_tokens = ORGANELLE_TOKENS * frac_nt
    rbcl_share = RBCL_CDS_NT / photo.slen.mean()
    rbcl_tokens = plastid_tokens * rbcl_share
    return {
        "plastid_records": len(photo),
        "plastid_nt": int(photo.slen.sum()),
        "plastid_frac_of_organelle_nt": round(frac_nt, 4),
        "plastid_tokens": plastid_tokens,
        "mean_plastid_genome_nt": float(photo.slen.mean()),
        "rbcl_share_of_plastid_genome": round(rbcl_share, 5),
        "rbcl_tokens": rbcl_tokens,
        "rbcl_frac_of_og2": rbcl_tokens / OG2_TOTAL_TOKENS,
    }


def accession_composition(df: pd.DataFrame) -> dict:
    """RefSeq vs primary-submission split. Primary accessions are the tell."""
    acc = df.acc.astype(str)
    return {
        "n": len(acc),
        "NC_": int(acc.str.startswith("NC_").sum()),
        "NW_": int(acc.str.startswith("NW_").sum()),
        "NT_": int(acc.str.startswith("NT_").sum()),
        # GenBank primary submissions: 1-2 letters + 5-6 digits, no underscore.
        "primary_submission": int(acc.str.match(r"^[A-Z]{1,2}\d{5,6}\.").sum()),
    }


def verify_partition(df: pd.DataFrame, n_bytes: int = 4_000_000) -> dict:
    """Range-read the partition head and check headers against our enumeration.

    Guards against the enumeration being a stale proxy for what the file holds.
    """
    req = urllib.request.Request(
        HF_ORGANELLE, headers={"User-Agent": "python-urllib", "Range": f"bytes=0-{n_bytes}"}
    )
    raw = urllib.request.urlopen(req, timeout=180).read()
    try:
        text = gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode("utf-8", "ignore")
    except EOFError:
        import zlib

        do = zlib.decompressobj(16 + zlib.MAX_WBITS)
        text = do.decompress(raw).decode("utf-8", "ignore")
    heads = [line[1:].strip() for line in text.split("\n") if line.startswith(">")]
    ours = list(df.acc.astype(str))
    return {
        "headers_read": len(heads),
        "in_enumeration": len(set(heads) & set(ours)),
        "order_matches": heads[: min(20, len(heads))] == ours[: min(20, len(heads))],
        "primary_submission_in_sample": sum(
            bool(re.match(r"^[A-Z]{1,2}\d{5,6}\.", h)) for h in heads
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary-csv", default="results/part_b/og2_organelle_summary.csv")
    ap.add_argument("--out", default="results/part_b/og2_organelle_classification.csv")
    ap.add_argument("--verify-partition", action="store_true", help="network range read")
    a = ap.parse_args()

    df = pd.read_csv(a.summary_csv)
    s = summarise(df)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(a.out)
    print(s.to_string())
    print()
    for k, v in accession_composition(df).items():
        print(f"  {k:<20} {v}")
    print()
    for k, v in rbcl_exposure(df).items():
        print(f"  {k:<32} {v:,.6g}" if isinstance(v, float) else f"  {k:<32} {v:,}")
    if a.verify_partition:
        print()
        for k, v in verify_partition(df).items():
            print(f"  {k:<32} {v}")


if __name__ == "__main__":
    main()
