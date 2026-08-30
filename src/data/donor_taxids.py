"""Resolve the evaluation donors to species taxids, once, into a committed file.

The species-level holdout is the premise of Part B: if a donor's species is in
training, "did finetuning help?" becomes "can it recall what it memorised" for
that donor. The old manifest carried a precomputed heldout_donor_species flag
with no generating script, and it was wrong -- 24 of the 120 donors were marked
trainable and were trained on.

A taxid is the right key. Accessions miss the same species submitted under other
numbers; organism strings miss it too, and worse: 'UNVERIFIED: Pteris
pseudowulaiensis voucher ...' parses to the genus 'UNVERIFIED:' under any
whitespace split, so a name-based holdout silently fails on exactly the records
that need it most.

Usage:
    python -m src.data.donor_taxids --out data/evaluation_donor_taxids.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from Bio import Entrez

PROMPTS = Path("data/prompts_corpus.csv")
BATCH = 100


def donor_accessions(prompts: Path = PROMPTS) -> list[str]:
    with open(prompts) as fh:
        return sorted({row["donor_acc"] for row in csv.DictReader(fh)})


def resolve(accessions: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(accessions), BATCH):
        chunk = accessions[i:i + BATCH]
        h = Entrez.esummary(db="nuccore", id=",".join(chunk))
        for d in Entrez.read(h):
            out.append({"acc": d["AccessionVersion"], "taxid": int(d["TaxId"]),
                        "title": str(d["Title"])[:120]})
        h.close()
        time.sleep(0.4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/evaluation_donor_taxids.csv"))
    ap.add_argument("--email", default=None)
    args = ap.parse_args()
    if args.email:
        Entrez.email = args.email

    accs = donor_accessions()
    rows = resolve(accs)
    missing = {a.split(".")[0] for a in accs} - {r["acc"].split(".")[0] for r in rows}
    if missing:
        raise SystemExit(f"could not resolve {len(missing)} donor accessions: "
                         f"{sorted(missing)[:5]}")
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["acc", "taxid", "title"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["acc"]))
    print(f"{len(rows)} donors -> {len({r['taxid'] for r in rows})} distinct species taxids")


if __name__ == "__main__":
    main()
