"""Find every full-length rbcL record in GenBank, per clade, and write a manifest.

This step did not exist. data/finetune_accessions.csv is a committed artefact
with no generating script: its arm assignment, its 780 cap and its
heldout_donor_species flag could not be regenerated, audited, or extended. Three
defects followed from that -- 24 evaluation donors carried the wrong flag and
were trained on; the cap was applied before the species holdout, so no clade
actually reached 780; and the cap's value was the median of pre-holdout,
pre-dedup availability, which is two filters upstream of where it takes effect.

None of the recorded per-clade counts reproduce from an obvious query either:
Rhodophyta is recorded at 2,338 and returns 3,316 today, and neither
"complete cds"[Title] (445), NOT UNVERIFIED[Title] (3,313) nor a 2026 date cut
(3,310) recovers it. So this module does not attempt to match the old numbers.
It states its query, records it in the output, and lets the counts be whatever
GenBank currently holds.

THE QUERY

    rbcL[Gene] AND <taxon>[Organism] AND 1400:1500[SLEN]

SLEN is the length of the RECORD, not of the CDS, and that is the point: it
selects standalone rbcL submissions -- barcode records -- rather than whole
plastid genomes that happen to contain rbcL. Those genome records are precisely
what Evo 2 already trained on (its organelle partition is 100% RefSeq), so
including them would re-import the land-plant bulk this finetune exists to
counterweight. 1400:1500 brackets the natural CDS range (longest observed 1,497)
with room for short flanks.

Usage:
    python -m src.data.discover_accessions --out data/rbcl_fullcds_accessions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from Bio import Entrez

# clade -> NCBI taxon, from results/part_b/clade_representation.csv. Ordered so
# the largest clades fail first if the query is wrong.
CLADE_TAXA = {
    "Eudicots": "eudicotyledons",
    "Monocots": "Liliopsida",
    "Red algae": "Rhodophyta",
    "Magnoliids": "Magnoliidae",
    "Diatoms": "Bacillariophyta",
    "Mosses": "Bryophyta",
    "Liverworts": "Marchantiophyta",
    "Brown algae": "Phaeophyceae",
    "Green algae": "Chlorophyta",
    "Conifers": "Pinopsida",
    "Ferns": "Polypodiopsida",
    "Lycophytes": "Lycopodiopsida",
    "Cycads": "Cycadopsida",
    "Gnetophytes": "Gnetidae",
    "Ginkgo": "Ginkgoopsida",
    "Hornworts": "Anthocerotophyta",
    "Eustigmatophytes": "Eustigmatophyceae",
    "Haptophytes": "Haptophyta",
    "Dinoflagellates": "Dinophyceae",
    "Cryptophytes": "Cryptophyceae",
    "Glaucophytes": "Glaucophyta",
    "Euglenids": "Euglenida",
    "Chlorarachniophytes": "Chlorarachniophyceae",
    "Streptophyte algae": "Charophyta",
    "Zygnematophyceae": "Zygnematophyceae",
    "Klebsormidiophyceae": "Klebsormidiophyceae",
}

TERM = "rbcL[Gene] AND {taxon}[Organism] AND 1400:1500[SLEN]"
SUMMARY_BATCH = 300
ATTEMPTS = 4
BACKOFF = 3.0


def _retry(fn, what: str):
    for attempt in range(ATTEMPTS):
        try:
            return fn()
        except Exception as e:
            if attempt == ATTEMPTS - 1:
                raise RuntimeError(f"{what} failed after {ATTEMPTS} attempts: {e}") from e
            back = BACKOFF * 2 ** attempt
            print(f"    {what}: {type(e).__name__}, retry in {back:.0f}s", flush=True)
            time.sleep(back)
    return None


def search_clade(taxon: str) -> list[str]:
    """Every matching accession id for one taxon, via the history server.

    usehistory + a paged fetch, because a bare retmax silently truncates: the
    previous retrieval hit a RETMAX=8000 ordering cap and lost whole clades
    (results/part_b/clade_coverage_gap.csv records diatoms, lycophytes, cycads
    and five more as retrieved 0.0).
    """
    term = TERM.format(taxon=taxon)
    h = _retry(lambda: Entrez.esearch(db="nuccore", term=term, retmax=0, usehistory="y"),
               f"esearch {taxon}")
    rec = Entrez.read(h); h.close()
    total = int(rec["Count"])
    ids: list[str] = []
    for start in range(0, total, 5000):
        h = _retry(lambda s=start: Entrez.esearch(
            db="nuccore", term=term, retstart=s, retmax=5000,
            webenv=rec["WebEnv"], query_key=rec["QueryKey"]), f"esearch page {taxon}")
        ids += Entrez.read(h)["IdList"]; h.close()
        time.sleep(0.4)
    if len(ids) != total:
        raise RuntimeError(f"{taxon}: retrieved {len(ids)} of {total} ids")
    return ids


def summarise(ids: list[str]) -> list[dict]:
    """accession, taxid, organism and length for each id."""
    out = []
    for i in range(0, len(ids), SUMMARY_BATCH):
        chunk = ids[i:i + SUMMARY_BATCH]
        h = _retry(lambda c=chunk: Entrez.esummary(db="nuccore", id=",".join(c)),
                   f"esummary batch {i // SUMMARY_BATCH}")
        for d in Entrez.read(h):
            out.append({"acc": d["AccessionVersion"], "taxid": int(d["TaxId"]),
                        "organism": str(d["Title"]), "slen": int(d["Length"])})
        h.close()
        time.sleep(0.4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/rbcl_fullcds_accessions.csv"))
    ap.add_argument("--email", default=None, help="contact address for NCBI E-utilities")
    ap.add_argument("--clades", default=None, help="comma-separated subset, for testing")
    args = ap.parse_args()
    if args.email:
        Entrez.email = args.email

    wanted = ([c.strip() for c in args.clades.split(",")] if args.clades
              else list(CLADE_TAXA))
    rows, counts = [], {}
    for clade in wanted:
        taxon = CLADE_TAXA[clade]
        ids = search_clade(taxon)
        recs = summarise(ids)
        for r in recs:
            r["clade"] = clade
            r["taxon"] = taxon
        rows += recs
        counts[clade] = len(recs)
        print(f"  {clade:22s} {taxon:22s} {len(recs):6,}", flush=True)

    # An accession can match two taxon queries (nested taxa); keep the first.
    seen, unique = set(), []
    for r in rows:
        if r["acc"] not in seen:
            seen.add(r["acc"]); unique.append(r)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["acc", "taxid", "organism", "clade", "taxon", "slen"])
        w.writeheader()
        w.writerows(unique)
    meta = {"query": TERM, "clades": counts, "rows": len(unique),
            "duplicate_accessions_dropped": len(rows) - len(unique),
            "species_taxids": len({r["taxid"] for r in unique})}
    Path(str(args.out) + ".provenance.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2)[:1200])


if __name__ == "__main__":
    main()
