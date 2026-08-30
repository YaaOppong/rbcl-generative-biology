"""Frame-quality audit: naive FASTA extraction vs annotated CDS + codon_start.

Quantifies the clade-structured frame confound across the manifest. Reuses the
pipeline's own extract_cds/infer_frame so the audit measures shipped code rather
than a reimplementation.

    python -m src.data.frame_audit --sample data/finetune_accessions.csv \
        --per-clade 150 --out results/part_b/frame_audit_records.csv
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd
from Bio import Entrez, SeqIO
from Bio.Seq import Seq

from src.data.build_dataset import base_acc, extract_cds, is_in_frame

BATCH = 150


def naive_frame(raw: str) -> tuple[int, int]:
    """Frame stats under the naive path: whole record, frame 0, trimmed to /3."""
    trimmed = raw[: len(raw) // 3 * 3]
    stops = str(Seq(trimmed).translate())[:-1].count("*") if trimmed else -1
    return len(raw) % 3, stops


def audit(sample: pd.DataFrame, email: str) -> pd.DataFrame:
    Entrez.email = email
    meta = {
        base_acc(str(a)): (c, o)
        for a, c, o in zip(sample.acc, sample.clade, sample.organism, strict=False)
    }
    accs = [str(a) for a in sample.acc]
    rows = []
    for i in range(0, len(accs), BATCH):
        handle = Entrez.efetch(
            db="nuccore", id=",".join(accs[i : i + BATCH]), rettype="gb", retmode="text"
        )
        for rec in SeqIO.parse(handle, "genbank"):
            b = base_acc(rec.id)
            clade, org = meta.get(b, ("?", "?"))
            raw = str(rec.seq).upper()
            mod3, nstops = naive_frame(raw)
            feats = [f for f in rec.features if f.type == "CDS"]
            cs = int(feats[0].qualifiers.get("codon_start", ["1"])[0]) if feats else 0
            cds = extract_cds(rec)
            rows.append(
                {
                    "acc": b,
                    "clade": clade,
                    "organism": org,
                    "raw_len": len(raw),
                    "has_cds": bool(feats),
                    "codon_start": cs,
                    "recovered": bool(cds) and not feats,
                    "cds_len": len(cds) if cds else 0,
                    "naive_mod3": mod3,
                    "naive_stops": nstops,
                    "naive_ok": (mod3 == 0 and nstops == 0),
                    "cds_ok": bool(cds) and is_in_frame(cds),
                }
            )
        handle.close()
        time.sleep(0.4)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    tab = df.groupby("clade").agg(
        n=("acc", "size"),
        annot_pct=("has_cds", lambda s: 100 * s.mean()),
        naive_pct=("naive_ok", lambda s: 100 * s.mean()),
        fixed_pct=("cds_ok", lambda s: 100 * s.mean()),
        recov=("recovered", "sum"),
    )
    return tab[tab.n >= min_n].round(1).sort_values("naive_pct")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", default="data/finetune_accessions.csv")
    ap.add_argument("--per-clade", type=int, default=150)
    ap.add_argument("--out", default="results/part_b/frame_audit_records.csv")
    ap.add_argument("--email", default=os.environ.get("NCBI_EMAIL"))
    args = ap.parse_args()
    if not args.email:
        raise SystemExit("set --email or NCBI_EMAIL (NCBI requires a contact address)")

    man = pd.read_csv(args.sample)
    pool = man[~man.heldout_donor_species] if "heldout_donor_species" in man else man
    samp = pool.groupby("clade", group_keys=False)[pool.columns].apply(
        lambda d: d.head(args.per_clade)
    )
    df = audit(samp, args.email)
    df.to_csv(args.out, index=False)
    tab = summarise(df)
    tab.to_csv(args.out.replace("_records", "_by_clade"))
    print(tab.to_string())
    print(
        f"\n{len(df)} records | naive {100 * df.naive_ok.mean():.1f}% "
        f"-> corrected {100 * df.cds_ok.mean():.1f}% frame-correct"
    )


if __name__ == "__main__":
    main()
