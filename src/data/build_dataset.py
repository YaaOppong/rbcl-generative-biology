"""Build LoRA finetuning datasets for rbcL from a committed accession manifest.

Sequences are NOT committed to this repository. This script fetches them from
NCBI GenBank using the accession manifest in data/finetune_accessions.csv, then
applies the species-level holdout that keeps evaluation donors out of training.

Usage:
    python -m src.data.build_dataset --arm B1_sparse_clade --out data/b1.jsonl
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import os
import time
from pathlib import Path
from urllib.error import URLError

import pandas as pd
from Bio import Entrez, SeqIO
from Bio.Seq import Seq

from src import provenance
from src.eval.metrics import is_full_length

MANIFEST = Path("data/finetune_accessions.csv")
# The full-coverage manifest, discovered by src.data.discover_accessions: every
# standalone full-CDS rbcL record in GenBank, no cap and no clade selection.
FULL_MANIFEST = Path("data/rbcl_fullcds_accessions.csv")
DONOR_TAXIDS = Path("data/evaluation_donor_taxids.csv")
# The Part A evaluation donors. Read at build time so the exclusion below is
# derived from the prompts that will actually be generated from, rather than
# trusting a precomputed column.
PROMPTS = Path("data/prompts_corpus.csv")
BATCH = 200
FETCH_ATTEMPTS = 4      # ~31 requests per arm; a drop at batch 30 must not restart it
FETCH_BACKOFF = 3.0     # seconds, doubling: 3, 6, 12
MIN_LEN = 1000  # floor for the legacy arms; the complete-CDS gate below is stricter

# Quality gates. Every one of these is a rejection, not a repair: a corpus is
# cheaper to shrink than a result is to caveat, and we have ~17k candidates.
#
# COMPLETE CDS is the gate that matters. The B2 corpus was 68.3% complete by the
# same predicate used to SCORE generations -- 23% of its sequences never
# terminated, and Diatoms terminated in only 45% of records. Training on
# non-terminating sequences while measuring whether generations terminate makes
# the training data teach the failure mode being scored.
#
# NO START-CODON GATE, deliberately. Requiring ATG looks like the natural
# companion to a complete-CDS gate, but it is confounded with clade: Red algae
# are 17.2% ATG and Brown algae 17.7%, against 96-100% for Eudicots, Mosses and
# Monocots, because algal barcode submissions are typically 5'-partial with a
# codon_start offset. Requiring it would cut the algal share from 36.9% to
# 17.3% and re-import the land-plant bias this corpus exists to correct. A
# 5'-truncated but otherwise complete CDS still carries that clade's codon usage
# and composition, which is what the finetune needs from it.
AMBIGUOUS_MAX = 0.0     # any non-ACGT base rejects the record


def is_in_frame(seq: str, max_ambiguous_frac: float = 0.01) -> bool:
    """True if seq is a plausible in-frame partial CDS.

    Checks: length is a multiple of three; no internal stop codon; IUPAC ambiguity
    codes are rare. Does NOT require a start codon or a terminal stop -- barcode
    rbcL records are overwhelmingly partial CDS at both ends, so requiring either
    would discard most of the corpus (only 166/300 of the B1 sample begin at ATG).
    """
    if len(seq) == 0 or len(seq) % 3 != 0:
        return False
    n_amb = sum(1 for c in seq if c not in "ACGT")
    if n_amb > max_ambiguous_frac * len(seq):
        return False
    prot = str(Seq(seq).translate())
    return prot[:-1].count("*") == 0


def base_acc(acc: str) -> str:
    """Strip the GenBank version suffix: 'PZ367540.1' -> 'PZ367540'.

    The manifest stores versioned accessions; efetch echoes them back versioned
    too, but the version can differ if a record was updated since the manifest was
    built. Keying on the unversioned accession on BOTH sides makes the join robust
    to that. Getting this wrong is silent: every lookup misses, every record is
    skipped, and the script still exits 0 having written an empty file.
    """
    return str(acc).split(".")[0]


def extract_cds(record, gene: str = "rbcL") -> str | None:
    """Extract the frame-correct CDS from a GenBank record.

    Two things here are easy to get wrong and both are silent:

    1. **codon_start.** GenBank annotates partial CDS features with a codon_start
       qualifier giving the offset of the first complete codon (1, 2 or 3).
       Biopython's ``feature.extract()`` returns the raw location span and does
       NOT apply it. Ignoring codon_start leaves the sequence out of frame: in the
       B1 sample, records with codon_start=3 translate to 38 internal stop codons
       instead of 0.
    2. **Trailing partial codon.** After applying the offset the sequence must be
       trimmed to a multiple of three, or translation runs off the end.

    Nearly every barcode rbcL submission is a *partial* CDS, so this is the common
    case rather than an edge case. Returns None if no matching CDS is annotated.
    """
    feats = [f for f in record.features if f.type == "CDS"]
    named = [
        f for f in feats if gene.lower() in str(f.qualifiers.get("gene", [""])[0]).lower()
    ]
    chosen = named or feats
    if not chosen:
        return infer_frame(str(record.seq).upper())
    f = max(chosen, key=lambda f: len(f))  # longest, if several
    sub = f.extract(record.seq)
    offset = int(f.qualifiers.get("codon_start", ["1"])[0]) - 1
    sub = sub[offset:]
    sub = sub[: len(sub) // 3 * 3]
    out = str(sub).upper()
    # Trust the annotation only if it actually yields a clean reading frame. A
    # small number of records carry a codon_start that disagrees with their own
    # sequence; falling back beats writing an out-of-frame record.
    return out if is_in_frame(out) else infer_frame(str(record.seq).upper())


def infer_frame(seq: str) -> str | None:
    """Recover the reading frame of an unannotated record by stop-codon search.

    Needed because CDS annotation is itself clade-structured: in the manifest
    audit, 100% of red algal and diatom records carried a CDS feature but only
    4% of eudicot records did. Requiring annotation would therefore silently
    discard whole clades -- the same shape of bias the frame fix exists to remove.

    Accepts a frame only if it is UNIQUELY stop-free. If two frames are both
    clean the record is ambiguous and is rejected rather than guessed at.
    """
    clean = []
    for off in (0, 1, 2):
        sub = seq[off:]
        sub = sub[: len(sub) // 3 * 3]
        if len(sub) >= MIN_LEN and is_in_frame(sub):
            clean.append(sub)
    return clean[0] if len(clean) == 1 else None


def fetch_sequences(
    accessions: list[str], email: str | None = None, gene: str = "rbcL"
) -> dict[str, str]:
    """Fetch frame-correct CDS from GenBank in batches. Returns {base_acc: seq}.

    Fetches GenBank flatfile rather than FASTA specifically to get the CDS feature
    table -- FASTA carries no codon_start, so a FASTA-based pipeline cannot put
    partial records in frame. Records with no annotated CDS are omitted (the caller
    counts them).

    Each batch is retried on transport failure. A full arm is ~31 requests over
    several minutes, and E-utilities drops connections mid-stream often enough
    that one IncompleteRead 30 batches in would otherwise throw the whole fetch
    away. Retries are per batch, so a success is never refetched. Passing
    ``email`` (or setting NCBI_EMAIL) buys a higher tolerance from NCBI.
    """
    if email:
        Entrez.email = email
    out: dict[str, str] = {}
    for i in range(0, len(accessions), BATCH):
        chunk = accessions[i : i + BATCH]
        for attempt in range(FETCH_ATTEMPTS):
            got: dict[str, str] = {}
            try:
                handle = Entrez.efetch(
                    db="nuccore", id=",".join(chunk), rettype="gb", retmode="text"
                )
                try:
                    for rec in SeqIO.parse(handle, "genbank"):
                        cds = extract_cds(rec, gene)
                        if cds:
                            got[base_acc(rec.id)] = cds
                finally:
                    handle.close()
            except (OSError, http.client.HTTPException, URLError) as e:
                # Partial batches are DISCARDED, not merged: a stream that died
                # mid-record can yield a truncated CDS, and a silently short
                # sequence in the training corpus is worse than a slow retry.
                if attempt == FETCH_ATTEMPTS - 1:
                    raise RuntimeError(
                        f"batch {i // BATCH} ({len(chunk)} accessions) failed after "
                        f"{FETCH_ATTEMPTS} attempts: {type(e).__name__}: {e}") from e
                back = FETCH_BACKOFF * 2 ** attempt
                print(f"  batch {i // BATCH}: {type(e).__name__}, retry in {back:.0f}s",
                      flush=True)
                time.sleep(back)
                continue
            out.update(got)
            break
        time.sleep(0.4)  # NCBI courtesy rate limit
    return out


def donor_species_taxids(path: Path = DONOR_TAXIDS) -> set[int]:
    """Species taxids of the 120 evaluation donors, for the species-level holdout.

    A taxid, not a flag and not a name. The old manifest's precomputed flag was
    wrong for 24 donors, and organism strings cannot be trusted either --
    "UNVERIFIED: Pteris pseudowulaiensis voucher ..." parses to the genus
    "UNVERIFIED:" under any whitespace split, which is precisely the record a
    name-based holdout must not miss.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required for the species-level holdout; build it with\n"
            "  python -m src.data.donor_taxids")
    with open(path) as fh:
        taxids = {int(r["taxid"]) for r in csv.DictReader(fh)}
    if not taxids:
        raise ValueError(f"no donor taxids in {path}")
    return taxids


def evaluation_donors(prompts: Path = PROMPTS) -> set[str]:
    """Base accessions of the Part A donors, versionless (AB050950.1 -> AB050950).

    Raises rather than returning an empty set when the file is missing: a silent
    no-op here is exactly how 24 donors ended up in the B2 training corpus.
    """
    if not prompts.exists():
        raise FileNotFoundError(
            f"{prompts} is required to exclude evaluation donors from training; "
            "it is tracked in git, so if it is missing, git pull")
    with open(prompts) as fh:
        donors = {row["donor_acc"].split(".")[0] for row in csv.DictReader(fh)}
    if not donors:
        raise ValueError(f"no donor accessions found in {prompts}")
    return donors


def is_complete_cds(seq: str) -> bool:
    """A full-length, in-frame ORF: the thing the model is asked to generate.

    Delegates to src.eval.metrics.is_full_length -- the SAME predicate that
    scores generated sequences. Training data and evaluation must agree on what
    counts as complete, or the corpus teaches one target and the metric rewards
    another.
    """
    return is_full_length(str(seq).upper())


def has_start_codon(seq: str) -> bool:
    """Begins at ATG.

    Optional, and confounded with clade -- which is why it is an ARM rather than
    a default. Red algae are 17.2% ATG and Brown algae 17.7%, against 96-100%
    for Eudicots, Mosses and Monocots, because algal barcode primers sit
    downstream of the start. Requiring it halves the algal share (34.4% ->
    17.3%) in a study whose subject is algal failure.

    Records without it are 5'-truncated by a median of 12 codons and at most 24
    (the 1,400 nt floor bounds it); the earliest active-site residue is 175, so
    nothing catalytic is lost. Whether that truncation matters is measurable,
    so it is measured: see configs/all_fullcds.yaml against
    configs/all_fullcds_atg.yaml.
    """
    return str(seq).upper().startswith("ATG")


def is_unambiguous(seq: str) -> bool:
    """No IUPAC ambiguity codes. 7.9% of B2's records carried at least one N,
    none above 1% of their length -- a few measurement gaps each. Sampling at
    generation is restricted to the four nucleotides anyway, so an N in training
    is noise that can never be reproduced."""
    s = str(seq).upper()
    return sum(c not in "ACGT" for c in s) <= AMBIGUOUS_MAX * len(s)


def _fetch_and_write(records, out_path: Path, email: str | None,
                     label: str = "", require_complete: bool = True,
                     require_start: bool = False) -> dict:
    """Fetch, filter, deduplicate and write one corpus. Returns the counts.

    REPRODUCIBILITY: records are sorted by accession before writing. The
    duplicate gate keeps the FIRST occurrence of each sequence, so without a
    defined order the corpus would depend on the manifest's row order -- two
    manifests holding the same accessions in a different order would produce
    different corpora, and the difference would be invisible in every summary
    statistic. Sorting makes the output a function of the accession SET alone.
    """
    records = records.sort_values("acc", kind="mergesort").copy()
    seqs = fetch_sequences(records.acc.astype(str).tolist(), email=email)

    n_missing = n_short = n_frame = n_dup = written = 0
    n_incomplete = n_ambiguous = n_no_start = 0
    written_nt = 0
    # Exact-duplicate gate. rbcL is a barcode locus: the same species is
    # sequenced by many labs over the same conserved region, so byte-identical
    # sequences recur under different accessions. A duplicate contributes no new
    # signal -- it only upweights that sequence in the gradient by sequencing
    # effort, which is a curation artifact, not biology. Keep the first
    # occurrence (by accession order) and count the rest.
    seen_seq: dict[str, str] = {}
    written_taxids: set[int] = set()
    with open(out_path, "w") as fh:
        for row in records.itertuples():
            seq = seqs.get(base_acc(row.acc))
            if seq is None:
                n_missing += 1
                continue
            if len(seq) < MIN_LEN:
                n_short += 1
                continue
            if require_complete and not is_complete_cds(seq):
                n_incomplete += 1
                continue
            if require_complete and not is_unambiguous(seq):
                n_ambiguous += 1
                continue
            if require_start and not has_start_codon(seq):
                n_no_start += 1
                continue
            digest = hashlib.sha256(seq.encode()).hexdigest()
            if digest in seen_seq:
                n_dup += 1
                continue
            seen_seq[digest] = row.acc
            # Frame gate. An out-of-frame training sequence teaches the model a
            # shifted codon distribution, and the effect is clade-structured:
            # in the B1 sample, land plants were 60/60 frame-correct under naive
            # FASTA extraction while algae were ~33/60. Since the primary
            # endpoint is algal pass rate, that confound would sit directly on
            # the result. Reject rather than repair.
            if not is_in_frame(seq):
                n_frame += 1
                continue
            fh.write(json.dumps({
                "accession": row.acc, "taxid": int(row.taxid),
                "organism": row.organism, "clade": row.clade, "sequence": seq,
            }) + "\n")
            written += 1
            written_nt += len(seq)
            written_taxids.add(int(row.taxid))

    # Fail loudly rather than leaving an empty or badly depleted training file for
    # the trainer to consume. A version-suffix mismatch between manifest and efetch
    # response would otherwise produce written=0 with exit code 0.
    if written == 0:
        raise RuntimeError(
            f"no sequences written {label}: {len(records)} requested, "
            f"{len(seqs)} fetched, {n_missing} unmatched, {n_short} below {MIN_LEN} nt, "
            f"{n_frame} out of frame. "
            "An accession-format mismatch between manifest and GenBank is the usual cause.")
    if written < 0.5 * len(records):
        print(f"WARNING: only {written}/{len(records)} records written "
              f"({n_missing} unmatched, {n_short} short, {n_frame} out of frame)"
              " -- check the manifest.")
    return {"requested": len(records), "fetched": len(seqs), "written": written,
            "unmatched": n_missing, "below_min_len": n_short,
            "rejected_incomplete_cds": n_incomplete,
            "rejected_ambiguous_bases": n_ambiguous,
            "rejected_no_start_codon": n_no_start,
            "complete_cds_required": require_complete,
            "start_codon_required": require_start,
            "rejected_out_of_frame": n_frame, "rejected_exact_duplicate": n_dup,
            "min_len": MIN_LEN, "total_nt": written_nt,
            "unique_sequences": len(seen_seq), "species_taxids": len(written_taxids),
            "species_taxids_requested": int(records.taxid.nunique()),
            "clades": int(records.clade.nunique()),
            "source": "NCBI GenBank via Entrez efetch"}


def derive_filtered(source: Path, out_path: Path, require_start: bool = True) -> dict:
    """Write a stricter corpus by filtering an existing one, without refetching.

    The start-codon arm differs from its twin by ONE predicate, so it is derived
    rather than rebuilt: refetching would introduce a second variable (GenBank
    can change between builds) and double the load on E-utilities for nothing.
    Both corpora therefore contain byte-identical sequences for every record they
    share, and the comparison isolates the gate.

    The source corpus is fingerprinted into the provenance, so the derivation is
    checkable after the fact.
    """
    with open(source) as fh:
        rows = [json.loads(line) for line in fh]
    kept = [r for r in rows if not require_start or has_start_codon(r["sequence"])]
    if not kept:
        raise RuntimeError(f"the filter removed every record from {source}")
    with open(out_path, "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in kept)
    src_digest = provenance.file_digest(source)
    prov = {"arm": "derived", "derived_from": str(source),
            "derived_from_sha256": src_digest["sha256"],
            "filter": "start_codon" if require_start else "none",
            "source_records": len(rows), "written": len(kept),
            "rejected_no_start_codon": len(rows) - len(kept),
            "start_codon_required": require_start,
            "total_nt": sum(len(r["sequence"]) for r in kept),
            "unique_sequences": len({r["sequence"] for r in kept}),
            "species_taxids": len({r["taxid"] for r in kept}),
            "clades": len({r["clade"] for r in kept}),
            "source": f"filtered from {source}"}
    Path(str(out_path) + ".provenance.json").write_text(json.dumps(prov, indent=2))
    return prov


def build_full(out_path: Path, email: str | None = None,
               limit: int | None = None, require_start: bool = False) -> dict:
    """Every full-CDS rbcL record in GenBank, minus the evaluation lineages.

    No cap and no clade selection, so nothing here depends on the old manifest's
    unreproducible arm assignment. The only exclusion is the species-level
    holdout, and it is derived from taxids rather than a precomputed flag.

    Balance is deliberately NOT imposed. The corpus inherits GenBank's own
    composition -- 60% vascular plants, 33% algae -- which is already a large
    re-weighting against Evo 2's 95/5, and leaves the capping question to be
    answered by comparison rather than assumed.
    """
    man = pd.read_csv(FULL_MANIFEST)
    n_before = len(man)
    held = donor_species_taxids()
    man = man[~man.taxid.isin(held)].copy()
    n_excluded = n_before - len(man)

    if limit is not None and limit < len(man):
        per = max(1, limit // max(1, man.clade.nunique()))
        man = man.groupby("clade", group_keys=False)[man.columns].head(per).head(limit).copy()

    arm = "ALL_fullcds_atg" if require_start else "ALL_fullcds"
    rec = _fetch_and_write(man, out_path, email, f"for {arm}", require_start=require_start)
    prov = {"arm": arm, "manifest": str(FULL_MANIFEST),
            "manifest_rows": n_before,
            "excluded_donor_species_rows": n_excluded,
            "held_species_taxids": len(held), "limit": limit, **rec}
    Path(str(out_path) + ".provenance.json").write_text(json.dumps(prov, indent=2))
    return prov


def build(
    arm: str, out_path: Path, email: str | None = None, limit: int | None = None
) -> dict:
    """Assemble one training arm. Returns a provenance record.

    limit: cap the number of records (for smoke-testing the pipeline). Sampling is
    stratified by clade so a small limit still spans the arm's taxonomic range.
    """
    if arm in ("ALL_fullcds", "ALL_fullcds_atg"):
        return build_full(out_path, email, limit,
                          require_start=arm.endswith("_atg"))

    man = pd.read_csv(MANIFEST)
    man = man[man.arm == arm].copy()
    if man.empty:
        raise ValueError(f"no manifest rows for arm={arm!r}")

    # --- LEAKAGE CONTROL ---------------------------------------------------
    # Evaluation uses 120 donor sequences. Excluding those *accessions* is not
    # sufficient: the same species appears in GenBank under other accessions,
    # so an accession-level filter still leaks the exact lineages we evaluate
    # on. We exclude at SPECIES level. See docs/DESIGN.md.
    n_before = len(man)
    train = man[~man.heldout_donor_species].copy()
    n_excluded = n_before - len(train)

    # ...and then at ACCESSION level, as a floor under that. The species flag is
    # a precomputed column in the manifest with no generating script in this
    # repo, so it cannot be regenerated or audited -- and it is wrong: 24 of the
    # 120 evaluation donors carried heldout_donor_species=False in the
    # B2_balanced pool and were trained on, their own target sequences included.
    # A donor inside training turns "did finetuning help?" into "can it recall
    # what it memorised" for that donor. This filter is derived from
    # prompts_corpus.csv, so it holds however the column was computed.
    donors = evaluation_donors()
    is_donor = train.acc.astype(str).str.split(".").str[0].isin(donors)
    n_donor_acc = int(is_donor.sum())
    train = train[~is_donor].copy()

    # ...and at SPECIES level by taxid, which is the control the flag was meant
    # to be. The flag missed 24 donors outright; it also leaves donor SPECIES
    # behind under other accessions -- three of them survived into B1 with both
    # filters above in place, because that species' other records carry
    # heldout_donor_species=False and a different accession.
    held = donor_species_taxids()
    is_held = train.taxid.isin(held)
    n_donor_species = int(is_held.sum())
    train = train[~is_held].copy()

    if limit is not None and limit < len(train):
        per = max(1, limit // max(1, train.clade.nunique()))
        train = (
            train.groupby("clade", group_keys=False)[train.columns]
            .head(per)
            .head(limit)
            .copy()
        )

    rec = _fetch_and_write(train, out_path, email, f'for arm={arm!r}')


    prov = {
        "arm": arm,
        "manifest": str(MANIFEST),
        "manifest_rows": n_before,
        "excluded_donor_species_rows": n_excluded,
        "excluded_donor_accessions": n_donor_acc,
        "excluded_donor_species_by_taxid": n_donor_species,
        "limit": limit,
        **rec,
    }
    Path(str(out_path) + ".provenance.json").write_text(json.dumps(prov, indent=2))
    return prov


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--derive-from", type=Path, default=None,
                    help="filter an existing corpus instead of refetching; used "
                         "for the start-codon arm so both corpora share source bytes")
    ap.add_argument("--arm", required=True,
                    choices=["B1_sparse_clade", "B2_balanced",
                             "ALL_fullcds", "ALL_fullcds_atg"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--email",
        default=os.environ.get("NCBI_EMAIL"),
        help="Contact email for NCBI E-utilities (optional; NCBI requests one).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap record count for a smoke test (clade-stratified). Omit for the full arm.",
    )
    args = ap.parse_args()
    if args.derive_from is not None:
        print(json.dumps(derive_filtered(
            args.derive_from, args.out,
            require_start=args.arm.endswith("_atg")), indent=2))
    else:
        print(json.dumps(build(args.arm, args.out, args.email, args.limit), indent=2))
