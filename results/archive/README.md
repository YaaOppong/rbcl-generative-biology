# Archived runs — superseded, do not cite

Runs here produced valid numbers but have been superseded. They are kept because
a failed or preliminary run is a result about the code, and because the current
results cannot be read without knowing what they replaced. **Nothing in this
directory should be quoted in the README, a manuscript, or a figure caption.**

Each entry keeps its original `provenance.json`, so what ran is still on record.

---

## `2026-08-21T11-10-04_a317084_preliminary`

**First successful rbcL finetune.** Evo 2 7B + LoRA, B2 all-clade corpus
(3,869 CDS / 2,891 species / 24 clades). Modal job
`754077e9-28c3-4214-aa69-11cbab7485e2`, H100, 40 min, exit 0.

Headline numbers, verified by recomputing stop codons from the sequences rather
than trusting the run summary:

- validation loss 0.280 -> 0.134 on held-out species (`val_novel`), with the
  gap to `val_seen` staying within 0.004 nats
- ORF-clean generations 80.7% -> 100% (n=62 prompts)
- 52% of finetuned generations share >90% of their 32-mers with the training
  corpus (median 0.92, vs 0.00 for base) — half the output is largely copied
- the gain survives that: among the 14 generations copying <50% of k-mers,
  finetuned is 100% ORF-clean against 64% for base on the same prompts

### Why it is superseded

1. **Generation protocol does not match Part A.** This run generated 300 new
   tokens from a 90 nt prefix, single replicate. Part A generated to a uniform
   1,500 nt total across five prefix levels with 3 replicates. "ORF-clean over
   300 nt" is a much weaker claim than over a full ~1,428 nt CDS — roughly 4.7x
   fewer codons in which to hit a stop — so these numbers cannot be set against
   Part A's published baseline.

2. **The adapter weights no longer exist.** They were left in the compute
   sandbox and lost when it expired, so this run is documented but not
   reloadable: no further generation can be done from *this* model. Successor
   runs persist the adapter to the weights volume and harvest it back before
   generating.

3. **Adapter initialisation was not seeded.** `torch.manual_seed` was set but
   `torch.cuda.manual_seed_all` was not, and since the device fix adapters are
   constructed on CUDA — so `kaiming_uniform_` drew from an unseeded generator.
   The training DataLoader also shuffled from global RNG state. This run is
   therefore not reproducible at its own seed. Fixed after it ran.

4. **Model scale was right by accident.** The config requested
   `evo2_1b_base`; 1B failed to load (patched fp8 config written to `/tmp`,
   which evo2 resolves package-relative) and the fallback selected 7B. The
   correct model ran, for the wrong reason. Configs now name `evo2_7b`
   explicitly, with the matched-control argument recorded in `docs/DESIGN.md`.

**What still stands from it:** the corpus, the three-way split, the memorisation
measurement and its control. The memorisation finding in particular is not
protocol-dependent — it compares generations against the training corpus, not
against Part A.
