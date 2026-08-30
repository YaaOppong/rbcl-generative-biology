# Generation

`runner.py` is the live decoding procedure, shared by every paper part:
batching, per-batch seeding, stop-codon truncation, the output schema and the
sampling constants. Parts A, B and C differ in their **arms**, not in how a
sequence is produced, so the entry point in `scripts/` (`generate_ab.py`)
declares only what a run varies -- which arms, which levels. A batch size or truncation rule
that differed between parts would be indistinguishable from a real effect.
Its invariants are tested in `tests/test_generation_protocol.py`, on CPU — the
torch imports sit inside the functions that need them for exactly that reason.

## Part A

`generate_corpus.py` produced the 1,800-sequence corpus analysed in
`results/part_a/`: 120 donors × 5 prompt levels (0, 90, 210, 450, 900 nt of the
donor's own 5' CDS) × 3 replicates, base Evo 2, no finetuning.


Sampling parameters: temperature 0.7 explicit; `top_k`/`top_p` left at the
`evo2` library defaults and not written to the run record at the time. This was
previously described here as an unrecoverable deviation, which was wrong — the
defaults are pinned in `Evo2.generate`'s signature (`ArcInstitute/evo2`,
`evo2/models.py`): **`top_k=4`, `top_p=1.0`**. So the regime for the existing
1,800 sequences is known exactly, and Part B's runs use the same values.

`top_k=4` is Arc's default, not a value we selected. Report it as inherited and
held constant across arms, never as tuned. Mechanically it confines sampling to
the four nucleotide tokens within a 512-token character-level vocabulary, which
is why generated sequences are ~100% ACGT.

Parameters are now recorded explicitly in each run's `provenance.json`.
