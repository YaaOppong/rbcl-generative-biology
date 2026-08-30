
import csv
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc
from collections import defaultdict

import torch
from evo2 import Evo2

os.makedirs("out", exist_ok=True)

# METHOD NOTE — reproducibility invariants. Every sequence in every replicate
# must be produced by an identical procedure, otherwise a nuisance parameter
# becomes confounded with the titration variable (prefix length).
#   * BATCH_SIZE is UNIFORM across all prefix levels and all replicates.
#     Batched autoregressive decoding is not bit-identical across batch
#     sizes (padding + reduction order + FFT prefill perturb logits, and
#     sampling amplifies that), so varying it per level would confound
#     "more donor context" with "different numerical path".
#     4 is the largest value that fits evo2_7b on a 40GB A100 without OOM.
#   * Uniform TOTAL sequence length (prefix + generated = TOTAL_LEN) for
#     every level, as encoded in prompts.csv (n_tokens = TOTAL_LEN - prefix_nt).
#     Total length is the invariant to hold, not new-token count: the
#     deliverable is a full-length rbcL CDS, and holding new-token count
#     constant instead would make high-prefix levels systematically longer.
#   * Generation runs to TOTAL_LEN and the CDS is recovered downstream by
#     truncating at the first in-frame stop codon. This is lossless: decoding
#     is autoregressive, so read-through past the stop cannot influence the
#     CDS that precedes it.
BATCH_SIZE = 4
TOTAL_LEN = 1500
TEMPERATURE = 0.7   # uniform across all levels and replicates

# Read prompts.csv with stdlib csv (no pandas)
prompts = []
with open("prompts_corpus.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['n_tokens'] = int(row['n_tokens'])  # = TOTAL_LEN - prefix_nt (see METHOD NOTE)
        row['prefix_nt'] = int(row['prefix_nt'])
        row['replicate'] = int(row['replicate'])
        row['seed'] = int(row['seed'])
        row['temperature'] = TEMPERATURE
        prompts.append(row)
print(f"Loaded {len(prompts)} prompts")

model = Evo2("evo2_7b")
print("Model loaded, GPU:", torch.cuda.get_device_name())

groups = defaultdict(list)
for p in prompts:
    groups[p['n_tokens']].append(p)

fieldnames = ["prompt_id","donor_acc","organism","tax_group","cluster","split",
              "level_name","prefix_nt","n_tokens","replicate","seed","seq_nt","seq_len"]
out_path = "out/generated.csv"
# open file once, write incrementally (append per batch) so a timeout preserves partial progress
f_out = open(out_path, "w", newline="")
writer = csv.DictWriter(f_out, fieldnames=fieldnames)
writer.writeheader()
f_out.flush()

start_time = time.time()
generated_count = 0
total = len(prompts)

for n_tokens in sorted(groups.keys(), reverse=True):
    group = groups[n_tokens]
    batch_size = BATCH_SIZE  # UNIFORM across all levels (see METHOD NOTE)
    print(f"\nn_tokens={n_tokens}, batch_size={batch_size}, n_prompts={len(group)}", flush=True)

    for bstart in range(0, len(group), batch_size):
        batch = group[bstart:bstart+batch_size]
        prefixes = [row['prompt'] for row in batch]

        # Deterministic, reproducible sampling: seed derived from the
        # identity of the first prompt in the batch (donor/level/replicate),
        # so re-running reproduces the same draws exactly.
        seed = batch[0]['seed']   # precomputed in prompts_corpus.csv from donor|level|replicate
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        t0 = time.time()
        out = model.generate(
            prompt_seqs=prefixes,
            n_tokens=n_tokens,
            temperature=batch[0]['temperature'],
        )
        elapsed = time.time() - t0

        for row, seq in zip(batch, out.sequences):
            full_seq = row['prompt'] + seq
            writer.writerow({
                "prompt_id": row['prompt_id'],
                "donor_acc": row['donor_acc'],
                "organism": row['organism'],
                "tax_group": row['tax_group'],
                "cluster": row['cluster'],
                "split": row['split'],
                "level_name": row['level_name'],
                "prefix_nt": row['prefix_nt'],
                "n_tokens": n_tokens,
                "replicate": row['replicate'],
                "seed": row['seed'],
                "seq_nt": full_seq,
                "seq_len": len(full_seq),
            })
            generated_count += 1
        f_out.flush()
        os.fsync(f_out.fileno())
        del out
        gc.collect()
        torch.cuda.empty_cache()

        rate = len(batch)/elapsed
        eta_min = (total-generated_count)/max(rate,0.01)/60
        print(f"  batch: {len(batch)} seqs in {elapsed:.1f}s ({rate:.2f} seq/s) | "
              f"{generated_count}/{total} done, ETA {eta_min:.1f}min", flush=True)

f_out.close()
print(f"\nDONE: {generated_count} sequences in {(time.time()-start_time)/3600:.2f}h")
