# Run ledger

Append-only. One row per run; `results/<arm>/<utc>_<sha>/`
holds that run's provenance.json, config, metrics, and error.txt.

| utc | arm | commit | dirty | job_id | outcome | note |
|---|---|---|---|---|---|---|
| 2026-08-21T11:10:04Z | b2_balanced | a317084 | False | 754077e9-28c3-4214-aa69-11cbab7485e2 | ok | ORF-clean 80.7->100%; 52% of ft output >90% kmers copied |
| 2026-08-24T08:36:26.254838+00:00 | b2_balanced | 8ed0059 | False | sb-G8Gp3bZ1LKWk7QIVsXaetv | failed (-1) | train 1 config(s) |
| 2026-08-24T08:38:30.969459+00:00 | b2_balanced | 787bd87 | False | sb-Xon1ZOOWOjSy5u8Xrw9cOp | ok | train 1 config(s) |
| 2026-08-24T09:55:40.357726+00:00 | b2_balanced | 2c499d4 | False | sb-qTikJf6njAQf901b7EMuje | ok | train 1 config(s) |
| 2026-08-24T11:02:13.957345+00:00 | generate_b | ad4dc61 | False | sb-o1V1PRJWwJH3YRzXTl7QhM | ok | part B: base,b2_balanced, 240 sequences |
| 2026-08-24T21:40:05.229393+00:00 | all_fullcds+all_fullcds_atg+b1_sparse_clade | 3e7946c | False | sb-3ol6ujcJ4ujVkCaMCUBcKR | failed (124) | train 3 config(s) |

### Gap: 2026-08-24 training run has no row

The three-adapter training run (`all_fullcds`, `all_fullcds_atg`,
`b1_sparse_clade`) is absent from the table above. Its sandbox was
`sb-3ol6ujcJ4ujVkCaMCUBcKR`, commit `3e7946c`, clean tree, and it succeeded --
all three adapters and all 27 record files are on the weights volume and
committed under `results/all_fullcds+all_fullcds_atg+b1_sparse_clade/`.

This is the research run behind the three published adapters, and it is complete:
each trained to `train_exit=0` over three epochs with a monotonically decreasing
`val_novel`. The sandbox exited only after that work was done and its records
were already on the weights volume.

No row was written because the launcher appends to this ledger in a `finally`
block that runs *after* record recovery, and recovery raised. Nothing was lost:
every field this table would carry is already in each run's `provenance.json`
(`compute_job_id`, commit, dirty flag, input sha256).

Deliberately **not** back-filled by hand. This table's value is that it is
append-only and machine-written; one typed row makes every other row a question.
The ordering fix is recorded in `docs/RUN_TRACKING.md`.
| 2026-08-25T09:21:19.661810+00:00 | generate_a | 766d53f | False | sb-MrsOwW8TfdgcPZm5nGtOXX | ok | part A: base, 120 sequences |
| 2026-08-25T09:33:16.129047+00:00 | generate_a | 9166fa5 | False | sb-GzOwuiEHjeW9pV8hb1VNVy | ok | part A: base, 600 sequences |
| 2026-08-25T10:51:45.375183+00:00 | generate_b | 999d76a | False | sb-8A1vZ0tlRUHPZkTU74Nu56 | ok | part B: base,all_fullcds,all_fullcds_atg,b1_sparse_clade, 480 sequences |
| 2026-08-25T13:48:51.833505+00:00 | generate_b | 9a75f78 | False | sb-HVI49ZeEvwEwFwN6jxxse0 | failed (124) | part B: base,all_fullcds,all_fullcds_atg,b1_sparse_clade, 480 sequences |
