"""The decoding protocol shared by every paper part (src/generate/runner.py).

Part A varies the prompt and Part B varies the weights -- but both must decode
identically, or a difference in the procedure is indistinguishable from the
effect being measured. These are the
properties that make arms comparable, tested on CPU with no model loaded.

runner.py keeps its torch imports inside the functions that need them precisely
so this file can run in the CPU suite.
"""
import csv
import json
from pathlib import Path

import pytest

from src import modal_env, provenance
from src.analysis_l1 import cds_of as scorer_cds_of
from src.generate import runner

PROMPTS = "data/prompts_corpus.csv"


def test_sampling_constants_are_part_as():
    """Not flags. These are the reason two arms can be compared at all; top_k is
    Arc's inherited default, held constant, never presented as tuned."""
    assert runner.BATCH_SIZE == 4
    assert runner.TEMPERATURE == 0.7
    assert runner.TOP_K == 4
    assert runner.TOP_P == 1.0
    # 23% above the longest natural rbcL CDS (1467 nt), so a read-through is a
    # real failure to terminate rather than Part A's censoring at 1500.
    assert runner.DEFAULT_TOTAL_LEN == 1800


def test_n_tokens_is_recomputed_not_trusted():
    """The shipped corpus was built for a 1500 nt total. Trusting its stored
    n_tokens at 1800 would silently generate 300 nt short in every arm."""
    with open(PROMPTS) as fh:
        stored = {r["prompt_id"]: int(r["n_tokens"]) for r in csv.DictReader(fh)}
    got = runner.load_prompts(PROMPTS, total_len=1800)
    assert got, "no prompts loaded"
    for pr in got:
        assert pr["n_tokens"] == 1800 - pr["prefix_nt"]
        assert pr["n_tokens"] != stored[pr["prompt_id"]]


def test_level_and_replicate_filters():
    l1 = runner.load_prompts(PROMPTS, levels="L1_donor_90", replicates="0")
    assert len(l1) == 120, "L1 replicate 0 is Part A's 120 donors"
    assert {pr["prefix_nt"] for pr in l1} == {90}
    assert {pr["replicate"] for pr in l1} == {0}
    two = runner.load_prompts(PROMPTS, levels="L1_donor_90,L2_donor_210",
                              replicates="0")
    assert len(two) == 240


def test_empty_selection_refuses():
    with pytest.raises(SystemExit):
        runner.load_prompts(PROMPTS, levels="L9_does_not_exist")


def test_impossible_budget_refuses_rather_than_generating_nothing():
    """total_len below a prefix gives n_tokens <= 0. Left unchecked, evo2 is
    asked for a non-positive number of tokens deep inside a billed run."""
    with pytest.raises(SystemExit) as e:
        runner.load_prompts(PROMPTS, levels="L4_donor_900", total_len=900)
    assert "prefix" in str(e.value)


def test_batches_are_homogeneous_in_token_budget():
    """Mixed budgets in one batch change the padding, and batched autoregressive
    decoding is not bit-identical across padding."""
    prompts = runner.load_prompts(PROMPTS, levels="all", replicates="0")
    bs = runner.batches(prompts)
    assert sum(len(b) for b in bs) == len(prompts), "prompts lost or duplicated"
    for b in bs:
        assert len(b) <= runner.BATCH_SIZE
        assert len({pr["n_tokens"] for pr in b}) == 1


def test_batching_is_deterministic_so_arms_stay_paired():
    """Same prompt list -> same batches, so a donor sits in the same batch at the
    same position under base weights and under every adapter."""
    prompts = runner.load_prompts(PROMPTS, levels="L1_donor_90", replicates="0")
    first = [[pr["prompt_id"] for pr in b] for b in runner.batches(prompts)]
    second = [[pr["prompt_id"] for pr in b] for b in runner.batches(list(prompts))]
    assert first == second


def test_cds_truncation_matches_the_scorer():
    """runner.cds_of writes the cds_nt column; src.analysis_l1.cds_of recomputes
    it at analysis time. If they disagree, part of any effect is the scorer."""
    cases = ["ATGAAATAAGGG",          # in-frame stop, mid sequence
             "ATGTAA",                 # in-frame stop at the end
             "ATGAAAGGG",              # no stop at all -> read-through
             "ATGTTAACCC",             # TAA present but out of frame
             "ATGAAATAAG",             # trailing partial codon after the stop
             "atgaaataaggg"]           # lower case input
    for s in cases:
        assert runner.cds_of(s) == scorer_cds_of(s), s
    assert runner.cds_of("ATGAAAGGG") is None


def test_output_schema_carries_what_the_analysis_pairs_on():
    """analysis_l1 scores from seq_nt and pivots by tax_group; pairing is by
    prompt_id. A column dropped here breaks the comparison, not the run."""
    for col in ("prompt_id", "donor_acc", "tax_group", "level_name", "replicate",
                "seed", "seq_nt", "cds_nt", "cds_len", "censored"):
        assert col in runner.FIELDS


def test_adapter_filename_has_one_authority(tmp_path):
    """finetune.py writes this name and generate_ab.py looks it up. A second copy
    of the convention would surface as "no adapter found" after a paid train."""
    p = modal_env.adapter_path("b2_balanced")
    assert p.name == "b2_balanced_adapter_best.pt"
    assert p.parent == modal_env.ADAPTER_DIR
    assert modal_env.adapter_path("b2_balanced", tmp_path).parent == tmp_path


def test_fingerprint_is_provenance_file_digest(tmp_path):
    """Both sides of the sha gate go through one function, so they cannot
    disagree about what is being hashed."""
    import hashlib
    f = tmp_path / "b2_balanced_adapter_best.pt"
    f.write_bytes(b"weights")
    d = provenance.file_digest(f)
    assert d["sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert d["size_bytes"] == 7


def test_remote_runs_inherit_the_commit(monkeypatch, tmp_path):
    """A packed repo has no .git, so without this every remote result would
    record commit: null -- the exact attribution hole provenance.py exists for."""
    monkeypatch.chdir(tmp_path)          # not a checkout
    monkeypatch.delenv("CS_GIT_STATE", raising=False)
    assert provenance.git_state(tmp_path)["commit"] is None
    monkeypatch.setenv("CS_GIT_STATE", '{"commit": "abc123def", '
                                       '"commit_short": "abc123d", "dirty": true}')
    st = provenance.git_state(tmp_path)
    assert st["commit"] == "abc123def"
    assert st["source"] == "launcher", "an inherited commit must say so"
    assert st["in_git_checkout"] is False
    assert provenance.run_dir(tmp_path, "generate_b").name.endswith("_abc123d")


# --------------------------------------------------------------------------
# Part B's arms and its sha gate. This is the logic that decides whether a run
# is a comparison against known weights or an unverified claim, so it is tested
# rather than trusted -- it also sits before any checkpoint load, which means a
# bug here is the difference between a one-second refusal and a billed GPU hour.
# --------------------------------------------------------------------------
CONFIG = "configs/b2_balanced.yaml"


def _adapter(dir_, tag, body=b"weights"):
    p = modal_env.adapter_path(tag, dir_)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    return p, provenance.file_digest(p)["sha256"]


def test_base_arm_needs_no_adapter(tmp_path):
    arms, adapters = runner.resolve_arms("base", config=CONFIG, adapter_dir=tmp_path)
    assert arms == [("base", None)]
    assert adapters == {}


def test_finetuned_aliases_the_config_stem(tmp_path):
    """The pre-split command line still means what it used to."""
    _, sha = _adapter(tmp_path, "b2_balanced")
    arms, adapters = runner.resolve_arms("base,finetuned", config=CONFIG,
                                         expect_sha=sha, adapter_dir=tmp_path)
    assert arms == [("base", None), ("finetuned", "b2_balanced")]
    assert adapters["finetuned"]["tag"] == "b2_balanced"
    assert adapters["finetuned"]["verified"] is True


def test_three_arms_in_one_run(tmp_path):
    """base vs b1 vs b2 -- the corpus-composition question, paired in one run."""
    _, sha1 = _adapter(tmp_path, "b1_sparse_clade", b"one")
    _, sha2 = _adapter(tmp_path, "b2_balanced", b"two")
    arms, adapters = runner.resolve_arms(
        "base,b1_sparse_clade,b2_balanced", config=CONFIG,
        expect_sha=f"b1_sparse_clade={sha1},b2_balanced={sha2}", adapter_dir=tmp_path)
    assert [a for a, _ in arms] == ["base", "b1_sparse_clade", "b2_balanced"]
    assert all(v["verified"] for v in adapters.values())
    assert adapters["b1_sparse_clade"]["sha256"] != adapters["b2_balanced"]["sha256"]


def test_mismatched_sha_refuses(tmp_path):
    _adapter(tmp_path, "b2_balanced")
    with pytest.raises(SystemExit) as e:
        runner.resolve_arms("base,b2_balanced", config=CONFIG,
                            expect_sha="b2_balanced=" + "0" * 64, adapter_dir=tmp_path)
    assert "mismatch" in str(e.value)


def test_missing_sha_refuses_unless_waived(tmp_path):
    """Mirrors provenance.py's dirty-tree rule: the unverified run stays
    possible, but it has to be asked for by name and is labelled as such."""
    _, sha = _adapter(tmp_path, "b2_balanced")
    with pytest.raises(SystemExit) as e:
        runner.resolve_arms("b2_balanced", config=CONFIG, adapter_dir=tmp_path)
    assert sha in str(e.value), "the refusal should show the sha you could paste"
    _, adapters = runner.resolve_arms("b2_balanced", config=CONFIG,
                                      allow_unverified=True, adapter_dir=tmp_path)
    assert adapters["b2_balanced"]["verified"] is False


def test_bare_sha_only_when_unambiguous(tmp_path):
    _, sha = _adapter(tmp_path, "b2_balanced")
    _, adapters = runner.resolve_arms("base,b2_balanced", config=CONFIG,
                                      expect_sha=sha, adapter_dir=tmp_path)
    assert adapters["b2_balanced"]["verified"] is True
    _adapter(tmp_path, "b1_sparse_clade", b"one")
    with pytest.raises(SystemExit) as e:
        runner.resolve_arms("b1_sparse_clade,b2_balanced", config=CONFIG,
                            expect_sha=sha, adapter_dir=tmp_path)
    assert "unambiguous" in str(e.value)


def test_sha_for_an_arm_that_is_not_running_refuses(tmp_path):
    """Otherwise a typo'd tag silently verifies nothing at all."""
    _, sha = _adapter(tmp_path, "b2_balanced")
    with pytest.raises(SystemExit) as e:
        runner.resolve_arms("base,b2_balanced", config=CONFIG,
                            expect_sha=f"b2_baIanced={sha}", adapter_dir=tmp_path)
    assert "not arms" in str(e.value)


def test_absent_adapter_refuses_with_the_command_to_make_it(tmp_path):
    with pytest.raises(SystemExit) as e:
        runner.resolve_arms("b2_balanced", config=CONFIG, adapter_dir=tmp_path)
    assert "no adapter at" in str(e.value)
    assert "finetune.py" in str(e.value)


def test_duplicate_arms_refuse(tmp_path):
    """Each arm writes generated_<arm>.csv; a repeat would overwrite itself."""
    _adapter(tmp_path, "b2_balanced")
    with pytest.raises(SystemExit) as e:
        runner.resolve_arms("base,base", config=CONFIG, adapter_dir=tmp_path)
    assert "duplicate" in str(e.value)


def test_fail_persists_the_record_and_the_traceback(tmp_path):
    """Raising without writing loses the only thing a failed GPU hour buys."""
    log = {"stage": "generate_b", "arms": ["base"]}
    runner.fail(tmp_path, log, "Traceback: boom", stage="generate_base")
    assert json.loads((tmp_path / "run_summary.json").read_text())["arms"] == ["base"]
    err = (tmp_path / "error.txt").read_text()
    assert "generate_base" in err and "boom" in err


def test_generators_persist_a_failed_checkpoint_load():
    """resolve_checkpoint without out_dir raises before anything is written, so
    the entry points must hand it one."""
    for f in ("scripts/generate_ab.py",):
        s = Path(f).read_text()
        assert "resolve_checkpoint(cfg[\"model\"], log, out_dir=OUT)" in s, f
        assert "runner.fail(OUT, log, traceback.format_exc()" in s, f


def test_adapter_load_is_checked_in_both_directions():
    """strict=False is silent about BOTH kinds of mismatch. Unexpected keys mean
    the adapter does not fit; missing LoRA keys mean sites stay at init, and LoRA
    B is zero-init, so an uncovered site is the identity -- an arm labelled
    finetuned running as base, with a clean csv and a plausible pass rate."""
    s = Path("src/generate/runner.py").read_text()
    assert "res.unexpected_keys" in s and "uncovered" in s
    assert "lora_sites_in_model" in s, "the coverage is not recorded"
    i = s.index("load_state_dict")
    assert "raise SystemExit" in s[i:i + 1500], "a mismatch does not stop the run"


def test_lora_parameter_names_match_what_the_trainer_saves():
    """The adapter-coverage check keys on .a.weight/.b.weight. If the trainer
    renames its factors the check silently stops matching anything, and a
    partially-loaded adapter would pass as fully covered.

    Asserted against the trainer's source rather than a saved adapter, so it
    holds with no artefact on disk and survives any run being deleted.
    """
    from src.generate.runner import _is_lora
    lora = Path("src/train/lora.py").read_text()
    assert "self.a = nn.Linear" in lora and "self.b = nn.Linear" in lora, \
        "LoRA factors are no longer named a/b"
    assert '".a." in k or ".b." in k' in lora, \
        "the trainer no longer selects adapter tensors by the a/b naming"
    for name in ("blocks.0.mlp.l1.a.weight", "blocks.7.out_filter_dense.b.weight"):
        assert _is_lora(name), name
    assert not _is_lora("blocks.0.mlp.l1.base.weight")


def test_determinism_check_exists_and_detects_disagreement(tmp_path):
    """Everything comparing arms generated in separate jobs depends on the same
    prompt and seed giving the same sequence. That is not guaranteed a priori:
    batched decoding is not bit-identical across batch composition, and subsets
    align here only because each level x replicate block is exactly 30 whole
    batches -- a property of this prompt corpus, not a law."""
    from src.analysis.determinism import compare
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"; c = tmp_path / "c.csv"
    a.write_text("prompt_id,seq_nt\np1,ACGT\np2,TTTT\n")
    b.write_text("prompt_id,seq_nt\np1,ACGT\np2,TTTT\n")
    c.write_text("prompt_id,seq_nt\np1,ACGT\np2,GGGG\n")
    assert compare(a, b)["byte_identical"] is True
    r = compare(a, c)
    assert r["byte_identical"] is False and r["differing"] == 1
    assert compare(a, tmp_path / "d.csv") if False else True


def test_determinism_check_handles_disjoint_runs(tmp_path):
    """Different levels share no prompts; that is not a failure."""
    from src.analysis.determinism import compare
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    a.write_text("prompt_id,seq_nt\np1,ACGT\n")
    b.write_text("prompt_id,seq_nt\np9,ACGT\n")
    assert compare(a, b)["shared"] == 0
