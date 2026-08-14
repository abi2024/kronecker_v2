"""Every module imports, and its public names exist.

Cheap insurance against a stale stub sitting where a real file was meant to
land: the rest of the suite only exercises what it happens to import, so a
module nothing tests can stay broken until an experiment fails hours later.
"""

import importlib

MODULES = {
    "kronecker_v2.vocab":            ["load_tokenizer", "vocab_bytes", "truncate", "script_of"],
    "kronecker_v2.collisions":       ["exact_collisions", "audit", "truncation_stats"],
    "kronecker_v2.codecs.base":      ["Codec", "OneHotCodec"],
    "kronecker_v2.codecs.wave":      ["WaveKroneckerCodec"],
    "kronecker_v2.codecs.baselines": ["build_matched", "matched_rank", "matched_buckets",
                                      "budget_report", "codec_budget"],
    "kronecker_v2.embedding":        ["CodecEmbedding"],
    "kronecker_v2.model":            ["GPT", "GPTConfig"],
    "kronecker_v2.tables":           ["build_onehot_table", "build_wave_table"],
    "kronecker_v2.runlog":           ["write_manifest", "append_run_row"],
    "kronecker_v2.eval.bpb":         ["bits_per_byte", "vocab_byte_lengths", "LN2"],
}


def test_every_module_imports_with_its_public_names():
    missing = []
    for mod, names in MODULES.items():
        m = importlib.import_module(mod)
        missing += [f"{mod}.{n}" for n in names if not hasattr(m, n)]
    assert not missing, f"stale or incomplete modules: {missing}"
