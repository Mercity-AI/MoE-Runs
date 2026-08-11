"""Tiny smoke test for baseline_llama_torchtitan_longcat_ngram.

Verifies FA4 + Liger + Muon + the LongCat n-gram embedder all run end to end on
a ~10M-param model over ~10k FineWeb tokens, with no W&B and no bucket sync.
Reuses the real train() / CONFIG so we exercise the exact code path.
"""

import copy

from baseline_llama_torchtitan_longcat_ngram import CONFIG, train

SCRATCH = "/tmp/claude-0/-workspace/d6abadc0-62da-4d18-ac9a-14f0063d4d20/scratchpad"

# hidden_size=192 -> ngram sub_dim 192/6=32 (6 tables = (4-1)*2).
# 3 attn heads / 1 kv head => head_dim 64 and exercises the FA4 pack_gqa path.
OVERRIDES = {
    # --- shrink to ~10M params ---
    "hidden_size": 192,
    "num_hidden_layers": 8,
    "num_attention_heads": 3,
    "num_key_value_heads": 1,
    "intermediate_size": 512,
    "max_position_embeddings": 512,
    "max_seq_len": 512,
    # 6 tables, spread apart, all >1600 (avoids the near-vocab-multiple warning);
    # default prime multipliers (>=vocab, coprime) cover the first 6.
    "ngram_table_vocab_sizes": [2003, 3001, 4001, 5003, 6007, 7001],
    "ngram_hash_multipliers": None,
    # --- ~10k tokens: 2 * 1 * (512-1) * 10 = 10,220 ---
    "per_device_batch_size": 2,
    "grad_accum_steps": 1,
    "max_steps": 10,
    "warmup_steps": 3,
    "target_train_tokens": None,
    # --- keep it a fast, self-contained smoke test ---
    "use_wandb": False,
    "sync_checkpoints_to_bucket": False,
    "eval_benchmarks": False,
    "eval_every_steps": 1000,
    "save_every_steps": 1000,
    "eval_max_examples": 8,
    "streaming_buffer_size": 200,
    "eval_streaming_buffer_size": 1,
    "dataloader_workers": 2,
    "hf_assets_dir": f"{SCRATCH}/trial_hf_assets",
    "output_dir": f"{SCRATCH}/trial_checkpoints",
}


if __name__ == "__main__":
    config = copy.deepcopy(CONFIG)
    config.update(OVERRIDES)
    train(config)
