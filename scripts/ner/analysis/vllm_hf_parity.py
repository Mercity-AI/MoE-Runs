"""Check that vLLM runs a checkpoint the same way HuggingFace does.

Run this on any new checkpoint (or after touching ``vllm_ngram_model.py``)
before trusting vLLM eval numbers::

    python analysis/vllm_hf_parity.py --run ../ner_runs/ngram_longcat_25_json
    python analysis/vllm_hf_parity.py --run ../qknorm_baseline_json/checkpoint-2198 \\
        --base ../checkpoints/baseline/step_003053_hf

What it does, on N validation sentences:
  1. Prints loading diagnostics (architecture, n-gram hash settings, BOS handling).
  2. Teacher-forced check: feeds BOS + prompt + gold answer to both engines and
     compares the next-token prediction on every answer token. A correct port
     agrees on ~100% of tokens; small log-prob differences (~0.01) are normal
     bf16 kernel noise.
  3. Generation check: greedy-decodes the same prompts in both engines (vLLM
     with CUDA graphs on) and counts identical outputs. A few divergences are
     normal; many mean the decode path is broken.

Exits with status 1 if token agreement is below --threshold.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import vllm_env  # noqa: E402,F401  — must precede vllm

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import ner_data  # noqa: E402
import vllm_ngram_model  # noqa: E402,F401  — registers custom architectures
from ner_eval_vllm import merge_adapter  # noqa: E402


def resolve_base(run: Path, base: Path | None) -> Path | None:
    if not (run / "adapter_config.json").is_file():
        return None
    if base is not None:
        return base
    cfg = yaml.safe_load((run / "ner_config.yaml").read_text())
    return Path(cfg["checkpoint"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="LoRA adapter dir or full model dir")
    ap.add_argument("--base", type=Path, default=None, help="base checkpoint (default: from ner_config.yaml)")
    ap.add_argument("--n", type=int, default=16, help="number of validation sentences")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    ap.add_argument("--threshold", type=float, default=0.99, help="min teacher-forced top-1 agreement")
    args = ap.parse_args()

    base = resolve_base(args.run, args.base)
    tmp = None
    if base is not None:
        tmp = tempfile.mkdtemp(prefix="parity_merged_")
        print(f"merging adapter {args.run} into {base}")
        merge_adapter(args.run, base, Path(tmp))
        model_dir = Path(tmp)
    else:
        model_dir = args.run

    # --- 1. diagnostics ---------------------------------------------------
    cfg = json.loads((model_dir / "config.json").read_text())
    tok = AutoTokenizer.from_pretrained(model_dir)
    auto_bos = tok("hello", add_special_tokens=True)["input_ids"][:1] == [tok.bos_token_id]
    print("\n== Diagnostics")
    print(f"  architecture          : {cfg['architectures'][0]}")
    print(f"  layers / hidden       : {cfg['num_hidden_layers']} / {cfg['hidden_size']}")
    print(f"  rope                  : {cfg.get('rope_parameters') or cfg.get('rope_theta')}")
    print(f"  tie_word_embeddings   : {cfg.get('tie_word_embeddings')}")
    if cfg["architectures"][0] == "LlamaLongCatNgram":
        mult = cfg.get("ngram_hash_multipliers")
        print(f"  n-gram max_n / heads  : {cfg['ngram_max_n']} / {cfg['ngram_num_heads']}")
        print(f"  n-gram hash           : {'multipliers ' + str(mult) if mult else 'base = vocab_size (older checkpoint)'}")
    print(f"  tokenizer adds BOS    : {auto_bos}  (training always prepends BOS explicitly; the eval scripts do too)")

    raw, schema, col = ner_data.load_split("few-nerd", "validation", args.n, output_format="json")
    bos = [tok.bos_token_id]
    prompts, full, starts = [], [], []
    for ex in raw:
        p = bos + tok(schema.build_prompt(ex["tokens"]), add_special_tokens=False)["input_ids"]
        t = tok(schema.target_text(ex["tokens"], ex[col]), add_special_tokens=False)["input_ids"]
        prompts.append(p)
        full.append(p + t + [tok.eos_token_id])
        starts.append(len(p))

    # --- 2a. HuggingFace pass ---------------------------------------------
    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, trust_remote_code=True,
                                              attn_implementation="sdpa").cuda().eval()
    hf_lp, hf_top, hf_gen = [], [], []
    with torch.no_grad():
        for s, p in zip(full, prompts):
            lp = hf(input_ids=torch.tensor([s], device="cuda")).logits.float().log_softmax(-1)[0]
            hf_lp.append([lp[i - 1, s[i]].item() for i in range(1, len(s))])
            hf_top.append(lp.argmax(-1).tolist())
            out = hf.generate(input_ids=torch.tensor([p], device="cuda"), max_new_tokens=args.max_new_tokens,
                              do_sample=False, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
            hf_gen.append(tok.decode(out[0, len(p):], skip_special_tokens=True).strip())
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    # --- 2b. vLLM pass ----------------------------------------------------
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    is_ngram = cfg["architectures"][0] == "LlamaLongCatNgram"
    llm = LLM(model=str(model_dir), trust_remote_code=True, dtype="bfloat16", max_model_len=2048,
              max_num_seqs=1 if is_ngram else 64, enable_prefix_caching=False,
              gpu_memory_utilization=args.gpu_memory_utilization)
    tf = llm.generate([TokensPrompt(prompt_token_ids=s) for s in full],
                      SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1))
    gen = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                       SamplingParams(max_tokens=args.max_new_tokens, temperature=0))

    # --- 3. compare -------------------------------------------------------
    diffs, agree, n_tok, nll_hf, nll_v = [], 0, 0, [], []
    for s, st, lph, toph, o in zip(full, starts, hf_lp, hf_top, tf):
        for i in range(st, len(s)):  # answer tokens only
            d = o.prompt_logprobs[i]
            v_lp = d[s[i]].logprob
            v_top = min(d.items(), key=lambda kv: kv[1].rank)[0]
            diffs.append(abs(v_lp - lph[i - 1]))
            agree += v_top == toph[i - 1]
            n_tok += 1
            nll_hf.append(-lph[i - 1])
            nll_v.append(-v_lp)
    same_gen = sum(h == g.outputs[0].text.strip() for h, g in zip(hf_gen, gen))
    rate = agree / n_tok

    print("\n== Teacher-forced check (answer tokens only)")
    print(f"  tokens compared       : {n_tok}")
    print(f"  top-1 agreement       : {rate:.2%}")
    print(f"  answer loss HF / vLLM : {np.mean(nll_hf):.4f} / {np.mean(nll_v):.4f}")
    print(f"  |logprob diff| mean/max: {np.mean(diffs):.4f} / {np.max(diffs):.4f}")
    print("\n== Generation check (greedy, vLLM with CUDA graphs)")
    print(f"  identical outputs     : {same_gen}/{len(prompts)}")
    for h, g in zip(hf_gen, gen):
        if h != g.outputs[0].text.strip():
            print(f"    HF  : {h[:150]}\n    vLLM: {g.outputs[0].text.strip()[:150]}")
            break

    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = rate >= args.threshold
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} (threshold {args.threshold:.0%})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
