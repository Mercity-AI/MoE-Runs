"""Teacher-forced loss and token accuracy on the answer tokens of the full validation set.

    python analysis/val_loss.py <adapter_dir> <base_checkpoint> [--out per_sentence.pt]

Same data pipeline as training (ner_data.build_examples), so the result is
comparable to the eval_loss / eval_accuracy the Trainer logs. Useful when a run
did not log them, and for paired comparisons: --out saves per-sentence losses.
"""
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ner_data  # noqa: E402

run, base = sys.argv[1], sys.argv[2]
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
tok = AutoTokenizer.from_pretrained(run)
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, trust_remote_code=True,
                                         attn_implementation="sdpa")
m = PeftModel.from_pretrained(m, run).merge_and_unload().cuda().eval()
ds, _ = ner_data.build_examples(tok, dataset_key="few-nerd", split="validation", max_seq_len=1024)
coll = DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding="longest", return_tensors="pt")
tot_loss, tot_tok, tot_ok, per_sent = 0.0, 0, 0, []
with torch.no_grad():
    for i in range(0, len(ds), 32):
        b = {k: v.cuda() for k, v in coll([ds[j] for j in range(i, min(i + 32, len(ds)))]).items()}
        logits = m(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.float()
        lab = b["labels"][:, 1:]
        lg = logits[:, :-1]
        mask = lab != -100
        nll = torch.nn.functional.cross_entropy(lg.transpose(1, 2), lab.clamp(min=0), reduction="none")
        tot_loss += (nll * mask).sum().item()
        tot_tok += mask.sum().item()
        tot_ok += ((lg.argmax(-1) == lab) & mask).sum().item()
        per_sent += ((nll * mask).sum(1) / mask.sum(1)).tolist()
print(f"{run}: target-token loss={tot_loss / tot_tok:.4f}  token acc={tot_ok / tot_tok:.4f}  "
      f"(n_sent={len(per_sent)}, n_tok={tot_tok})")
if out:
    torch.save(torch.tensor(per_sent), out)
