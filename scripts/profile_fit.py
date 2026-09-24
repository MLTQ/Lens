import os, sys, time, torch
sys.path.insert(0, ".")
import jlens
from transformers import PreTrainedTokenizerFast
from bonsai_lens.fit import jacobian_rows_for_ids
from bonsai_lens.ternary import load_bonsai
if os.environ.get("TF32"): torch.backends.cuda.matmul.allow_tf32 = True
model, _, _ = load_bonsai("models/bonsai-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf", "models/bonsai-meta/config.json", verbose=False)
tok = PreTrainedTokenizerFast(tokenizer_file="models/bonsai-meta/tokenizer.json")
lm = jlens.from_hf(model, tok, force_bos=False)
ids = tok(open("vendor-jacobian-lens/README.md").read()[:4000], return_tensors="pt").input_ids[:, :128].cuda()
src = list(range(63)); d = 5120
acc = {l: torch.zeros(d, d) for l in src}
jacobian_rows_for_ids(lm, ids, src, 63, acc, dim_batch=32, max_dims=32)  # warmup
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    torch.cuda.synchronize(); t0 = time.time()
    jacobian_rows_for_ids(lm, ids, src, 63, acc, dim_batch=32, max_dims=64)
    torch.cuda.synchronize(); print(f"2 passes: {time.time()-t0:.1f}s")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=60))
