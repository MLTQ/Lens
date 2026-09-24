import sys, time, numpy as np, mlx.core as mx
sys.path.insert(0, ".")
from bonsai_lens.mlx_backend import BonsaiMLX
t0 = time.time(); m = BonsaiMLX("models/bonsai-mlx"); print(f"load {time.time()-t0:.0f}s")
ref = np.load("runs/ref.npz")
ids = m.encode("Fact: The currency used in the country shaped like a boot is")
print("ids match:", ids == ref["ids"][0].tolist(), ids)
t0 = time.time(); hs = m.residuals(ids); print(f"forward {time.time()-t0:.2f}s", hs.shape)
for l in (0, 10, 31, 62, 63):
    a = np.array(hs[l].astype(mx.float32)); b = ref[f"h{l}"]
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    print(f"L{l}: rel err {np.linalg.norm(a - b) / np.linalg.norm(b):.3e}  min cos {cos.min():.4f}  |h| {np.linalg.norm(b, axis=-1).mean():.1f}")
lg = np.array(m.readout(hs[63][-1:], 63, "logit"))[0]
print("mlx top5:", [m.decode_token(i) for i in np.argsort(-lg)[:5]])
print("logit corr:", np.corrcoef(lg, ref["logits_last"])[0, 1])
