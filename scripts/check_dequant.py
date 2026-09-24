import sys, torch; sys.path.insert(0, ".")
import bonsai_lens.ternary as T
g = torch.Generator().manual_seed(0)
packed = torch.randint(0, 255, (300, 5120 // 4), dtype=torch.uint8, generator=g).cuda()
packed = packed & 0b10101010 | ((packed & 0b01010101) & ~(packed >> 1) & 0b01010101)  # avoid code 3
scales = (torch.rand(300, 5120 // 128, generator=g) * 0.02).half().cuda()
a = T.dequantize(packed, scales)
T.HAVE_TRITON = False
b = T.dequantize(packed, scales)
print("triton vs torch max abs diff:", (a.float() - b.float()).abs().max().item(), a.dtype, a.shape)
