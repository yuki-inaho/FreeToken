"""Dense rows-GEMV: how far from memory roofline, and does any tiling close the gap?

Every candidate goes through the production _gemv_rows via _ROWS_TUNE, so the comparison
includes the real epilogue/fixup paths. Weights rotate over a pool >= 4x the 96 MB L2 so each
launch streams from HBM (reusing one tensor measures L2 and reports >150% of peak).
"""
import torch, triton, itertools, sys
sys.path.insert(0, "/home/nuko/ft/python")
import freetoken.kernel.triton.fp8_pertensor_linear as F

PEAK, M, dev, L2 = 1.79e12, 4, "cuda", 96 << 20
SHAPES = [("gdn.in_proj_qkvz", 16384, 2560), ("gdn.out_proj", 2560, 6144),
          ("qsa.qkv_proj", 13312, 2560), ("hc.mix", 2560, 2560),
          ("shared.gate_up", 1280, 2560), ("shared.down", 2560, 640),
          ("lm_head", 8192, 2560)]

def timed(call, n, iters=8):
    for i in range(n): call(i)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n): call(i)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters): g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / (iters * n) * 1e-3

print(f"{'shape':20s} {'MB':>5s} | {'prod us':>8s} {'GB/s':>6s} {'%pk':>4s} | {'best us':>8s} {'GB/s':>6s} "
      f"{'%pk':>4s}  {'config':22s} gain (ok/tried)")
results = {}
for name, N, K in SHAPES:
    ncop = max(4, int(4 * L2 / (N * K)) + 1)
    pool = [torch.randint(1, 120, (N, K), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)  # 0x7F/0xFF are NaN in e4m3
            for _ in range(ncop)]
    ws = torch.rand(N, device=dev, dtype=torch.float32) * 0.01
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    F._ROWS_TUNE = {}; F._ROWS_WARPS = 1
    run = lambda i: F._gemv_rows(a, pool[i], ws, torch.bfloat16)
    t_prod = timed(run, ncop)
    ref = run(0).float()
    best, ok, tried = (t_prod, "production(BN16 BKauto skauto w1)"), 0, 0
    for BN, BK, sk, nw in itertools.product((8, 16, 32, 64), (64, 128, 256), (1, 2, 4, 8, 16), (1, 2, 4)):
        if N % BN or sk > triton.cdiv(K, BK): continue
        tried += 1
        F._ROWS_TUNE = {"BLOCK_N": BN, "BLOCK_K": BK, "SPLIT_K": sk}; F._ROWS_WARPS = nw
        try:
            out = run(0); torch.cuda.synchronize()
            if not torch.allclose(out.float(), ref, rtol=3e-2, atol=3e-2): continue
            ok += 1
            t = timed(run, ncop)
        except Exception:
            continue
        if t < best[0]: best = (t, f"BN{BN} BK{BK} sk{sk} w{nw}")
    F._ROWS_TUNE = {}; F._ROWS_WARPS = 1
    gp, gb = N * K / t_prod / 1e9, N * K / best[0] / 1e9
    results[name] = (t_prod, best)
    print(f"{name:20s} {N*K/1e6:5.1f} | {t_prod*1e6:8.2f} {gp:6.0f} {gp*1e9/PEAK*100:4.0f} | {best[0]*1e6:8.2f} "
          f"{gb:6.0f} {gb*1e9/PEAK*100:4.0f}  {best[1]:22s} {t_prod/best[0]:.2f}x ({ok}/{tried})", flush=True)
