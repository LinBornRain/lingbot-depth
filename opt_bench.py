#!/usr/bin/env python3
"""opt_bench.py — LingBot-Depth FPS 优化候选实测
测: bf16/fp16 × tokens × torch.compile(3 模式) × GPU预处理 × 双缓冲D2H
"""
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "/home/lin/projects/lingbot-depth")
import mdm_native_patch
mdm_native_patch.apply_native_patch()

from mdm.model.v2 import MDMModel

DEVICE = "cuda"
H, W = 480, 640
CKPT = "/home/lin/projects/lingbot-depth/ckpt/model.pt"

def make_inputs(hole_ratio=0.3):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    depth = rng.integers(0, 3000, (H, W), dtype=np.uint16)
    depth[rng.random((H, W)) < hole_ratio] = 0
    K = np.array([[365.088 / W, 0, 316.962 / W],
                  [0, 365.112 / H, 242.293 / H],
                  [0, 0, 1]], dtype=np.float32)
    return image, depth, K

def prep_cpu(image, depth_m, K):
    """当前实现:CPU 归一化再传 GPU。"""
    image_t = torch.tensor(image / 255.0, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)[None]
    depth_t = torch.tensor(depth_m, dtype=torch.float32, device=DEVICE)[None]
    K_t = torch.tensor(K, dtype=torch.float32, device=DEVICE)[None]
    return image_t, depth_t, K_t

def prep_gpu(image, depth_m, K):
    """优化:uint8 直接传 GPU,归一化在 GPU 做。"""
    image_u8 = torch.from_numpy(image).permute(2, 0, 1)[None].to(DEVICE, non_blocking=True)
    image_t = image_u8.float() / 255.0
    depth_t = torch.from_numpy(depth_m).to(DEVICE, non_blocking=True)[None]
    K_t = torch.tensor(K, dtype=torch.float32, device=DEVICE)[None]
    return image_t, depth_t, K_t

def timeit(fn, iters=8, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return np.mean(ts)

def main():
    image, depth_u16, K = make_inputs()
    depth_m = depth_u16.astype(np.float32) / 1000.0
    depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"[设备] {torch.cuda.get_device_name(0)}")
    model = MDMModel.from_pretrained(CKPT).to(DEVICE)
    model.eval()

    results = []

    def run_forward(prep, dtype, num_tokens):
        image_t, depth_t, K_t = prep(image, depth_m, K)
        with torch.inference_mode():
            def fwd():
                with torch.autocast(device_type="cuda", dtype=dtype):
                    out = model.forward(image_t, num_tokens=num_tokens, depth=depth_t)
                out["depth_reg"].float()
                return out
            ms = timeit(fwd)
        return ms

    # 1) 基线 bf16
    for tk in (1200, 1800, 2400, 3600):
        ms = run_forward(prep_cpu, torch.bfloat16, tk)
        results.append((f"bf16   base  t={tk:4d}", ms))

    # 2) fp16(Blackwell fp16 速率 = 2x bf16)
    for tk in (1800, 3600):
        ms = run_forward(prep_cpu, torch.float16, tk)
        results.append((f"fp16   base  t={tk:4d}", ms))

    # 3) GPU 预处理 (uint8 直传)
    for tk in (1800,):
        ms = run_forward(prep_gpu, torch.bfloat16, tk)
        results.append((f"bf16   gpuPP t={tk:4d}", ms))

    # 4) torch.compile(固定输入形状 → static)
    image_t, depth_t, K_t = prep_cpu(image, depth_m, K)
    for mode in ("default", "max-autotune", "reduce-overhead"):
        try:
            compiled = torch.compile(model, mode=mode, dynamic=False)
            with torch.inference_mode():
                def fwd_c():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = compiled(image_t, num_tokens=1800, depth=depth_t)
                    out["depth_reg"].float()
                    return out
                fwd_c()  # 触发编译
                ms = timeit(fwd_c, iters=6, warmup=1)
            results.append((f"compile {mode:14s} t=1800", ms))
        except Exception as e:
            results.append((f"compile {mode:14s} t=1800", f"FAIL: {type(e).__name__}"))

    # 5) 双缓冲 D2H:两个流,推理与上次拷贝重叠
    try:
        s_main = torch.cuda.Stream()
        s_copy = torch.cuda.Stream()
        image_t, depth_t, K_t = prep_cpu(image, depth_m, K)
        buf = None
        with torch.inference_mode():
            def fwd_pipe():
                global buf
                with torch.cuda.stream(s_main):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model.forward(image_t, num_tokens=1800, depth=depth_t)
                    out_cuda = out["depth_reg"].float()
                # 上次的结果在这轮推理时拷贝回 CPU(重叠)
                s_copy.wait_stream(s_main)
                with torch.cuda.stream(s_copy):
                    if buf is not None:
                        buf.cpu()
                    buf = out_cuda
                torch.cuda.current_stream().wait_stream(s_copy)
                return out
            ms = timeit(fwd_pipe)
        results.append((f"bf16   pipeD2H t=1800", ms))
    except Exception as e:
        results.append((f"bf16   pipeD2H t=1800", f"FAIL: {type(e).__name__}"))

    print("\n================ 结果 ================")
    print(f"{'方案':32s} {'耗时ms':>10s} {'FPS':>8s}")
    for name, ms in results:
        if isinstance(ms, float):
            print(f"{name:32s} {ms:10.1f} {1000/ms:8.1f}")
        else:
            print(f"{name:32s} {ms}")

if __name__ == "__main__":
    main()
