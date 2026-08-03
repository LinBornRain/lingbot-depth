#!/usr/bin/env python3
"""opt_bench3.py — CUDA Graph 决定性测试(固定输入,同语义)
若 graph 大幅提速 → 启动开销是瓶颈,值得做 full-grid 改造;
若无提升 → 纯算力/带宽瓶颈,只有降 token 一条路。
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
CKPT = "/home/lin/projects/lingbot-depth/ckpt/model.pt"

def make_inputs(hole_ratio=0.3):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = rng.integers(0, 3000, (480, 640), dtype=np.uint16)
    depth[rng.random((480, 640)) < hole_ratio] = 0
    K = np.array([[365.088 / 640, 0, 316.962 / 640],
                  [0, 365.112 / 480, 242.293 / 480],
                  [0, 0, 1]], dtype=np.float32)
    return image, depth, K

def timeit(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return np.mean(ts), np.min(ts)

def main():
    print(f"[设备] {torch.cuda.get_device_name(0)}")
    model = MDMModel.from_pretrained(CKPT).to(DEVICE)
    model.eval()

    for hole in (0.3, 0.7):
        image, depth_u16, K = make_inputs(hole)
        depth_m = depth_u16.astype(np.float32) / 1000.0
        image_t = torch.tensor(image / 255.0, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)[None]
        depth_t = torch.tensor(depth_m, dtype=torch.float32, device=DEVICE)[None]
        K_t = torch.tensor(K, dtype=torch.float32, device=DEVICE)[None]

        def fwd():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model.forward(image_t, num_tokens=1800, depth=depth_t)
            return out["depth_reg"].float()

        # 基线
        with torch.inference_mode():
            ms_avg, ms_min = timeit(fwd)
            print(f"\n[空洞{int(hole*100)}%] 基线 forward@1800: avg={ms_avg:.1f}ms min={ms_min:.1f}ms")

            # CUDA Graph(固定输入 → 每帧形状相同)
            try:
                torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN attention 无法图捕获
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                g = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        fwd()
                torch.cuda.current_stream().wait_stream(s)
                with torch.cuda.graph(g):
                    out_g = fwd()
                # 预热 replay
                for _ in range(3):
                    g.replay()
                torch.cuda.synchronize()
                ts = []
                for _ in range(10):
                    t0 = time.perf_counter()
                    g.replay()
                    torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) * 1000)
                print(f"[空洞{int(hole*100)}%] CUDA Graph @1800: avg={np.mean(ts):.1f}ms min={np.min(ts):.1f}ms")
                # 数值一致性检查
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_ref = model.forward(image_t, num_tokens=1800, depth=depth_t)
                diff = (out_ref["depth_reg"].float() - out_g.float()).abs().max().item()
                print(f"  数值一致性: max|Δdepth| = {diff:.2e}")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"CUDA Graph FAIL: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
