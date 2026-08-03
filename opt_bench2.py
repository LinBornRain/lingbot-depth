#!/usr/bin/env python3
"""opt_bench2.py — 补测: GPU利用率 / 双缓冲D2H / 360p输入 / fp16 tail"""
import sys
import time
import threading
import subprocess
import numpy as np
import torch

sys.path.insert(0, "/home/lin/projects/lingbot-depth")
import mdm_native_patch
mdm_native_patch.apply_native_patch()

from mdm.model.v2 import MDMModel

DEVICE = "cuda"
CKPT = "/home/lin/projects/lingbot-depth/ckpt/model.pt"

def make_inputs(H, W, hole_ratio=0.3):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    depth = rng.integers(0, 3000, (H, W), dtype=np.uint16)
    depth[rng.random((H, W)) < hole_ratio] = 0
    K = np.array([[365.088 / W, 0, 316.962 / W],
                  [0, 365.112 / H, 242.293 / H],
                  [0, 0, 1]], dtype=np.float32)
    return image, depth, K

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
    print(f"[设备] {torch.cuda.get_device_name(0)}")
    model = MDMModel.from_pretrained(CKPT).to(DEVICE)
    model.eval()

    # ---------- 1) GPU 利用率(纯推理循环) ----------
    image, depth_u16, K = make_inputs(480, 640)
    depth_m = depth_u16.astype(np.float32) / 1000.0
    image_t = torch.tensor(image / 255.0, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)[None]
    depth_t = torch.tensor(depth_m, dtype=torch.float32, device=DEVICE)[None]
    K_t = torch.tensor(K, dtype=torch.float32, device=DEVICE)[None]
    stop = threading.Event()
    utils = []
    def sampler():
        while not stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=3)
                utils.append(int(out.stdout.strip()))
            except Exception:
                pass
            time.sleep(0.2)
    th = threading.Thread(target=sampler); th.start()
    with torch.inference_mode():
        for _ in range(5):  # 预热
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model.forward(image_t, num_tokens=1800, depth=depth_t)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model.forward(image_t, num_tokens=1800, depth=depth_t)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 30 * 1000
    stop.set(); th.join()
    print(f"\n[1] 纯 forward @1800: {ms:.1f}ms | GPU 利用率: {np.mean(utils):.0f}% (n={len(utils)})")

    # ---------- 2) 双缓冲 D2H(修复版) ----------
    s_main = torch.cuda.Stream()
    s_copy = torch.cuda.Stream()
    holder = {"buf": None}
    with torch.inference_mode():
        def fwd_pipe():
            with torch.cuda.stream(s_main):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model.forward(image_t, num_tokens=1800, depth=depth_t)
                out_cuda = out["depth_reg"].float()
                if holder["buf"] is not None:  # 上一轮的拷贝与当前推理重叠
                    holder["buf"].cpu()
                holder["buf"] = out_cuda
                return out
        ms = timeit(fwd_pipe)
    print(f"[2] 双缓冲 D2H @1800: {ms:.1f}ms (vs 基线 {71.1:.1f}ms)")

    # ---------- 3) 360p 输入(尾段更便宜) ----------
    image3, depth3_u16, K3 = make_inputs(360, 640)
    depth3_m = depth3_u16.astype(np.float32) / 1000.0
    i3 = torch.tensor(image3 / 255.0, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)[None]
    d3 = torch.tensor(depth3_m, dtype=torch.float32, device=DEVICE)[None]
    K3t = torch.tensor(K3, dtype=torch.float32, device=DEVICE)[None]
    with torch.inference_mode():
        def fwd_360():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model.forward(i3, num_tokens=1800, depth=d3)
            out["depth_reg"].float().cpu()
            return out
        ms = timeit(fwd_360)
    print(f"[3] 640x360 输入 @1800(含D2H): {ms:.1f}ms")

    # ---------- 4) fp16 尾段(输出 fp16 减拷贝) ----------
    with torch.inference_mode():
        def fwd_f16tail():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model.forward(image_t, num_tokens=1800, depth=depth_t)
            out["depth_reg"].float().cpu()
            return out
        ms = timeit(fwd_f16tail)
    print(f"[4] 基线 + D2H @1800: {ms:.1f}ms")

if __name__ == "__main__":
    main()
