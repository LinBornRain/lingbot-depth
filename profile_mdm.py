#!/usr/bin/env python3
"""profile_mdm.py — LingBot-Depth 推理瓶颈剖析
用法: python profile_mdm.py [tokens...]  默认 [1200, 2400, 3600]
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

def make_inputs():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    depth = rng.integers(0, 3000, (H, W), dtype=np.uint16)
    depth[rng.random((H, W)) < 0.3] = 0  # 30% 空洞(贴近真实场景)
    K = np.array([[365.088 / W, 0, 316.962 / W],
                  [0, 365.112 / H, 242.293 / H],
                  [0, 0, 1]], dtype=np.float32)
    return image, depth, K

def to_gpu(image, depth_m, K):
    image_t = torch.tensor(image / 255.0, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)[None]
    depth_t = torch.tensor(depth_m, dtype=torch.float32, device=DEVICE)[None]
    K_t = torch.tensor(K, dtype=torch.float32, device=DEVICE)[None]
    return image_t, depth_t, K_t

def bench(model, image, depth_m, K, num_tokens, iters=8):
    image_t, depth_t, K_t = to_gpu(image, depth_m, K)
    with torch.inference_mode():
        model.infer(image_t, depth_in=depth_t, intrinsics=K_t, num_tokens=num_tokens)  # 预热

    t = {"tensor_prep": [], "forward": [], "tail_d2h": [], "total": []}
    with torch.inference_mode():
        for _ in range(iters):
            t0 = time.perf_counter()
            image_t, depth_t, K_t = to_gpu(image, depth_m, K)
            t1 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model.forward(image_t, num_tokens=num_tokens, depth=depth_t)
            t2 = time.perf_counter()
            depth = out["depth_reg"].float()
            points = out["mask"].float().cpu() if out.get("mask") is not None else None
            depth.cpu()
            t3 = time.perf_counter()
            t["tensor_prep"].append(t1 - t0)
            t["forward"].append(t2 - t1)
            t["tail_d2h"].append(t3 - t2)
            t["total"].append(t3 - t0)
    return {k: np.mean(v) * 1000 for k, v in t.items()}

def profiler_breakdown(model, image, depth_m, K, num_tokens):
    """torch profiler 归属 forward 时间: encoder(注意力/MLP) vs neck/head(卷积)。"""
    image_t, depth_t, K_t = to_gpu(image, depth_m, K)
    with torch.inference_mode():
        model.infer(image_t, depth_in=depth_t, intrinsics=K_t, num_tokens=num_tokens)
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model.forward(image_t, num_tokens=num_tokens, depth=depth_t)
    ev = prof.key_averages()
    total_cuda = sum(e.self_device_time_total for e in ev) / 1000.0  # us -> ms
    print(f"\n--- torch profiler (5 次 forward 均值, num_tokens={num_tokens}) ---")
    print(f"forward CUDA 总时间: {total_cuda:.1f} ms")
    cats = {}
    for e in ev:
        name = e.key
        if "aten::" not in name and "cuda::" not in name:
            continue
        if any(k in name for k in ("_scaled_dot_product", "efficient_attention", "bmm", "matmul", "addmm", "linear", "sdp")):
            c = "attention/qkv/linear"
        elif any(k in name for k in ("convolution", "conv2d", "conv")):
            c = "conv(neck/heads)"
        elif any(k in name for k in ("layer_norm", "norm")):
            c = "norm"
        elif any(k in name for k in ("gelu", "relu", "silu", "activation")):
            c = "activation"
        elif any(k in name for k in ("copy", "to", "clone")):
            c = "copy/trans"
        elif any(k in name for k in ("softmax", "reshape", "permute", "view", "cat", "interpolate", "unflatten")):
            c = "reshape/interp"
        else:
            c = "other"
        cats[c] = cats.get(c, 0) + e.self_device_time_total / 1000.0
    for c, ms in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c:20s}: {ms:7.1f} ms ({100*ms/total_cuda:.0f}%)")

def main():
    tokens_list = [int(x) for x in sys.argv[1:]] or [1200, 2400, 3600]
    image, depth_u16, K = make_inputs()
    depth_m = depth_u16.astype(np.float32) / 1000.0
    depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"[设备] {torch.cuda.get_device_name(0)}")
    model = MDMModel.from_pretrained("/home/lin/projects/lingbot-depth/ckpt/model.pt").to(DEVICE)
    model.eval()
    for tk in tokens_list:
        t = bench(model, image, depth_m, K, tk)
        print(f"\n=== num_tokens={tk} ===")
        for k, v in t.items():
            print(f"  {k:12s}: {v:6.1f} ms")
        print(f"  -> 推理 FPS: {1000/t['total']:.1f}")
    profiler_breakdown(model, image, depth_m, K, tokens_list[-1])

if __name__ == "__main__":
    main()
