#!/usr/bin/env python3
"""bisect_v1v2.py — 逐 block 对比 v1 与 v2 的输入/输出,定位第一个分歧点"""
import sys
import torch
import numpy as np

sys.path.insert(0, "/home/lin/projects/lingbot-depth")
from mdm.model.v2 import MDMModel
import mdm_native_patch as mp
import mdm.model.dinov2_rgbd.models.vision_transformer as vt

rng = np.random.default_rng(0)
image = torch.tensor(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) / 255.0,
                     dtype=torch.float32, device="cuda").permute(2, 0, 1)[None]
depth_u16 = rng.integers(0, 3000, (480, 640), dtype=np.uint16)
depth_u16[rng.random((480, 640)) < 0.3] = 0
depth = torch.tensor(depth_u16.astype(np.float32) / 1000.0, dtype=torch.float32, device="cuda")[None, None]

model = MDMModel.from_pretrained("ckpt/model.pt").to("cuda")
model.eval()
bb = model.encoder.backbone

records = {"in": [], "out": []}
rec_v1 = {"in": [], "out": []}

def make_v2zero():
    def _get(self, x_img, x_depth, x_img_mask=None, x_depth_mask=None, n=1, **kw):
        x_list = self.prepare_tokens_with_masks(x_img, x_depth, x_img_mask, x_depth_mask, **kw)
        B = len(x_list); ns = [x.shape[1] for x in x_list]; max_n = max(ns)
        device, dtype = x_list[0].device, x_list[0].dtype
        x = torch.zeros(B, max_n, x_list[0].shape[-1], device=device, dtype=dtype)
        valid = torch.zeros(B, max_n, dtype=torch.bool, device=device)
        for i, xi in enumerate(x_list):
            n = xi.shape[1]; x[i, :n] = xi[0]; valid[i, :n] = True
        bias = torch.where(valid[:, None, :] & valid[:, :, None],
                           torch.zeros(1, dtype=dtype, device=device),
                           torch.full((1,), float("-inf"), dtype=dtype, device=device))
        tot = len(self.blocks)
        blocks_to_take = range(tot - n, tot) if isinstance(n, int) else n
        out = []
        for i, blk in enumerate(self.blocks):
            records["in"].append(x[0, : ns[0]].float().cpu().clone())
            x = x + blk.ls1(mp._sdpa_attn(blk.attn, blk.norm1(x), bias))
            x = x + blk.ls2(blk.mlp(blk.norm2(x)))
            records["out"].append(x[0, : ns[0]].float().cpu().clone())
            if i in blocks_to_take:
                out.append(x)
            x = x * valid[:, :, None].float()
        return [[o[i:i+1, :ns[i]] for i in range(B)] for o in out]
    return _get

def run_v1():
    # v1: 记录每个 block 的输入输出
    orig_fn = mp._nested_forward_native
    def rec_fn(self, x_list):
        ns = [x.shape[1] for x in x_list]
        rec_v1["in"].append(x_list[0][0].float().cpu().clone())
        out = orig_fn(self, x_list)
        rec_v1["out"].append(out[0][0].float().cpu().clone())
        return out
    mp._nested_forward_native = rec_fn
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model.encoder(image, depth, 37, 49, return_class_token=True)
    mp._nested_forward_native = orig_fn

mp.apply_native_patch()
run_v1()
print("v1 记录 block 数:", len(rec_v1["in"]))

# 清空,跑 v2zero
records["in"] = []; records["out"] = []
vt.DinoVisionTransformer._get_intermediate_layers_not_chunked = make_v2zero()
with torch.inference_mode():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model.encoder(image, depth, 37, 49, return_class_token=True)
print("v2zero 记录 block 数:", len(records["in"]))

for i in range(min(24, len(rec_v1["in"]), len(records["in"]))):
    d_in = (rec_v1["in"][i] - records["in"][i]).abs().max().item()
    d_out = (rec_v1["out"][i] - records["out"][i]).abs().max().item()
    flag = "  <<< 分歧!" if d_out > 1e-4 else ""
    print(f"block {i:2d}: in_diff={d_in:.2e} out_diff={d_out:.2e}{flag}")
