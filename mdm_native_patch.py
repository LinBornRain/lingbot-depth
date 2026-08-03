"""mdm_native_patch.py — 用 PyTorch 原生 SDPA 替换 xformers 依赖。

LingBot-Depth 的 depth masking 产生变长 token 序列(NestedTensorBlock 嵌套路径),
原实现硬依赖 xformers 的 BlockDiagonalMask + memory_efficient_attention。
这里把嵌套路径重写为:padding 到最长序列 + block-diagonal 注意力掩码 + F.scaled_dot_product_attention,
数学等价,零 xformers 依赖。推理模式下的 block 结构(残差/norm/ls)完全保留。
"""

import torch
import torch.nn.functional as F

from mdm.model.dinov2_rgbd.layers.block import NestedTensorBlock


def _sdpa_attn(self, x, attn_mask):
    """qkv -> 多头 -> SDPA,attn_mask 为 additive bias (B, N, N)。"""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = torch.unbind(qkv, 0)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    x = x.permute(0, 2, 1, 3).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _nested_forward_native(self, x_list):
    """等价于原 forward_nested(非训练随机深度路径):padding + 块对角 mask。"""
    B = len(x_list)
    ns = [x.shape[1] for x in x_list]
    max_n = max(ns)
    device, dtype = x_list[0].device, x_list[0].dtype

    x = torch.zeros(B, max_n, x_list[0].shape[-1], device=device, dtype=dtype)
    valid = torch.zeros(B, max_n, dtype=torch.bool, device=device)
    for i, xi in enumerate(x_list):
        n = xi.shape[1]
        x[i, :n] = xi[0]  # xi: (1, n, d)
        valid[i, :n] = True

    # block-diagonal additive bias: 同一样本内可见 token 可互相关注,其余 -inf
    attn_bias = torch.where(
        valid[:, None, :] & valid[:, :, None],
        torch.zeros(1, dtype=dtype, device=device),
        torch.full((1,), float("-inf"), dtype=dtype, device=device),
    )  # (B, N, N)

    def attn_residual_func(t):
        return self.ls1(_sdpa_attn(self.attn, self.norm1(t), attn_bias))

    def ffn_residual_func(t):
        return self.ls2(self.mlp(self.norm2(t)))

    x = x + attn_residual_func(x)
    x = x + ffn_residual_func(x)
    return [x[i : i + 1, :n] for i, n in enumerate(ns)]  # 保持 (1, n, d) 原格式


_ORIG_FORWARD = NestedTensorBlock.forward


def _forward(self, x_or_x_list):
    if isinstance(x_or_x_list, torch.Tensor):
        return _ORIG_FORWARD(self, x_or_x_list)  # 普通批量路径,原样
    elif isinstance(x_or_x_list, list):
        return _nested_forward_native(self, x_or_x_list)
    else:
        raise AssertionError


def apply_native_patch():
    """给 NestedTensorBlock 打补丁,之后不再需要 xformers。可重复调用。"""
    if getattr(NestedTensorBlock, "_native_patched", False):
        return
    NestedTensorBlock.forward = _forward
    NestedTensorBlock._native_patched = True
