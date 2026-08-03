"""mdm_native_patch_v2.py — 一次性 padding 版 SDPA 补丁。

v1 的问题:每个 Block 都做 pad→bias→SDPA→split,24 层重复分配/计算。
v2:在 encoder 入口 pad 一次,24 个 Block 全程跑 (1, N, D) 批量张量,
   block 内只做 norm/qkv/SDPA(bias)/proj/MLP,末尾才 split 回变长序列。
   数学等价(块对角注意力掩码),kernel 数更少,且形状固定后可配 CUDA Graph / torch.compile。
"""

import torch
import torch.nn.functional as F

from mdm.model.dinov2_rgbd.models import vision_transformer as vt


def _sdpa_attn(self, x, attn_mask):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = torch.unbind(qkv, 0)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    x = x.permute(0, 2, 1, 3).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _run_block(blk, x, attn_bias):
    """一个 transformer block,输入/输出都是批量张量。"""
    def attn_residual(t):
        return blk.ls1(_sdpa_attn(blk.attn, blk.norm1(t), attn_bias))
    def ffn_residual(t):
        return blk.ls2(blk.mlp(blk.norm2(t)))
    x = x + attn_residual(x)
    x = x + ffn_residual(x)
    return x


_ORIG_GET_LAYERS = vt.DinoVisionTransformer._get_intermediate_layers_not_chunked


def _get_intermediate_layers_padded(self, x_img, x_depth, x_img_mask=None, x_depth_mask=None,
                                    n=1, return_mae_aux=False, **kwargs):
    """替换版:list 输入 → 一次性 pad → 全程批量张量 → 末尾 split。"""
    x_list = self.prepare_tokens_with_masks(x_img, x_depth, x_img_mask, x_depth_mask, **kwargs)

    if isinstance(x_list, list):
        B = len(x_list)
        ns = [x.shape[1] for x in x_list]
        max_n = max(ns)
        device, dtype = x_list[0].device, x_list[0].dtype
        x = torch.zeros(B, max_n, x_list[0].shape[-1], device=device, dtype=dtype)
        valid = torch.zeros(B, max_n, dtype=torch.bool, device=device)
        for i, xi in enumerate(x_list):
            ni = xi.shape[1]
            x[i, :ni] = xi[0]
            valid[i, :ni] = True
        attn_bias = torch.where(
            valid[:, None, :] & valid[:, :, None],
            torch.zeros(1, dtype=dtype, device=device),
            torch.full((1,), float("-inf"), dtype=dtype, device=device),
        )
        split_back = True
    else:
        x = x_list
        attn_bias = None
        split_back = False

    total_block_len = len(self.blocks[-1]) if self.chunked_blocks else len(self.blocks)
    blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n

    if self.chunked_blocks:
        output, i = [], 0
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:
                x = _run_block(blk, x, attn_bias)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
    else:
        output = []
        for i, blk in enumerate(self.blocks):
            x = _run_block(blk, x, attn_bias)
            if i in blocks_to_take:
                output.append(x)

    if split_back:
        output = [[out[i : i + 1, : ns[i]] for i in range(B)] for out in output]

    return output  # 与原始函数一致:直接返回,不包 tuple


def apply_native_patch_v2():
    """一次性 padding 版补丁(v2)。与 v1 互斥,调用后 v1 的 Block 补丁不再生效。"""
    if getattr(vt.DinoVisionTransformer, "_native_patched_v2", False):
        return
    vt.DinoVisionTransformer._get_intermediate_layers_not_chunked = _get_intermediate_layers_padded
    vt.DinoVisionTransformer._native_patched_v2 = True
