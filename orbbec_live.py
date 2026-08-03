#!/usr/bin/env python3
"""
orbbec_live.py — Orbbec 335L 实时 RGB-D → LingBot-Depth 优化 → 点云

管线:
  Orbbec 335L (pyorbbecsdk, depth 对齐到 color)
    → RGB + raw depth(mm) + color 内参
    → LingBot-Depth (MDM v0.5, bf16, GPU)
    → 优化 depth(米) + 有效性 mask
    → 点云(camera 系) + 可视化 + PLY 落盘

用法:
  python orbbec_live.py                        # 抓 1 帧,处理并保存到 out/
  python orbbec_live.py --frames 60 --fps 10   # 连续处理 60 帧,每秒最多 10 帧
  python orbbec_live.py --show                 # 附加实时窗口(RGB/raw/refined 对比)
  python orbbec_live.py --input-dir DIR        # 离线模式:处理已有 rgb.png + raw_depth.png(16bit mm)
  python orbbec_live.py --tokens 1800          # 降低 token 数提速(默认 3600)

输出(out/<时间戳>/):
  rgb.png              输入 RGB
  depth_raw.png        原始深度(伪彩)
  depth_refined.png    LingBot-Depth 优化深度(伪彩)
  depth_comparison.png 上下对比图
  depth_raw.npy        raw depth (float32, 米, 无效=0)
  depth_refined.npy    优化 depth (float32, 米, 不可信区=0)
  mask.npy             有效性 mask (bool)
  point_cloud.ply      RGB 点云(camera 系)
  stats.json           内参/帧率/点数等
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------- 相机采集 (OrbbecSDK ctypes 封装) ----------------

def open_camera(width=640, height=480, fps=30):
    """打开 Orbbec 相机,启用 color+depth 并做 depth→color 对齐。
    返回 (OrbbecCamera, intrinsics_dict, color_w, color_h, depth_scale)。"""
    from orbbec_ctypes import OrbbecCamera
    cam = OrbbecCamera(width=width, height=height, fps=fps, align=True)
    intrinsics = cam.intrinsics
    color_w = intrinsics["width"]
    color_h = intrinsics["height"]
    depth_scale = 1000.0  # Y16 单位 mm
    print(f"[相机] {color_w}x{color_h} color(对齐) + depth, depth_scale={depth_scale}")
    return cam, intrinsics, color_w, color_h, depth_scale


def grab_frame(cam):
    """抓一帧对齐后的 color(RGB uint8) + depth(uint16 mm)。"""
    got = cam.grab(timeout_ms=2000)
    if got is None:
        return None
    color, depth = got
    return color, depth


# ---------------- 模型 ----------------

def load_model(model_id, device):
    import mdm_native_patch_v2  # 一次性 padding 版 SDPA 补丁(与原始语义数值一致,+3% 速度)
    mdm_native_patch_v2.apply_native_patch_v2()

    from mdm.model.v2 import MDMModel
    t0 = time.time()
    print(f"[模型] 加载 {model_id} ...")
    model = MDMModel.from_pretrained(model_id).to(device)
    model.eval()
    print(f"[模型] 加载完成 {time.time()-t0:.1f}s, 参数 {sum(p.numel() for p in model.parameters())/1e6:.0f}M")
    return model


@torch.inference_mode()
def refine_depth(model, image_rgb, depth_m, intrinsics, device, num_tokens):
    """image_rgb: (H,W,3) uint8; depth_m: (H,W) float32 米,无效=0; intrinsics: dict。
    返回 depth_refined (H,W) float32(不可信=0), mask (H,W) bool, points (H,W,3)(不可信=inf)。"""
    h, w = image_rgb.shape[:2]
    image_t = torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)[None]
    depth_t = torch.tensor(depth_m, dtype=torch.float32, device=device)[None]

    K = np.array([
        [intrinsics["fx"] / w, 0.0, intrinsics["cx"] / w],
        [0.0, intrinsics["fy"] / h, intrinsics["cy"] / h],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    K_t = torch.tensor(K, dtype=torch.float32, device=device)[None]

    output = model.infer(
        image_t,
        depth_in=depth_t,
        intrinsics=K_t,
        num_tokens=num_tokens,
        apply_mask=True,
        use_fp16=True,
    )
    depth_ref = output["depth"].squeeze(0).float().cpu().numpy()   # inf = 不可信
    mask = output["mask"].squeeze(0).cpu().numpy() if output.get("mask") is not None else None
    points = output["points"].squeeze(0).float().cpu().numpy() if output.get("points") is not None else None

    if mask is None:
        mask = np.isfinite(depth_ref)
    depth_clean = np.where(np.isfinite(depth_ref), depth_ref, 0.0).astype(np.float32)
    return depth_clean, mask, points


# ---------------- 可视化 / 落盘 ----------------

def depth_color(depth_m, vmax=None):
    valid = np.isfinite(depth_m) & (depth_m > 0)
    d = depth_m.copy()
    d[~valid] = 0
    vmin = 0.0
    if vmax is None:
        vmax = d[valid].max() if valid.any() else 1.0
    norm = np.clip((d - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def make_pointcloud(rgb, depth_m, mask, intrinsics, downsample=1):
    """camera 系点云 + RGB 颜色。跳过不可信像素。"""
    h, w = depth_m.shape
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    valid = mask & np.isfinite(depth_m) & (depth_m > 0)
    v, u = np.mgrid[0:h, 0:w]
    z = depth_m
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1)          # (H, W, 3)
    colors = rgb[..., ::-1]                     # RGB -> BGR (PLY 按 BGR 存 trimesh 直接给 RGB 也行,统一用 RGB)
    colors = rgb
    if downsample > 1:
        pts = pts[::downsample, ::downsample]
        colors = colors[::downsample, ::downsample]
        valid = valid[::downsample, ::downsample]
    pts_v = pts[valid]
    col_v = colors[valid]
    return pts_v, col_v


def save_results(out_dir, rgb, depth_raw_m, depth_ref, mask, points, intrinsics, meta):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "depth_raw.png"), depth_color(depth_raw_m))
    cv2.imwrite(str(out_dir / "depth_refined.png"), depth_color(depth_ref))

    raw_c = depth_color(depth_raw_m)
    ref_c = depth_color(depth_ref)
    comp = np.concatenate([raw_c, ref_c], axis=0)
    cv2.putText(comp, "raw", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(comp, "lingbot-depth refined", (8, raw_c.shape[0] + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(out_dir / "depth_comparison.png"), comp)

    np.save(out_dir / "depth_raw.npy", depth_raw_m)
    np.save(out_dir / "depth_refined.npy", depth_ref)
    np.save(out_dir / "mask.npy", mask)

    pts, cols = make_pointcloud(rgb, depth_ref, mask, intrinsics, downsample=2)
    import trimesh
    pc = trimesh.PointCloud(pts, colors=cols)
    pc.export(str(out_dir / "point_cloud.ply"))
    n_raw = int((depth_raw_m > 0).sum())
    n_ref = int(mask.sum())
    meta.update({"valid_px_raw": n_raw, "valid_px_refined": n_ref, "ply_points": int(len(pts))})
    with open(out_dir / "stats.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[保存] {out_dir}/")
    print(f"       有效像素: raw {n_raw} → refined {n_ref} (+{100*(n_ref-n_raw)/max(n_raw,1):.0f}%)")
    print(f"       点云: point_cloud.ply ({len(pts):,} 点)")


# ---------------- 离线模式 ----------------

def run_offline(model, device, input_dir, out_dir, num_tokens):
    input_dir = Path(input_dir)
    rgb_path = None
    for ext in [".png", ".jpg", ".jpeg"]:
        cand = input_dir / f"rgb{ext}"
        if cand.exists():
            rgb_path = cand
            break
    if rgb_path is None:
        raise FileNotFoundError(f"输入目录缺少 rgb.png/jpg: {input_dir}")
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(input_dir / "raw_depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    scale = 1000.0 if depth.max() > 100 else 1.0
    depth_m = depth / scale
    depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    h, w = rgb.shape[:2]
    intr_file = input_dir / "intrinsics.txt"
    if intr_file.exists():
        K = np.loadtxt(intr_file, dtype=np.float32)
        intrinsics = {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                      "cx": float(K[0, 2]), "cy": float(K[1, 2]), "width": w, "height": h}
    else:
        intrinsics = {"fx": 460.139587, "fy": 460.199005, "cx": 319.656128, "cy": 237.396271,
                      "width": w, "height": h}
        print("[离线] 未找到 intrinsics.txt,使用 example 0 的内参")
    t0 = time.time()
    depth_ref, mask, points = refine_depth(model, rgb, depth_m, intrinsics, device, num_tokens)
    print(f"[推理] {time.time()-t0:.2f}s")
    meta = {"mode": "offline", "input": str(input_dir), "intrinsics": intrinsics,
            "num_tokens": num_tokens, "infer_sec": round(time.time() - t0, 3)}
    save_results(out_dir, rgb, depth_m, depth_ref, mask, points, intrinsics, meta)


# ---------------- 在线模式 ----------------

def backproject(depth_m, valid, rgb, intrinsics):
    """把对齐后的深度图反投影成相机系点云 (H,W,3) + RGB 颜色 (H,W,3)。"""
    h, w = depth_m.shape
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    v, u = np.mgrid[0:h, 0:w]
    z = np.where(valid, depth_m, 0.0).astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1).astype(np.float32)
    return pts, rgb


def render_pointcloud(points, colors, yaw=0.0, pitch=0.0, f=400.0, out_w=640, out_h=480):
    """用 numpy 把点云从虚拟视角渲染成 2D 图(z-buffer)。
    points: (N,3) 相机系米;colors: (N,3) RGB uint8。
    返回 BGR uint8 (out_h, out_w, 3)。"""
    import math
    img = np.full((out_h, out_w, 3), 24, dtype=np.uint8)
    pts = np.asarray(points).reshape(-1, 3)
    cols = np.asarray(colors).reshape(-1, 3)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx_, sx_ = math.cos(pitch), math.sin(pitch)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # yaw: 绕 Y 轴; pitch: 绕 X 轴
    x1, z1 = x * cy + z * sy, -x * sy + z * cy
    y2, z2 = y * cx_ - z1 * sx_, y * sx_ + z1 * cx_
    zz = z2
    valid = zz > 0.05
    u = np.round(f * x1[valid] / zz[valid] + out_w / 2).astype(int)
    v = np.round(f * y2[valid] / zz[valid] + out_h / 2).astype(int)
    depth_v = zz[valid]
    col_v = cols[valid]
    inside = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
    u, v, depth_v, col_v = u[inside], v[inside], depth_v[inside], col_v[inside]
    # 深度排序(远→近),近的覆盖远的
    order = np.argsort(-depth_v)
    u, v, col_v = u[order], v[order], col_v[order]
    img[v, u] = col_v[:, ::-1]  # RGB -> BGR
    return img


def run_live(model, device, args):
    import threading
    cam, intrinsics, w, h, depth_scale = open_camera()
    out_root = Path(args.out)
    meta = {"mode": "live", "intrinsics": intrinsics, "num_tokens": args.tokens}

    # 可选的 Open3D 交互点云窗口
    o3d_vis = o3d_pcd = None
    if args.o3d:
        try:
            import open3d as o3d
            o3d_vis = o3d.visualization.Visualizer()
            o3d_vis.create_window(window_name="LingBot-Depth Point Cloud (mouse orbit)", width=640, height=480)
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_vis.add_geometry(o3d_pcd)
            print("[预览] Open3D 点云窗口已打开(鼠标拖拽旋转/滚轮缩放)")
        except Exception as e:
            print(f"[预览] Open3D 窗口不可用({e}),仅显示 2D 投影点云")
            o3d_vis = o3d_pcd = None

    # 跨线程共享状态
    state = {
        "latest": None,          # (color_rgb, depth_m) 最新相机帧
        "refined": None,         # (depth_ref, mask, points, color, depth_m) 最近推理结果
        "infer_seq": 0,          # 已触发推理的相机帧序号
        "infer_done": 0,         # 已完成的推理结果对应序号
        "infer_running": False,
        "infer_ms": 0.0,
        "pause": False,
        "stop": False,
        "yaw": 0.0, "pitch": 0.35,
    }
    lock = threading.Lock()
    cache = {"ref_seq": -1, "ref_c": None, "cloud_ref": None, "fill_show": None}  # refined 面板缓存

    def refine_worker():
        """后台推理线程:有新帧就跑一帧推理,结果发布到 state['refined']。"""
        while True:
            with lock:
                if state["stop"]:
                    return
                if state["pause"] or state["infer_running"]:
                    time.sleep(0.02)
                    continue
                latest = state["latest"]
                want_seq = state["infer_seq"]
                if latest is None or want_seq == state["infer_done"]:
                    time.sleep(0.02)
                    continue
                color, depth_m = latest
                state["infer_running"] = True
            t0 = time.time()
            depth_ref, mask, points = refine_depth(model, color, depth_m, intrinsics, device, args.tokens)
            dt = (time.time() - t0) * 1000
            with lock:
                state["refined"] = (depth_ref, mask, points, color, depth_m)
                state["infer_done"] = want_seq
                state["infer_ms"] = dt
                state["infer_running"] = False
            print(f"[推理 #{want_seq}] {dt:.0f}ms | 覆盖 {100*mask.mean():.0f}%", flush=True)

    worker = threading.Thread(target=refine_worker, daemon=True)
    worker.start()

    frame_id = 0
    show = args.show
    if show:
        cv2.namedWindow("lingbot-depth: RGB | RAW | REFINED | CLOUD", cv2.WINDOW_NORMAL)
    try:
        while True:
            grabbed = grab_frame(cam)
            if grabbed is None:
                print("[相机] 超时无帧,重试...")
                continue
            color, depth_u16 = grabbed
            frame_id += 1
            depth_m = depth_u16.astype(np.float32) / depth_scale
            depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)

            with lock:
                state["latest"] = (color.copy(), depth_m.copy())
                state["infer_seq"] = frame_id

            if show:
                # 顶行 30fps: RGB / RAW / REFINED;底行: ORBBEC 点云 / LINGBOT 点云 / 补全区域
                with lock:
                    refined = state["refined"]
                    infer_ms = state["infer_ms"]
                    paused = state["pause"]
                    yaw, pitch = state["yaw"], state["pitch"]
                raw_c = depth_color(depth_m)
                p1 = color[..., ::-1].copy()
                # Orbbec 原始(对齐)点云 —— 15fps 足够(CPU 渲染,避免抢推理线程)
                if frame_id % 2 == 1:
                    raw_valid = (depth_m > 0.0) & (depth_m < 65.0)
                    pts_raw, col_raw = backproject(depth_m, raw_valid, color, intrinsics)
                    ys, xs = np.where(raw_valid)
                    if len(ys) > 20000:  # 自适应采样 ~20k 个有效点
                        sel = np.linspace(0, len(ys) - 1, 20000).astype(int)
                        ys, xs = ys[sel], xs[sel]
                    cloud_raw = render_pointcloud(pts_raw[ys, xs], col_raw[ys, xs], yaw=yaw, pitch=pitch)
                    last_cloud_raw = cloud_raw
                else:
                    cloud_raw = last_cloud_raw if "last_cloud_raw" in dir() else np.full_like(raw_c, 16)
                if refined is not None:
                    # 只有新推理结果时才重算 refined 相关面板(伪彩/点云/FILLED)
                    if cache["ref_seq"] != state["infer_done"]:
                        depth_ref, mask, points, rcolor, rdepth = refined
                        ref_c = depth_color(depth_ref)
                        # LingBot 点云(同一视角)
                        pts_ref = np.asarray(points).reshape(-1, 3)
                        col_ref = np.asarray(rcolor).reshape(-1, 3)
                        step2 = max(1, len(pts_ref) // 20000)
                        cloud_ref = render_pointcloud(pts_ref[::step2], col_ref[::step2], yaw=yaw, pitch=pitch)
                        # 补全区域:raw 无效但 refined 有效(绿色高亮)
                        filled = (depth_m <= 0) & (depth_ref > 0)
                        fill_show = np.zeros_like(ref_c)
                        fill_show[filled] = ref_c[filled]
                        cache["ref_c"], cache["cloud_ref"], cache["fill_show"] = ref_c, cloud_ref, fill_show
                        cache["ref_seq"] = state["infer_done"]
                        # Open3D 交互窗口:只在有新推理结果时更新几何(避免 GL 与 CUDA 抢 GPU)
                        if o3d_vis is not None:
                            pts3 = np.asarray(points).reshape(-1, 3)
                            col3 = np.asarray(rcolor).reshape(-1, 3) / 255.0
                            keep = np.isfinite(pts3).all(axis=1) & (pts3[:, 2] > 0)
                            sel = np.linspace(0, keep.sum() - 1, min(keep.sum(), 30000)).astype(int) if keep.any() else slice(None)
                            idx = np.where(keep)[0][sel] if keep.any() else np.where(keep)[0]
                            o3d_pcd.points = o3d.utility.Vector3dVector(pts3[idx])
                            o3d_pcd.colors = o3d.utility.Vector3dVector(col3[idx])
                            o3d_vis.update_geometry(o3d_pcd)
                    ref_c = cache["ref_c"]
                    cloud_ref = cache["cloud_ref"]
                    fill_show = cache["fill_show"]
                else:
                    ref_c = np.full_like(raw_c, 16)
                    cloud_ref = np.full_like(raw_c, 16)
                    fill_show = np.full_like(raw_c, 16)
                cv2.putText(p1, "RGB", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(raw_c, "RAW DEPTH", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(ref_c, f"LINGBOT-DEPTH {infer_ms:.0f}ms" + (" [P]" if paused else ""),
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(cloud_raw, "CLOUD: ORBBEC RAW (aligned)", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(cloud_ref, "CLOUD: LINGBOT-DEPTH", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(fill_show, "FILLED BY LINGBOT (green)", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                top = np.concatenate([p1, raw_c, ref_c], axis=1)
                bot = np.concatenate([cloud_raw, cloud_ref, fill_show], axis=1)
                disp = np.concatenate([top, bot], axis=0)
                if args.shot and refined is not None:
                    out_root.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_root / "preview_latest.png"), disp)
                cv2.imshow("lingbot-depth: RGB|RAW|REFINED / CLOUD-ORBBEC|CLOUD-LINGBOT|FILLED", disp)
                if o3d_vis is not None:
                    o3d_vis.poll_events()
                    if frame_id % 2 == 1:  # 15fps 重绘,减轻 GL/CUDA 争抢
                        o3d_vis.update_renderer()
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("p"):
                    with lock:
                        state["pause"] = not state["pause"]
                elif key == 81:  # 左
                    with lock:
                        state["yaw"] -= 0.1
                elif key == 83:  # 右
                    with lock:
                        state["yaw"] += 0.1
                elif key == 82:  # 上
                    with lock:
                        state["pitch"] = min(1.2, state["pitch"] + 0.1)
                elif key == 84:  # 下
                    with lock:
                        state["pitch"] = max(-1.2, state["pitch"] - 0.1)
                elif key == ord("s"):
                    with lock:
                        refined_save = state["refined"]
                    if refined_save is not None:
                        dref, msk, pts, rcol, rdep = refined_save
                        save_results(out_root / time.strftime("%Y%m%d_%H%M%S"), rcol, rdep,
                                     dref, msk, pts, intrinsics, meta)

            if args.save_every and frame_id % args.save_every == 0:
                with lock:
                    refined_save = state["refined"]
                if refined_save is not None:
                    dref, msk, pts, rcol, rdep = refined_save
                    meta.update({"frame": frame_id})
                    save_results(out_root / time.strftime("%Y%m%d_%H%M%S"), rcol, rdep,
                                 dref, msk, pts, intrinsics, meta)

            if args.frames and frame_id >= args.frames:
                break
    except KeyboardInterrupt:
        print("\n[相机] 手动中断")
    finally:
        with lock:
            state["stop"] = True
        worker.join(timeout=2.0)
        if o3d_vis is not None:
            try:
                o3d_vis.destroy_window()
            except Exception:
                pass
        if show:
            cv2.destroyAllWindows()
        try:
            cam.close()
        except Exception:
            pass
    print(f"[统计] 共采集 {frame_id} 帧")


def main():
    ap = argparse.ArgumentParser(description="Orbbec 335L + LingBot-Depth 实时深度优化与点云")
    ap.add_argument("--model", default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5",
                    help="HF 模型名或本地 model.pt 路径")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokens", type=int, default=1800, nargs="?", const=1800,
                    help="ViT token 数(默认/裸 --tokens = 1800 ≈ 26fps;3600 最高质量 ≈ 13fps;900 ≈ 34fps)")
    ap.add_argument("--frames", type=int, default=0, help="处理帧数(0=无限)")
    ap.add_argument("--fps", type=float, default=0, help="限帧率(0=不限)")
    ap.add_argument("--out", default="out", help="输出根目录")
    ap.add_argument("--show", action="store_true", help="实时四联预览窗口(RGB|RAW|REFINED|CLOUD)")
    ap.add_argument("--o3d", action="store_true", help="配合 --show:额外开 Open3D 交互点云窗口(鼠标轨道)")
    ap.add_argument("--shot", action="store_true", help="配合 --show:每次推理后把四联图存到 out/preview_latest.png")
    ap.add_argument("--save-every", type=int, default=0, help="每隔 N 帧保存一次(0=不自动保存,预览按 s 保存)")
    ap.add_argument("--input-dir", default=None, help="离线模式:处理该目录下 rgb.png + raw_depth.png")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")

    if args.o3d:
        args.show = True

    model = load_model(args.model, device)
    if args.tokens is None:
        args.tokens = getattr(model, "num_tokens_range", [1200, 3600])[1]
    print(f"[配置] num_tokens={args.tokens}")

    if args.input_dir:
        run_offline(model, device, args.input_dir, args.out, args.tokens)
    else:
        run_live(model, device, args)


if __name__ == "__main__":
    main()
