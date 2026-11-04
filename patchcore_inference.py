#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PatchCore 推理脚本 —— anomalib 2.7.0

和训练脚本配对使用：读 patchcore_train.py 产出的 run_info.json，
按同样的超参重建模型，再从 checkpoint 恢复 memory bank 做推理。

三件容易踩的事，这里都处理了：
  1) 2.7 里 resize/归一化在模型自带的 PreProcessor 里，数据集只负责读图；
  2) engine.predict() 返回的是 "按 batch 组织" 的 Batch 对象，
     pred_score / anomaly_map 都还是带 batch 维的 Tensor，要摊平才能逐张用；
  3) Windows 下 cv2.imread/imwrite 遇到中文路径会静默返回 None，
     这里统一走 np.fromfile + cv2.imdecode / cv2.imencode + tofile。

用法
    python patchcore_inference.py --run-info runs/demo/run_info.json \
        --source datasets/demo/defect --out runs/demo/pred

    # 只跑单张图
    python patchcore_inference.py --run-info runs/demo/run_info.json --source test.png

    # 手动绕过 run_info 指定 ckpt
    python patchcore_inference.py --run-info runs/demo/run_info.json --ckpt path/to/model.ckpt --source imgs/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

try:
    import anomalib
    from anomalib.data import PredictDataset
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
except ImportError as exc:  # pragma: no cover
    sys.exit(f"[错误] 导入 anomalib 失败，请先 pip install \"anomalib[cu130]\"。原始报错：{exc}")

try:
    import cv2  # anomalib 依赖 opencv-python-headless，正常都会有
except ImportError:  # pragma: no cover
    cv2 = None


# --------------------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PatchCore 推理（anomalib 2.7.0）")
    p.add_argument("--run-info", default="runs/patchcore_demo/run_info.json",
                   help="训练脚本产出的 run_info.json")
    p.add_argument("--ckpt", default=None, help="显式指定 checkpoint，覆盖 run_info 里的路径")
    p.add_argument("--source", required=True, help="待推理的图片或目录")
    p.add_argument("--out", default="runs/patchcore_demo/pred", help="结果输出目录")
    p.add_argument("--threshold", type=float, default=None,
                   help="自定义判定阈值；不给就用 checkpoint 里后处理器的自适应阈值（pred_label）")
    p.add_argument("--accelerator", default="auto", choices=["auto", "gpu", "cpu"])
    p.add_argument("--no-vis", action="store_true", help="不生成热力图叠图，只出 CSV")
    p.add_argument("--save-maps", action="store_true", help="额外把原始 anomaly map 存成 .npy")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# 把 engine.predict 的返回值摊平成逐张图的记录
# --------------------------------------------------------------------------------------
def _as_list(value: Any) -> list[Any]:
    """Tensor / ndarray / list / 标量 / str 统一成 python list。"""
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (str, Path)):
        return [value]
    return [value]


def _as_map(maps: Any, index: int) -> np.ndarray | None:
    """从 batch 维度里取出第 index 张图的 anomaly map，统一成 HxW 的 float32。"""
    if maps is None:
        return None
    if isinstance(maps, (list, tuple)):
        return _as_map(maps[index], 0) if index < len(maps) else None
    if isinstance(maps, torch.Tensor):
        m = maps.detach().float().cpu()
    elif isinstance(maps, np.ndarray):
        m = maps.astype(np.float32)
    else:
        return None
    if m.ndim == 4:  # (B, 1, H, W)
        m = m[min(index, m.shape[0] - 1)]
    if m.ndim == 3:  # (1, H, W)
        m = m[0]
    return np.asarray(m, dtype=np.float32)


def iter_predictions(predictions: Any) -> Iterator[dict[str, Any]]:
    """anomalib 2.x 的 predict 返回 list[Batch]，一个 Batch 是一整批；
    这里按 batch 内的顺序拆成单图记录。"""
    if not predictions:
        return
    if not isinstance(predictions, (list, tuple)):  # 兜底：个别版本只返回单个 Batch
        predictions = [predictions]
    for batch in predictions:
        paths = _as_list(getattr(batch, "image_path", None))
        scores = _as_list(getattr(batch, "pred_score", None))
        labels = _as_list(getattr(batch, "pred_label", None))
        maps = getattr(batch, "anomaly_map", None)
        n = max(len(paths), len(scores), len(labels), 1)
        for i in range(n):
            yield {
                "image_path": str(paths[i]) if i < len(paths) else "",
                "pred_score": float(scores[i]) if i < len(scores) else float("nan"),
                "pred_label": int(labels[i]) if i < len(labels) else -1,
                "anomaly_map": _as_map(maps, i),
            }


# --------------------------------------------------------------------------------------
# 可视化（Unicode 路径安全）
# --------------------------------------------------------------------------------------
def read_image(path: str | Path) -> np.ndarray | None:
    if cv2 is None:
        return None
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def write_image(path: str | Path, image: np.ndarray) -> bool:
    if cv2 is None:
        return False
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def save_overlay(image_path: str | Path, anomaly_map: np.ndarray | None, out_path: Path) -> bool:
    """min-max 归一化 -> JET 伪彩 -> 与原图 55:45 混合。"""
    if cv2 is None or anomaly_map is None or anomaly_map.size == 0:
        return False
    bgr = read_image(image_path)
    if bgr is None:
        return False
    h, w = bgr.shape[:2]

    amap = anomaly_map.astype(np.float32)
    lo, hi = float(amap.min()), float(amap.max())
    amap = (amap - lo) / (hi - lo + 1e-8)
    amap = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap((amap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis = cv2.addWeighted(bgr, 0.55, heat, 0.45, 0.0)
    return write_image(out_path, vis)


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    info_path = Path(args.run_info).expanduser().resolve()
    if not info_path.exists():
        sys.exit(f"[错误] 找不到 {info_path}，先跑 patchcore_train.py 生成它")
    info = json.loads(info_path.read_text(encoding="utf-8"))

    ckpt = Path(args.ckpt).expanduser().resolve() if args.ckpt else info.get("ckpt_path")
    if not ckpt or not Path(ckpt).exists():
        sys.exit(f"[错误] checkpoint 不存在：{ckpt}（用 --ckpt 指定）")

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        sys.exit(f"[错误] --source 不存在：{source}")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "overlays"
    if not args.no_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = out_dir / "maps"
    if args.save_maps:
        maps_dir.mkdir(parents=True, exist_ok=True)

    # 1) 用与训练完全一致的超参重建模型骨架，权重/memory bank 从 ckpt 恢复
    m = info["model"]
    pre_processor = Patchcore.configure_pre_processor(image_size=tuple(info.get("image_size", (256, 256))))
    model = Patchcore(
        backbone=m["backbone"],
        layers=list(m["layers"]),
        coreset_sampling_ratio=m.get("coreset_sampling_ratio", 0.1),
        num_neighbors=m.get("num_neighbors", 9),
        pre_processor=pre_processor,
    )
    print(f"[环境] anomalib {getattr(anomalib, '__version__', 'unknown')} | torch {torch.__version__}")
    print(f"[模型] {m['backbone']} @ {m['layers']} | k={m.get('num_neighbors', 9)} | "
          f"image_size={info.get('image_size', [256, 256])}")
    print(f"[权重] {ckpt}")

    # 2) 数据：PredictDataset 支持单张图或目录
    try:
        dataset = PredictDataset(path=str(source), image_size=tuple(info.get("image_size", (256, 256))))
    except TypeError:
        # 个别小版本去掉了 dataset 层的 image_size（改由模型 PreProcessor 决定），这里兜一下
        dataset = PredictDataset(path=str(source))

    # engine = Engine(default_root_dir=str(out_dir), accelerator=args.accelerator, logger=False)
    # predictions = engine.predict(
    #     model=model,
    #     dataset=dataset,
    #     ckpt_path=str(ckpt),
    #     return_predictions=True,  # 不写盘、直接拿 Tensor，后面自己出 CSV 和叠图
    # )
    # 1. 强制 CPU 加载权重
    ckpt_dict = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt_dict["state_dict"])

    # 2. 推理时不再传 ckpt_path，避免 Lightning 重新按 CUDA 恢复
    engine = Engine(default_root_dir=str(out_dir), accelerator="cpu", logger=False)
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        return_predictions=True,
    )

    # 3) 逐张落结果
    rows: list[dict[str, Any]] = []
    for rec in iter_predictions(predictions):
        score, label = rec["pred_score"], rec["pred_label"]
        if args.threshold is not None:
            # 注意：这里比较的是模型输出的 pred_score，其量纲由后处理器的归一化决定
            # （anomalib 2.x 默认会归一化到 0~1）。换数据集重新标定阈值，别直接搬。
            label = int(score >= args.threshold)

        src = Path(rec["image_path"]) if rec["image_path"] else None

        amap = rec["anomaly_map"]
        if args.save_maps and amap is not None and src is not None:
            np.save(maps_dir / f"{src.stem}.npy", amap)

        overlay_path = ""
        if src is not None and not args.no_vis:
            dst = vis_dir / f"{src.stem}_pred.png"
            if save_overlay(src, amap, dst):
                overlay_path = str(dst)

        rows.append({
            "image_path": rec["image_path"],
            "pred_score": round(score, 6),
            "pred_label": label,
            "anomaly_map_max": round(float(amap.max()), 6) if isinstance(amap, np.ndarray) else "",
            "overlay_path": overlay_path,
        })

    if not rows:
        print("[警告] 没有拿到任何预测结果，检查 --source 下是否有图片")
        return 1

    csv_path = out_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 4) 控制台摘要
    scores = np.array([r["pred_score"] for r in rows], dtype=np.float64)
    n_ng = sum(1 for r in rows if r["pred_label"] == 1)
    print(f"\n[结果] {len(rows)} 张 | 判为异常 {n_ng} 张 | "
          f"score min/mean/max = {scores.min():.4f} / {scores.mean():.4f} / {scores.max():.4f}")
    print("[Top-5 最可疑]")
    for r in sorted(rows, key=lambda x: x["pred_score"], reverse=True)[:5]:
        flag = "NG" if r["pred_label"] == 1 else "OK"
        print(f"  {r['pred_score']:.4f}  {flag}  {Path(r['image_path']).name}")
    print(f"\nCSV      : {csv_path}")
    if not args.no_vis:
        print(f"overlays : {vis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
