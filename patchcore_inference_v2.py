#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PatchCore 推理 v2 —— 加载 ckpt -> 推理 -> 打印结果（anomalib 2.7.0）

没有 json、没有 argparse、没有可视化。核心就三步，方便直接搬进自己的工程测速：
    build_model() -> PredictDataset(SOURCE) -> engine.predict(ckpt_path)
第 0 轮加载 ckpt，之后的轮次沿用已加载权重计时（纯净的推理速度）。
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

# ===== 与 patchcore_train_v2.py 保持一致 =====
CKPT = Path("runs/v2/patchcore_v2.ckpt")
SOURCE = Path("datasets/demo/defect")   # 单张图片或一个目录

BACKBONE = "wide_resnet50_2"
LAYERS = ["layer2", "layer3"]
NUM_NEIGHBORS = 9
IMAGE_SIZE = (256, 256)

WARMUP = 1    # 预热轮数：第 0 轮要加载 ckpt，测速建议 >=1 把它排除在外
ROUNDS = 3    # 正式计时轮数，取平均
# ============================================


def build_model() -> Patchcore:
    """骨架必须和训练端一致；memory bank 等权重全部来自 ckpt。"""
    return Patchcore(
        backbone=BACKBONE,
        layers=LAYERS,
        num_neighbors=NUM_NEIGHBORS,
        pre_processor=Patchcore.configure_pre_processor(image_size=IMAGE_SIZE),
    )

def build_model_cpu() -> Patchcore:
    """骨架必须和训练端一致；memory bank 等权重全部来自 ckpt。"""
    model = Patchcore(
            backbone=BACKBONE,
            layers=LAYERS,
            num_neighbors=NUM_NEIGHBORS,
            pre_processor=Patchcore.configure_pre_processor(image_size=IMAGE_SIZE),
        )
    ckpt_dict = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt_dict["state_dict"])
    return model

def flatten(predictions) -> list[tuple[str, float, int]]:
    """engine.predict 返回 list[Batch]（带 batch 维），摊平成 (path, score, label)。"""
    out: list[tuple[str, float, int]] = []
    for batch in predictions or []:
        scores = batch.pred_score.detach().cpu().reshape(-1).tolist()
        labels = (batch.pred_label.detach().cpu().reshape(-1).tolist()
                  if batch.pred_label is not None else [-1] * len(scores))
        paths = batch.image_path
        if not isinstance(paths, (list, tuple)):
            paths = [paths]
        for i, s in enumerate(scores):
            out.append((str(paths[i]) if i < len(paths) else "?", float(s), int(labels[i])))
    return out


def main() -> None:
    ckpt = CKPT.resolve()
    if not ckpt.exists():
        raise SystemExit(f"[错误] ckpt 不存在：{ckpt}（先跑 patchcore_train_v2.py）")

    model = build_model_cpu()
    engine = Engine(accelerator="cpu", devices=1, logger=False, default_root_dir="runs/v2/pred")
    dataset = PredictDataset(path=str(SOURCE.resolve()), image_size=IMAGE_SIZE)

    # 第 0 轮传 ckpt_path 加载权重，之后传 None 沿用已加载权重 -> 计时不含 IO
    times: list[float] = []
    preds = None
    for r in range(WARMUP + ROUNDS):
        t0 = time.perf_counter()
        preds = engine.predict(
            model=model,
            dataset=dataset,
            # ckpt_path=str(ckpt) if r == 0 else None,
            return_predictions=True,
        )
        dt = time.perf_counter() - t0
        if r < WARMUP:
            print(f"[warmup {r}] {dt:.2f}s（含 ckpt 加载，不计入测速）")
        else:
            times.append(dt)
            print(f"[round {r - WARMUP}] {dt:.2f}s")

    records = flatten(preds)
    if not records:
        raise SystemExit("[错误] 没有预测结果，检查 SOURCE 路径")
    print(f"\n共 {len(records)} 张")
    for path, score, label in records:
        print(f"  {'NG' if label == 1 else 'OK'}  {score:.4f}  {Path(path).name}")

    if times:
        n = len(records)
        mean = sum(times) / len(times)
        print(f"\n测速：{len(times)} 轮 x {n} 张 | {mean:.2f}s/轮 = {mean / n * 1000:.1f} ms/张 "
              f"(batch 推理，不含 ckpt 加载)")


if __name__ == "__main__":
    main()
