#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PatchCore 训练脚本 —— anomalib 2.7.0

先明确一件事：PatchCore 的 "训练" 没有梯度下降。
    fit() 实际只做三步：
      1) 用 ImageNet 预训练的 backbone（默认 wide_resnet50_2）对每张正常图提 patch 特征；
      2) layer2 + layer3 的上采样特征拼接后堆成 memory bank；
      3) k-center-greedy coreset 采样，按 coreset_sampling_ratio 保留代表性子集。
    所以 max_epochs 恒等于 1，没有 loss 曲线，也谈不上 "训不收敛"。
    真正影响效果/成本的是三个量：backbone+layers（特征维度）、
    coreset_sampling_ratio（memory bank 大小 → 显存/耗时）、num_neighbors（kNN 的 k）。

用法
    # 1) 自有数据（推荐）：root/good 放正常图，root/defect 放缺陷图，root/mask 放掩膜（可选）
    python patchcore_train.py --dataset folder --root datasets/demo --out runs/demo

    # 2) MVTec AD 某个类别
    python patchcore_train.py --dataset mvtec --root datasets/MVTecAD --category bottle --out runs/bottle

产出
    runs/demo/run_info.json   推理脚本直接读它（checkpoint 路径 + 模型超参 + 指标）
    runs/demo/.../*.ckpt      Lightning checkpoint（含 memory bank 与后处理阈值 buffer）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import anomalib
    from anomalib.data import Folder, MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
except ImportError as exc:  # pragma: no cover - 仅用于给出更友好的报错
    sys.exit(
        "[错误] 导入 anomalib 失败。请先安装 2.7.0：\n"
        '    pip install "anomalib[cu130]"    # CUDA 13.0；cu126 / cpu / xpu 按需替换\n'
        f"原始报错：{exc}"
    )


# --------------------------------------------------------------------------------------
# 命令行参数
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PatchCore 训练（anomalib 2.7.0）")

    g = p.add_argument_group("数据")
    g.add_argument("--dataset", choices=["folder", "mvtec"], default="folder",
                   help="folder: 自有目录数据；mvtec: MVTec AD 单类别")
    g.add_argument("--root", default="datasets/demo", help="数据集根目录")
    g.add_argument("--category", default="bottle", help="MVTec AD 类别（仅 --dataset mvtec 生效）")
    g.add_argument("--name", default="demo", help="数据集名字，会体现在日志/ckpt 路径里")
    g.add_argument("--normal-dir", default="good", help="正常图子目录（相对 root）")
    g.add_argument("--abnormal-dir", default="defect", help="缺陷图子目录（相对 root），没有可传空串")
    g.add_argument("--mask-dir", default="", help="缺陷掩膜子目录（相对 root），可选")

    g = p.add_argument_group("PatchCore 超参")
    g.add_argument("--backbone", default="wide_resnet50_2",
                   help="timm/torchvision backbone 名，如 wide_resnet50_2 / resnet18")
    g.add_argument("--layers", nargs="+", default=["layer2", "layer3"],
                   help="取特征的层；resnet18 建议 layer1 layer2")
    g.add_argument("--coreset-sampling-ratio", type=float, default=0.1,
                   help="memory bank 保留比例，越小越省显存、越容易漏检细微缺陷")
    g.add_argument("--num-neighbors", type=int, default=9, help="kNN 的 k")
    g.add_argument("--image-size", nargs=2, type=int, default=[256, 256],
                   help="送入网络的尺寸，默认 256 256（PatchCore 原论文设置）")

    g = p.add_argument_group("训练/IO")
    g.add_argument("--out", default="runs/patchcore_demo", help="结果目录（ckpt 与 run_info.json）")
    g.add_argument("--train-batch-size", type=int, default=16)
    g.add_argument("--eval-batch-size", type=int, default=16)
    g.add_argument("--num-workers", type=int, default=4, help="Windows 上若卡住可改 0")
    g.add_argument("--accelerator", default="auto", choices=["auto", "gpu", "cpu"])
    g.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------------------
# 数据与模型
# --------------------------------------------------------------------------------------
def build_datamodule(args: argparse.Namespace):
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        sys.exit(
            f"[错误] 数据集根目录不存在：{root}\n"
            "自有数据请按下面的结构组织后重跑：\n"
            f"  {root}/\n"
            f"  ├── {args.normal_dir}/      # 只放 OK 图，PatchCore 只用这部分\n"
            f"  ├── {args.abnormal_dir or 'defect'}/      # 缺陷图，仅用于测试集评估\n"
            f"  └── {args.mask_dir or 'mask'}/        # 像素级掩膜，可选（不给自己算不出 P-AUROC）"
        )

    common = dict(
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    if args.dataset == "mvtec":
        return MVTecAD(root=str(root), category=args.category, **common)

    kwargs: dict[str, Any] = dict(name=args.name, root=str(root), normal_dir=args.normal_dir, **common)
    if args.abnormal_dir:
        kwargs["abnormal_dir"] = args.abnormal_dir
    if args.mask_dir:
        kwargs["mask_dir"] = args.mask_dir

    normal_path = root / args.normal_dir
    if not normal_path.is_dir():
        sys.exit(f"[错误] 正常图目录不存在：{normal_path}")
    n_img = sum(1 for f in normal_path.rglob("*") if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    if n_img == 0:
        sys.exit(f"[错误] {normal_path} 里没有图片")
    print(f"[数据] 正常样本 {n_img} 张 -> {normal_path}")

    return Folder(**kwargs)


def build_model(args: argparse.Namespace) -> tuple[Patchcore, dict[str, Any]]:
    """构造 Patchcore。注意 2.7 里 resize/归一化由模型自带的 PreProcessor 负责，
    数据集层不再持有 image_size（这也是 2.x 和 1.x 最大的写法差别之一）。"""
    pre_processor = Patchcore.configure_pre_processor(image_size=tuple(args.image_size))
    model = Patchcore(
        backbone=args.backbone,
        layers=list(args.layers),
        pre_trained=True,
        coreset_sampling_ratio=args.coreset_sampling_ratio,
        num_neighbors=args.num_neighbors,
        pre_processor=pre_processor,
    )
    meta = {
        "backbone": args.backbone,
        "layers": list(args.layers),
        "coreset_sampling_ratio": args.coreset_sampling_ratio,
        "num_neighbors": args.num_neighbors,
        "image_size": list(args.image_size),
    }
    return model, meta


# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------
def to_float(value: Any) -> float | None:
    try:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().reshape(-1)[0])
        return float(value)
    except Exception:
        return None


def find_checkpoint(out_dir: Path, engine: Engine) -> str | None:
    """优先用 Engine 的 best_model_path；拿不到就回退成目录里最新的 .ckpt。"""
    try:
        best = engine.best_model_path
        if best and Path(best).exists():
            return str(Path(best).resolve())
    except Exception:
        pass
    ckpts = sorted(out_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    return str(ckpts[-1].resolve()) if ckpts else None


def read_image_threshold(model: Patchcore) -> float | None:
    """后处理阈值是 register_buffer 持久化的，fit 结束时的验证阶段会算出来。
    这里读出来写进 run_info.json，仅作参考（推理脚本默认仍用模型自带的 pred_label）。"""
    pp = getattr(model, "post_processor", None)
    for attr in ("image_threshold", "normalized_image_threshold"):
        try:
            val = to_float(getattr(pp, attr))
            if val is not None and val == val:
                return val
        except Exception:
            continue
    return None


def print_metrics(metrics: dict[str, float]) -> None:
    if not metrics:
        print("[指标] 没有拿到测试指标（可能没有异常样本做评估）")
        return
    width = max(len(k) for k in metrics)
    print("\n[指标] 测试集结果")
    print("-" * (width + 14))
    for k, v in sorted(metrics.items()):
        print(f"{k:<{width}}  {v:>8.4f}")
    print("-" * (width + 14))
    print("提示：PatchCore 关注 image_AUROC（判 OK/NG）和 pixel_AUROC / AUPRO（定位缺陷）")


def check_gpu(accelerator: str) -> None:
    """启动前把设备状态说清楚：Windows 下 pip 默认装的是 CPU 版 torch，
    Lightning 会静默回退 CPU，很多人在这里浪费一下午。"""
    if torch.cuda.is_available():
        print(f"[设备] CUDA 可用：{torch.cuda.get_device_name(0)} | "
              f"torch {torch.__version__} (cuda {torch.version.cuda})")
        return

    why = "当前 PyTorch 是 CPU 版（torch.version.cuda = None）" if not torch.version.cuda \
        else "torch 带 CUDA 但没检测到可用 GPU / 驱动不匹配"
    print(f"[警告] CUDA 不可用：{why}，本次将在 CPU 上训练，coreset 采样会明显变慢")
    print('    修复：pip uninstall torch torchvision torchaudio && '
          'pip install torch torchvision torchaudio --index-url '
          'https://download.pytorch.org/whl/cu130   # cu126/cu128/cu130 按驱动选')
    if accelerator == "gpu":
        sys.exit("[错误] 你显式指定了 --accelerator gpu，但 CUDA 不可用，直接退出以免白跑")


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    print(f"[环境] anomalib {getattr(anomalib, '__version__', 'unknown')} | torch {torch.__version__}")
    print(f"[输出] {out_dir}")
    check_gpu(args.accelerator)

    datamodule = build_datamodule(args)
    model, model_meta = build_model(args)

    # PatchCore 的 trainer_arguments 自带 max_epochs=1 / devices=1，这里显式写出来是为了可读性。
    # logger=False 省掉 tensorboard 依赖；要看曲线就换成 anomalib.loggers.AnomalibTensorBoardLogger。
    engine = Engine(
        default_root_dir=str(out_dir),
        max_epochs=1,
        accelerator=args.accelerator,
        devices=1,
        logger=False,
    )

    print("\n[1/3] fit：提特征 + 建 memory bank + coreset 采样（无反向传播）")
    engine.fit(model=model, datamodule=datamodule)

    # memory bank 规模直接决定显存/耗时，值得打出来
    bank = getattr(getattr(model, "model", None), "memory_bank", None)
    if isinstance(bank, torch.Tensor):
        n_patch, dim = bank.shape
        print(f"[memory bank] {n_patch} 个 patch × {dim} 维 ≈ {bank.numel() * 4 / 1024**2:.1f} MB (fp32)")

    print("\n[2/3] test：算 image/pixel 级指标，同时定出后处理阈值")
    metrics: dict[str, float] = {}
    results = engine.test(model=model, datamodule=datamodule)
    if results:
        metrics = {k: v for k, v in ((k, to_float(v)) for k, v in dict(results[0]).items()) if v is not None}
    print_metrics(metrics)

    print("\n[3/3] 保存 run_info.json")
    ckpt_path = find_checkpoint(out_dir, engine)
    threshold = read_image_threshold(model)
    run_info = {
        "anomalib_version": getattr(anomalib, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "ckpt_path": ckpt_path,
        "model": model_meta,
        "dataset": {
            "kind": args.dataset,
            "name": args.name if args.dataset == "folder" else args.category,
            "root": str(Path(args.root).expanduser().resolve()),
            "normal_dir": args.normal_dir if args.dataset == "folder" else None,
            "abnormal_dir": (args.abnormal_dir or None) if args.dataset == "folder" else None,
            "mask_dir": (args.mask_dir or None) if args.dataset == "folder" else None,
        },
        "metrics": metrics,
        "image_threshold_reference": threshold,
    }
    info_path = out_dir / "run_info.json"
    info_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"checkpoint : {ckpt_path}")
    print(f"run_info   : {info_path}")
    print("\n下一步：\n    python patchcore_inference.py --run-info "
          f'"{info_path}" --source path/to/images --out {out_dir / "pred"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
