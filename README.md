# PatchCore Demo（anomalib 2.7.0）

两个脚本，一条完整的「训练 → 推理」链路：

| 文件 | 作用 | 关键点 |
|---|---|---|
| `patchcore_train.py` | 建 memory bank、跑测试指标、落 checkpoint + `run_info.json` | `Engine.fit` / `Engine.test` |
| `patchcore_inference.py` | 读 `run_info.json` 重建模型、恢复 ckpt、出新样本的分数/标签/热力图 | `PredictDataset` + `Engine.predict(ckpt_path=...)` |

## 安装

```bash
pip install "anomalib[cu130]"     # CUDA 13.0；按机器换成 cu126 / cpu / xpu
```

2.7.0 的硬门槛：Python ≥ 3.10，PyTorch ≥ 2.10。

## 数据组织

自有数据（默认 `--dataset folder`）：

```
datasets/demo/
├── good/          # 只要正常样本，PatchCore 只用这里
├── defect/        # 缺陷图，只用于测试集评估（可留空，但指标就没意义了）
└── mask/          # 缺陷掩膜，可选；没有就算不出 pixel_AUROC / AUPRO
```

MVTec AD 直接用 `--dataset mvtec --root datasets/MVTecAD --category bottle`。

## 跑起来

```bash
# 训练 + 测试
python patchcore_train.py --dataset folder --root datasets/demo --out runs/demo

# 推理（整目录）
python patchcore_inference.py --run-info runs/demo/run_info.json --source datasets/demo/defect --out runs/demo/pred

# 推理（单张）
python patchcore_inference.py --run-info runs/demo/run_info.json --source test.png
```

## 输出

```
runs/demo/
├── run_info.json              # ckpt 路径 + 模型超参 + 测试指标 + 参考阈值
├── <Model>/<Dataset>/.../weights/lightning/model.ckpt
└── pred/
    ├── predictions.csv       # image_path, pred_score, pred_label, anomaly_map_max, overlay_path
    └── overlays/*.png        # 原图 + JET 伪彩热力图叠加
```

## 三个必须知道的点

1. **PatchCore 没有反向传播。** `fit()` = 提特征 + coreset 采样，`max_epochs` 恒为 1，没有 loss 曲线。调参只有三个真正有效的旋钮：`backbone/layers`、`coreset_sampling_ratio`、`num_neighbors`。
2. **内存账要自己算。** 256×256 输入、`layer2+layer3` → 每张图 32×32=1024 个 1536 维 patch。1000 张正常图就是 ~102 万 patch，`coreset_sampling_ratio=0.1` 后约 10 万 × 1536 × 4B ≈ 630 MB。图多/图大先降 ratio 或换 `resnet18`。
3. **阈值不属于模型，属于产线。** ckpt 里存的后处理阈值是在你训练集的验证划分上按 F1 自适应算出来的，换场景必须重新标定 —— 所以推理脚本留了 `--threshold` 覆盖口子（注意它比较的是归一化后的 `pred_score`）。

## 常见坑

- `num_workers` 在 Windows 上偶发卡死，改 `--num-workers 0`。
- 2.7 起 checkpoint 默认 `weights_only=True` 加载，自己往 hparams 里塞自定义对象会在推理时炸。
- 2.7 移除了 legacy ONNX 导出（`dynamo=False`），要导出请用默认 `dynamo=True` + `anomalib[openvino]`。
- `image_size` 在 2.x 由模型自带的 `PreProcessor` 决定（`Patchcore.configure_pre_processor(...)`），不再是 datamodule 的参数；两个脚本必须保持一致，`run_info.json` 就是干这个的。
