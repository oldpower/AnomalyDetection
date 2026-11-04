import time
import numpy as np
import onnxruntime as ort
from PIL import Image


def benchmark_onnx(
    onnx_path: str,
    image_path: str,
    input_size: tuple[int, int] = (256, 256),
    providers: list[str] | None = None,
    warmup_iters: int = 10,
    bench_iters: int = 100,
):
    """
    对 EfficientAD ONNX 模型做延迟与吞吐量测试。

    Args:
        onnx_path: ONNX 模型路径
        image_path: 用于构造输入的图片路径
        input_size: 模型输入尺寸 (H, W)
        providers: ONNX Runtime 执行提供者
        warmup_iters: 预热次数（不计入统计）
        bench_iters: 正式测试次数

    Returns:
        dict，包含各阶段延迟统计和吞吐量
    """
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # ---------- 1. 创建 session（只创建一次） ----------
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    # ---------- 2. 准备预处理后的固定输入 ----------
    img = Image.open(image_path).convert("RGB")
    img = img.resize((input_size[1], input_size[0]), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    x = np.ascontiguousarray(x)

    # ---------- 3. 预热 ----------
    for _ in range(warmup_iters):
        session.run(None, {input_name: x})

    # ---------- 4. 纯推理延迟 ----------
    infer_times = []
    for _ in range(bench_iters):
        t0 = time.perf_counter()
        session.run(None, {input_name: x})
        infer_times.append((time.perf_counter() - t0) * 1000)  # ms

    # ---------- 5. 端到端延迟（含读图 + 预处理 + 推理） ----------
    e2e_times = []
    for _ in range(bench_iters):
        t0 = time.perf_counter()

        img = Image.open(image_path).convert("RGB")
        img = img.resize((input_size[1], input_size[0]), Image.BILINEAR)
        x_tmp = np.asarray(img, dtype=np.float32) / 255.0
        x_tmp = np.transpose(x_tmp, (2, 0, 1))
        x_tmp = np.expand_dims(x_tmp, axis=0)
        x_tmp = np.ascontiguousarray(x_tmp)

        session.run(None, {input_name: x_tmp})

        e2e_times.append((time.perf_counter() - t0) * 1000)

    # ---------- 6. 统计 ----------
    def stats(times):
        arr = np.array(times)
        return {
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p99_ms": float(np.percentile(arr, 99)),
        }

    infer_stats = stats(infer_times)
    e2e_stats = stats(e2e_times)

    return {
        "provider": session.get_providers()[0],
        "inference": infer_stats,
        "end_to_end": e2e_stats,
        "throughput_fps_inference": 1000.0 / infer_stats["mean_ms"],
        "throughput_fps_e2e": 1000.0 / e2e_stats["mean_ms"],
    }


def print_benchmark_report(res: dict):
    """格式化打印测速结果。"""
    print(f"Provider: {res['provider']}")
    print()
    print("=== 纯推理延迟 (ms) ===")
    for k, v in res["inference"].items():
        print(f"  {k:10s}: {v:8.3f}")
    print(f"  吞吐量    : {res['throughput_fps_inference']:8.1f} FPS")
    print()
    print("=== 端到端延迟 (ms) ===")
    for k, v in res["end_to_end"].items():
        print(f"  {k:10s}: {v:8.3f}")
    print(f"  吞吐量    : {res['throughput_fps_e2e']:8.1f} FPS")

if __name__ == "__main__":
    res = benchmark_onnx(onnx_path="results/EfficientAd/my_product/weights/onnx/efficientad.onnx",
                         image_path="datasets/mvtec/bottle/test/broken_large/000.png",
                         warmup_iters=10,
                         bench_iters=100,)
    print_benchmark_report(res)