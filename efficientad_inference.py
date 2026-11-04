import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import cv2
from anomalib.models import EfficientAd
from anomalib.engine import Engine

# 1. 加载你训练好的 checkpoint
ckpt_path = "./results/EfficientAd/my_product/latest/weights/lightning/model.ckpt"
export_onnx_root = "./results/EfficientAd/my_product"

def export_onnx():
    model = EfficientAd.load_from_checkpoint(ckpt_path)

    # 2. 初始化 Engine
    engine = Engine()

    # 3. 导出为 ONNX
    # input_size 必须指定，且应与训练时一致（你用的是 256x256）
    exported_path = engine.export(
        model=model,
        export_type="onnx",
        export_root= export_onnx_root,   # 输出根目录
        input_size=(256, 256),
        model_file_name="efficientad",
    )

    print(f"ONNX 模型已导出至: {exported_path}")





def infer_onnx(
    onnx_path: str,
    image_path: str,
    input_size: tuple[int, int] = (256, 256),
    providers: list[str] | None = None,
):
    """
    对单张图片执行 EfficientAD ONNX 推理。

    Args:
        onnx_path: ONNX 模型路径
        image_path: 输入图片路径
        input_size: 模型输入尺寸 (H, W)，需与导出时一致
        providers: ONNX Runtime 执行提供者，默认优先 CUDA，回退 CPU

    Returns:
        dict，包含 pred_label / pred_score / pred_mask / anomaly_map
    """
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # 1. 加载模型（实际部署时建议把 session 创建移到函数外复用）
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    # 2. 读图并预处理
    img = Image.open(image_path).convert("RGB")
    img = img.resize((input_size[1], input_size[0]), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0      # HWC, [0,1]
    x = np.transpose(x, (2, 0, 1))                      # CHW
    x = np.expand_dims(x, axis=0)                       # NCHW
    # 注意：ImageNet 归一化已固化进 ONNX 图，这里不再做 (x-mean)/std

    # 3. 推理
    outputs = session.run(None, {input_name: x})
    out_names = [o.name for o in session.get_outputs()]
    result = dict(zip(out_names, outputs))

    # 4. 整理输出
    pred_score = result["pred_score"]          # float32 [B,1]
    pred_label = result["pred_label"]          # bool    [B,1]
    anomaly_map = result["anomaly_map"]        # float32 [B,1,256,256]
    pred_mask = result["pred_mask"]            # bool    [B,1,256,256]

    return {
        "pred_score": float(pred_score[0, 0]),
        "pred_label": bool(pred_label[0, 0]),
        "anomaly_map": anomaly_map[0, 0],      # [256,256]
        "pred_mask": pred_mask[0, 0],          # [256,256] bool
    }


def infer_onnx_with_visualization(
    onnx_path: str,
    image_path: str,
    input_size: tuple[int, int] = (256, 256),
    providers: list[str] | None = None,
    save_path: str | None = None,
    colormap: int = cv2.COLORMAP_JET,
    heatmap_alpha: float = 0.5,
):
    """
    对单张图片执行 EfficientAD ONNX 推理，并生成可视化结果。

    Args:
        onnx_path: ONNX 模型路径
        image_path: 输入图片路径
        input_size: 模型输入尺寸 (H, W)
        providers: ONNX Runtime 执行提供者
        save_path: 可视化结果保存路径，None 则不保存
        colormap: 热力图颜色映射，默认 JET
        heatmap_alpha: 热力图叠加透明度

    Returns:
        dict，包含 pred_score / pred_label / anomaly_map / pred_mask / vis_image
    """
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # ---------- 1. 加载模型 ----------
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    out_names = [o.name for o in session.get_outputs()]

    # ---------- 2. 读图与预处理 ----------
    img_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img_pil.size

    img_resized = img_pil.resize((input_size[1], input_size[0]), Image.BILINEAR)
    x = np.asarray(img_resized, dtype=np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)

    # ---------- 3. 推理 ----------
    outputs = session.run(None, {input_name: x})
    result = dict(zip(out_names, outputs))

    pred_score = float(result["pred_score"][0, 0])
    pred_label = bool(result["pred_label"][0, 0])
    anomaly_map = result["anomaly_map"][0, 0]      # [H, W]
    pred_mask = result["pred_mask"][0, 0]          # [H, W] bool

    # ---------- 4. 可视化 ----------
    # 4.1 把异常图缩放到原图尺寸
    anomaly_map_resized = cv2.resize(
        anomaly_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
    )

    # 4.2 归一化到 [0,1] 后转 uint8，用于着色
    amap_norm = anomaly_map_resized.astype(np.float32)
    amap_norm = (amap_norm - amap_norm.min()) / (amap_norm.max() - amap_norm.min() + 1e-8)
    amap_uint8 = (amap_norm * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(amap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # 4.3 原图与热力图叠加
    base = np.asarray(img_pil, dtype=np.float32)
    vis = base * (1 - heatmap_alpha) + heatmap_rgb.astype(np.float32) * heatmap_alpha
    vis = np.clip(vis, 0, 255).astype(np.uint8)

    # 4.4 画 pred_mask 轮廓（缩放到原图尺寸）
    mask_resized = cv2.resize(
        pred_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    )
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.drawContours(vis_bgr, contours, -1, (0, 0, 255), 2)

    # 4.5 标注分数与标签
    vis_pil = Image.fromarray(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(vis_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    text = f"score: {pred_score:.4f}  label: {'ABNORMAL' if pred_label else 'NORMAL'}"
    text_color = (255, 0, 0) if pred_label else (0, 200, 0)
    draw.text((10, 10), text, fill=text_color, font=font)

    # 4.6 保存
    if save_path is not None:
        vis_pil.save(save_path)

    return {
        "pred_score": pred_score,
        "pred_label": pred_label,
        "anomaly_map": anomaly_map,
        "pred_mask": pred_mask,
        "vis_image": vis_pil,
    }

if __name__ == "__main__":
    # export_onnx()

    # onnx推理demo
    input_good_image = "datasets/mvtec/bottle/test/good/000.png"
    input_broken_image = "datasets/mvtec/bottle/test/broken_large/000.png"

    # res = infer_onnx(onnx_path = "results/EfficientAd/my_product/weights/onnx/efficientad.onnx",
    #                  image_path= input_good_image,)
    # print("score:", res["pred_score"])
    # print("label:", res["pred_label"])
    # print("anomaly_map range:", res["anomaly_map"].min(), res["anomaly_map"].max())
    # print("mask positive pixels:", int(res["pred_mask"].sum()))

    # onnx推理demo + 可视化
    res = infer_onnx_with_visualization(onnx_path = "results/EfficientAd/my_product/weights/onnx/efficientad.onnx",
                                        image_path= input_broken_image,
                                        save_path = "./assets/vis_broken.png",)
    print("score:", res["pred_score"])
    print("label:", res["pred_label"])
    res["vis_image"].show()