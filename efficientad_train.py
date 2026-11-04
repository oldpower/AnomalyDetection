import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="anomalib.*")

from pathlib import Path
from anomalib.data import Folder
from anomalib.models import EfficientAd
from anomalib.engine import Engine

def train():
    # 1. 路径修正：root 指向 bottle 类别目录
    DATA_ROOT = Path("./datasets/mvtec/bottle")

    # 2. 构造数据模块
    datamodule = Folder(
        name="my_product",
        root=DATA_ROOT,
        normal_dir="train/good",
        abnormal_dir="test/broken_large",
        train_batch_size=1,
        eval_batch_size=32,          # 新版用 eval_batch_size，不是 test_batch_size
        num_workers=4,
        # normalization="imagenet",
        # image_size 从 Folder 移除，改由模型 pre_processor 控制
    )

    # 3. 通过 pre_processor 设置图像尺寸
    pre_processor = EfficientAd.configure_pre_processor(image_size=(256, 256))

    # 4. 初始化 EfficientAD 模型
    model = EfficientAd(
        model_size="small",
        imagenet_dir="./datasets/imagenette/train",   # 确认该目录存在
        lr=1e-4,
        weight_decay=1e-5,
        pre_processor=pre_processor,
    )

    # 5. 训练引擎：不要强行 max_epochs=1，让模型按自身策略训练
    engine = Engine(
        accelerator="auto",
        devices=1,
        default_root_dir="./results",
    )

    # 6. 训练
    engine.fit(model=model, datamodule=datamodule)

    # 7. 测试
    engine.test(model=model, datamodule=datamodule)

    # 8. 推理
    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path="./results/EfficientAd/my_product/latest/weights/lightning/model.ckpt",
    )
    return predictions


def test():
    from torchvision import datasets
    ds = datasets.ImageFolder("./datasets/imagenette/train")
    print(len(ds), ds.classes[:5])
    # 9469 ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079']

if __name__ == "__main__":
    # test()
    train()