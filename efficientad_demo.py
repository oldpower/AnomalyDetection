from anomalib.models import EfficientAd
from anomalib.engine import Engine

# 初始化 EfficientAD 模型 (支持 small 或 medium 规格)
model = EfficientAd(model_size="small") 

# 使用 Anomalib 的 Engine 进行训练或推理
engine = Engine()
# engine.fit(model=model, datamodule=datamodule)