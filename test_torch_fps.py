import time
import torch
from nets.yolo import YoloBody  # 导入你的原始模型

# ===================== 配置（和你导出时一致） =====================
MODEL_WEIGHT_PATH = "weight/yolov8_n.pth"  # 你的权重路径
INPUT_SHAPE = (3, 224, 224)               # 输入大小
DEVICE = "cpu"                            # 运行设备
WARMUP = 10                               # 预热次数
TEST_TIMES = 100                           # 测试次数

# ===================== 加载模型 =====================
print("🔹 加载原始 PyTorch 模型...")

# 构建 YOLO 模型（和你导出时一样）
model = YoloBody(
    input_shape=INPUT_SHAPE,
    phi='n',  # yolov8-n 模型
    num_classes=80  # COCO类别
)

# 加载权重
checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()  # 推理模式
print("✅ 模型加载完成")

# 构造输入
dummy_input = torch.randn(1, *INPUT_SHAPE).to(DEVICE)

# ===================== 预热 =====================
print(f"🔹 预热 {WARMUP} 次...")
with torch.no_grad():
    for _ in range(WARMUP):
        model(dummy_input)

# ===================== 正式测速 =====================
print(f"🔹 开始测试 {TEST_TIMES} 次推理...")
start_time = time.time()

with torch.no_grad():
    for _ in range(TEST_TIMES):
        model(dummy_input)

total_time = time.time() - start_time

# ===================== 计算结果 =====================
avg_ms = (total_time / TEST_TIMES) * 1000
fps = 1000 / avg_ms

print("-" * 50)
print(f"✅ PyTorch 原始模型性能：")
print(f"🕒 单帧平均耗时：{avg_ms:.2f} ms")
print(f"⚡ 理论帧率：{fps:.1f} FPS")
print("-" * 50)