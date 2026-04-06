import torch
import torch.nn as nn
import torchvision.models as models
import onnx
import onnxsim
import onnxruntime as ort
import numpy as np
import warnings
import logging
import argparse
import time
import platform
from pathlib import Path
from typing import Tuple, Dict, Optional, Union, List
from packaging import version
from scipy import stats
from nets.yolo import YoloBody
import onnxruntime

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("onnx_export.log", encoding="utf-8", mode="w")
    ]
)
logger = logging.getLogger("ONNX_Exporter")

MODEL_REGISTRY = {
    "yolov8": lambda pretrained: YoloBody(input_shape=[640, 640, 3], phi='n', num_classes=80),
}

class YOLOv8ExportWrapper(nn.Module):
    """
    YOLOv8导出包装器，包含完整后处理逻辑
    输出格式: [batch, 84, 8400]
    - 前4通道: x1, y1, x2, y2 (已经是640x640尺度上的绝对坐标)
    - 后80通道: 80个类别的sigmoid概率
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # 原始forward，返回5个值
        dbox, cls, _, anchors, strides = self.model(x)

        # dbox: [batch, 4, 8400] - distance (left, top, right, bottom)
        # anchors: [2, 8400] - anchor points (cx, cy)
        # strides: [1, 8400] - stride values

        # 获取anchor points并扩展维度
        anchor_cx = anchors[0:1, :].unsqueeze(0)  # [1, 1, 8400]
        anchor_cy = anchors[1:2, :].unsqueeze(0)  # [1, 1, 8400]

        # 分离distance的四个分量
        left = dbox[:, 0:1, :]   # [batch, 1, 8400]
        top = dbox[:, 1:2, :]    # [batch, 1, 8400]
        right = dbox[:, 2:3, :]  # [batch, 1, 8400]
        bottom = dbox[:, 3:4, :] # [batch, 1, 8400]

        # 计算xyxy坐标: x1 = cx - left, y1 = cy - top, x2 = cx + right, y2 = cy + bottom
        x1 = (anchor_cx - left) * strides
        y1 = (anchor_cy - top) * strides
        x2 = (anchor_cx + right) * strides
        y2 = (anchor_cy + bottom) * strides

        # 合并为 [batch, 4, 8400]
        boxes_xyxy = torch.cat([x1, y1, x2, y2], dim=1)

        # 关键修复：确保sigmoid被正确应用
        cls_prob = torch.sigmoid(cls)

        # 合并为 [batch, 84, 8400]
        output = torch.cat([boxes_xyxy, cls_prob], dim=1)

        return output


def check_version_compatibility():
    torch_ver = version.parse(torch.__version__)
    onnx_ver = version.parse(onnx.__version__)
    ort_ver = version.parse(ort.__version__)

    assert torch_ver >= version.parse("1.8.0"), f"PyTorch版本过低（当前{torch.__version__}，需≥1.8.0）"
    assert onnx_ver >= version.parse("1.10.0"), f"ONNX版本过低（当前{onnx.__version__}，需≥1.10.0）"
    assert ort_ver >= version.parse("1.8.0"), f"ONNX Runtime版本过低（当前{ort.__version__}，需≥1.8.0）"
    logger.info(
        f"✅ 版本兼容性校验通过：PyTorch={torch.__version__}，ONNX={onnx.__version__}，ONNXRuntime={ort.__version__}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch 模型导出 ONNX 工具（企业级）")
    parser.add_argument("--model-name", type=str, default="yolov8",
                        choices=list(MODEL_REGISTRY.keys()), help="模型名称")
    parser.add_argument("--weight-path", type=str, default="", help="自定义模型权重路径（空则用预训练权重）")
    parser.add_argument("--onnx-save-path", type=str, default="", help="ONNX 保存路径（空则自动生成）")
    parser.add_argument("--opset-version", type=int, default=17, choices=[11, 12, 13, 14, 17], help="ONNX 算子版本")
    parser.add_argument("--batch-size", type=int, default=1, help="示例批次大小（需≥1）")
    parser.add_argument("--input-shape", type=int, nargs=3, default=[3, 640, 640], help="输入形状 (C, H, W)")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"], help="运行设备")
    parser.add_argument("--dynamic-batch", action="store_true", default=False, help="是否启用动态批次")
    parser.add_argument("--dynamic-shape", action="store_true", default=False, help="是否启用动态高宽")
    parser.add_argument("--verify-model", action="store_true", default=True, help="导出后验证模型有效性")
    parser.add_argument("--benchmark", action="store_true", default=True, help="导出后测试推理性能")
    parser.add_argument("--simplify-onnx", action="store_true", default=True, help="简化ONNX模型")
    parser.add_argument("--task-type", type=str, default="detection", choices=["classification", "detection"],
                        help="任务类型")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志输出级别")

    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.log_level))

    if args.batch_size < 1:
        raise ValueError(f"batch_size必须≥1，当前值：{args.batch_size}")
    if len(args.input_shape) != 3 or any(d <= 0 for d in args.input_shape):
        raise ValueError(f"input_shape必须是3个正整数（C,H,W），当前值：{args.input_shape}")

    if not args.onnx_save_path:
        args.onnx_save_path = f"./{args.model_name}_bs{args.batch_size}_opset{args.opset_version}.onnx"

    print("===== 最终解析的参数 =====")
    for key, value in vars(args).items():
        print(f"{key} = {value}")
    print("==========================")

    return args

def get_device(args_device: str) -> torch.device:
    cuda_available = torch.cuda.is_available()

    if args_device == "auto":
        device = torch.device("cuda" if cuda_available else "cpu")
    elif args_device == "cuda":
        if cuda_available:
            device = torch.device("cuda")
        else:
            logger.warning("⚠️ 指定使用CUDA，但当前环境无CUDA，自动降级为CPU运行")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    logger.info(f"✅ 最终运行设备：{device}（当前环境CUDA可用：{cuda_available}）")
    return device

def check_cuda_memory(device: torch.device, threshold_ratio: float = 0.8) -> None:
    if device.type == "cuda":
        mem_used = torch.cuda.memory_allocated(device) / (1024 ** 3)
        mem_total = torch.cuda.memory_total(device) / (1024 ** 3)
        mem_used_ratio = mem_used / mem_total

        if mem_used_ratio > threshold_ratio:
            logger.warning(
                f"⚠️ CUDA内存不足预警：已用{mem_used:.2f}GB/{mem_total:.2f}GB（{mem_used_ratio:.1%}），触发缓存清理"
            )
            torch.cuda.empty_cache()

def remove_outliers(data: List[float], z_threshold: float = 3.0) -> List[float]:
    if len(data) < 2:
        return data
    z_scores = np.abs(stats.zscore(data))
    return [d for d, z in zip(data, z_scores) if z < z_threshold]

def load_model(
        model_name: str,
        weight_path: str = "",
        device: torch.device = torch.device("cpu")
) -> torch.nn.Module:
    logger.info(f"开始加载模型：{model_name}，目标设备：{device}")

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"不支持的模型：{model_name}，支持列表：{list(MODEL_REGISTRY.keys())}")
    model = MODEL_REGISTRY[model_name](pretrained=not bool(weight_path))

    if weight_path:
        weight_path = Path(weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"权重文件不存在：{weight_path}")
        logger.info(f"加载自定义权重：{weight_path}")
        state_dict = torch.load(weight_path, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)

        if missing_keys:
            logger.warning(f"⚠️ 权重缺失：{missing_keys}")
        if unexpected_keys:
            logger.warning(f"⚠️ 冗余权重（已忽略）：{unexpected_keys}")

    model = model.to(device).eval()
    logger.info("✅ 模型加载完成（已切换到eval模式并移至目标设备）")
    return model

def create_dummy_input(
        batch_size: int,
        input_shape: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    logger.info(f"构造虚拟输入：批次={batch_size}，形状={input_shape}，设备={device}")
    dummy_input = torch.randn(batch_size, *input_shape, dtype=dtype)
    dummy_input = dummy_input.to(device)
    check_cuda_memory(device)
    return dummy_input

def get_dynamic_axes(
        dynamic_batch: bool = True,
        dynamic_shape: bool = False
) -> Union[Dict[str, Dict[int, str]], None]:
    if not dynamic_batch and not dynamic_shape:
        logger.info("动态维度配置：无")
        return None

    dynamic_axes = {"input": {}, "output": {}}
    if dynamic_batch:
        dynamic_axes["input"][0] = "batch_size"
        dynamic_axes["output"][0] = "batch_size"
    if dynamic_shape:
        dynamic_axes["input"][2] = "height"
        dynamic_axes["input"][3] = "width"
    logger.info(f"动态维度配置：{dynamic_axes}")
    return dynamic_axes

def export_onnx(
        model: torch.nn.Module,
        dummy_input: torch.Tensor,
        onnx_save_path: str,
        opset_version: int = 17,
        dynamic_axes: Optional[Dict] = None,
        input_names: list = ["input"],
        output_names: list = ["output"],
        simplify_onnx: bool = True
) -> None:
    onnx_save_path = Path(onnx_save_path)
    onnx_save_path.parent.mkdir(parents=True, exist_ok=True)
    if not onnx_save_path.parent.is_dir():
        raise PermissionError(f"保存路径无写入权限：{onnx_save_path.parent}")

    logger.info(f"开始导出ONNX模型：保存路径={onnx_save_path}，算子版本={opset_version}")
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_save_path),
                opset_version=opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes or {},
                do_constant_folding=True,
                export_params=True,
            )

        logger.info(f"✅ ONNX模型导出成功：{onnx_save_path}")

        if simplify_onnx:
            logger.info("开始简化ONNX模型...")
            input_shapes = {input_names[0]: list(dummy_input.shape)}

            onnx_model = onnx.load(str(onnx_save_path))
            simplified_model, check = onnxsim.simplify(
                onnx_model,
                dynamic_input_shape=False,
                input_shapes=input_shapes
            )
            assert check, "简化后的模型校验失败"
            onnx.save(simplified_model, str(onnx_save_path))
            logger.info("✅ ONNX模型简化成功")

    except Exception as e:
        if onnx_save_path.exists():
            onnx_save_path.unlink()
            logger.warning(f"❌ 导出失败，已清理无效文件：{onnx_save_path}")
        logger.error(f"ONNX导出失败：{e}", exc_info=True)
        raise

def verify_onnx_model(
    onnx_path: str,
    pytorch_model: torch.nn.Module,
    dummy_input: torch.Tensor,
    device: torch.device,
    task_type: str = "detection"
):
    logger.info("验证ONNX模型结构完整性...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info("✅ ONNX模型结构校验通过")

    logger.info("验证PyTorch vs ONNX推理结果一致性")

    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input)
        if isinstance(pytorch_output, (list, tuple)):
            pytorch_output = pytorch_output[0]
        pytorch_output = pytorch_output.detach().cpu().numpy()

    ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_output = ort_outs[0]

    # 详细统计输出
    logger.info(f"PyTorch输出形状: {pytorch_output.shape}")
    logger.info(f"ONNX输出形状: {onnx_output.shape}")
    logger.info(f"PyTorch输出范围: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
    logger.info(f"ONNX输出范围: [{onnx_output.min():.6f}, {onnx_output.max():.6f}]")
    logger.info(f"PyTorch前4通道范围: [{pytorch_output[:, :4, :].min():.6f}, {pytorch_output[:, :4, :].max():.6f}]")
    logger.info(f"ONNX前4通道范围: [{onnx_output[:, :4, :].min():.6f}, {onnx_output[:, :4, :].max():.6f}]")
    logger.info(f"PyTorch后80通道范围: [{pytorch_output[:, 4:, :].min():.6f}, {pytorch_output[:, 4:, :].max():.6f}]")
    logger.info(f"ONNX后80通道范围: [{onnx_output[:, 4:, :].min():.6f}, {onnx_output[:, 4:, :].max():.6f}]")

    cos_sim = np.sum(pytorch_output * onnx_output) / (np.linalg.norm(pytorch_output) * np.linalg.norm(onnx_output) + 1e-8)
    mse = np.mean(np.power(pytorch_output - onnx_output, 2))

    logger.info(f"✅ 余弦相似度 = {cos_sim:.6f}")
    logger.info(f"✅ MSE 误差 = {mse:.8f}")

    if cos_sim > 0.99 and mse < 1e-3:
        logger.info("🎉 ONNX 与 PyTorch 推理结果完全一致！")
    else:
        logger.warning("⚠️  存在差异，请检查模型导出逻辑")

def benchmark_onnx_model(onnx_path: str, dummy_input: torch.Tensor, device: torch.device, warmup: int = 10,
                         test_times: int = 100):
    logger.info(f"开始性能基准测试：预热{warmup}次，测试{test_times}次")
    input_np = dummy_input.cpu().numpy()

    ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name

    for _ in range(warmup):
        ort_session.run(None, {input_name: input_np})

    start = time.time()
    for _ in range(test_times):
        ort_session.run(None, {input_name: input_np})
    cost = time.time() - start

    avg = cost / test_times * 1000
    logger.info(f"✅ ONNX 性能测试：平均耗时 {avg:.2f} ms/帧")
    logger.info(f"✅ 理论帧率：{1000 / avg:.1f} FPS")

def main():
    try:
        check_version_compatibility()
        args = parse_args()
        device = get_device(args.device)

        check_cuda_memory(device)
        model = load_model(args.model_name, args.weight_path, device)

        logger.info("使用YOLOv8ExportWrapper包装模型...")
        model = YOLOv8ExportWrapper(model)
        model = model.to(device).eval()

        dummy_input = create_dummy_input(args.batch_size, tuple(args.input_shape), device)
        dynamic_axes = get_dynamic_axes(args.dynamic_batch, args.dynamic_shape)

        check_cuda_memory(device)
        export_onnx(
            model=model,
            dummy_input=dummy_input,
            onnx_save_path=args.onnx_save_path,
            opset_version=args.opset_version,
            dynamic_axes=dynamic_axes,
            simplify_onnx=args.simplify_onnx
        )

        if args.verify_model:
            verify_onnx_model(args.onnx_save_path, model, dummy_input, device, args.task_type)

        if args.benchmark:
            benchmark_onnx_model(args.onnx_save_path, dummy_input, device)

        del model, dummy_input
        if device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info("🎉 ONNX模型导出全流程完成！")
        logger.info(f"输出格式说明: [batch, 84, 8400]")
        logger.info(f"  - 前4通道: x1, y1, x2, y2 (640x640尺度上的绝对坐标)")
        logger.info(f"  - 后80通道: 80个类别的概率 [0, 1]")
    except Exception as e:
        logger.error(f"❌ 全流程执行失败：{str(e)}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
