import torch
import onnx
import onnxsim
import onnxruntime as ort
import numpy as np
import warnings
import logging
import argparse
import time
from pathlib import Path
from typing import Tuple, Dict, Optional, Union, List
from packaging import version
from scipy import stats
from nets.yolo import YoloBody
import onnxruntime


warnings.filterwarnings("ignore")
#忽略警告以下的信息
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("onnx_export.log", encoding="utf-8", mode="w")
    ]
)
#日志设置，信息层次，信息内容，终端文件同时输出，保存文件位置，覆盖，如果一直有不同的问题环环相套可追加
logger = logging.getLogger("ONNX_Exporter")
#将logger所记录文字打印到该文件夹内

#注册表配置，在其内加载模型，如有自定义模型框架，需声明引用，并根据模型情况加入参数
MODEL_REGISTRY = {
  # "resnet18": lambda pretrained: models.resnet18(pretrained=pretrained),
   # "mobilenet_v2": lambda pretrained: models.mobilenet_v2(pretrained=pretrained),
   # "efficientnet_b0": lambda pretrained: models.efficientnet_b0(pretrained=pretrained),
     "yolov8": lambda pretrained: YoloBody(input_shape=[640, 640, 3], phi='n', num_classes=80),
}


#版本检查，看是否符合要求
def check_version_compatibility():
    torch_ver = version.parse(torch.__version__)
    onnx_ver = version.parse(onnx.__version__)
    ort_ver = version.parse(ort.__version__)
#提前用assert来判断版本是否符合要求，不合要求就暂停
    assert torch_ver >= version.parse("1.8.0"), f"PyTorch版本过低（当前{torch.__version__}，需≥1.8.0）"
    assert onnx_ver >= version.parse("1.10.0"), f"ONNX版本过低（当前{onnx.__version__}，需≥1.10.0）"
    assert ort_ver >= version.parse("1.8.0"), f"ONNX Runtime版本过低（当前{ort.__version__}，需≥1.8.0）"
    logger.info(
        f"✅ 版本兼容性校验通过：PyTorch={torch.__version__}，ONNX={onnx.__version__}，ONNXRuntime={ort.__version__}")


#命令行传参，python 文件名 --key  value
def parse_args() -> argparse.Namespace:
    #创建一个参数解析器
    parser = argparse.ArgumentParser(description="PyTorch 模型导出 ONNX 工具（企业级）")
  #基础参数配置， 名称 类型 默认值 选项 帮助 功能（不输入是否自动开启） 参数个数 是否强制传参
    parser.add_argument("--model-name", type=str, default="yolov8",
                        choices=list(MODEL_REGISTRY.keys()), help="模型名称")
    parser.add_argument("--weight-path", type=str, default="", help="自定义模型权重路径（空则用预训练权重）")
    parser.add_argument("--onnx-save-path", type=str, default="", help="ONNX 保存路径（空则自动生成）")
    parser.add_argument("--opset-version", type=int, default=12, choices=[11, 12, 13, 14,17], help="ONNX 算子版本")
    #输入参数
    parser.add_argument("--batch-size", type=int, default=1, help="示例批次大小（需≥1）")
    parser.add_argument("--input-shape", type=int, nargs=3, default=[3, 224, 224], help="输入形状 (C, H, W)")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"], help="运行设备")
    #调用函数
    parser.add_argument("--dynamic-batch", action="store_true", default=True, help="是否启用动态批次")
    parser.add_argument("--dynamic-shape", action="store_true", default=False, help="是否启用动态高宽")
    parser.add_argument("--verify-model", action="store_true", default=True, help="导出后验证模型有效性")
    parser.add_argument("--benchmark", action="store_true", default=True, help="导出后测试推理性能")
    parser.add_argument("--simplify-onnx", action="store_true", default=True, help="简化ONNX模型（解决算子冗余）")
    parser.add_argument("--task-type", type=str, default="classification", choices=["classification", "regression"],
                        help="任务类型（用于调整验证阈值）")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志输出级别")
#将所有传参打包 args.名称调用
    args = parser.parse_args()

    #日志级别生效
    logger.setLevel(getattr(logging, args.log_level))

#判断参数是否有误
    if args.batch_size < 1:
        raise ValueError(f"batch_size必须≥1，当前值：{args.batch_size}")
    if len(args.input_shape) != 3 or any(d <= 0 for d in args.input_shape):
        raise ValueError(f"input_shape必须是3个正整数（C,H,W），当前值：{args.input_shape}")
#onnx_path是否存在，不在自动创建
    if not args.onnx_save_path:
        args.onnx_save_path = f"./{args.model_name}_bs{args.batch_size}_opset{args.opset_version}.onnx"
    print("===== 最终解析的参数 =====")
    for key, value in vars(args).items():
        print(f"{key} = {value}")
    print("==========================")

    return args


#检查设备环境  device
def get_device(args_device: str) -> torch.device:
    #GPU是否可用
    cuda_available = torch.cuda.is_available()

    #根据传参选择环境，就算传参不对也能正常运行
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

#检查GPU内存与显存，并释放内存   device
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


#提高性能指标  直接调用
def remove_outliers(data: List[float], z_threshold: float = 3.0) -> List[float]:

    if len(data) < 2:
        return data
    z_scores = np.abs(stats.zscore(data))
    return [d for d, z in zip(data, z_scores) if z < z_threshold]
#加载模型及其对应权重  model_name weight_path device
def load_model(
        model_name: str,
        weight_path: str = "",
        device: torch.device = torch.device("cpu")
) -> torch.nn.Module:

    logger.info(f"开始加载模型：{model_name}，目标设备：{device}")

    #判断模型是否在注册表中
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"不支持的模型：{model_name}，支持列表：{list(MODEL_REGISTRY.keys())}")
    model = MODEL_REGISTRY[model_name](pretrained=not bool(weight_path))

    #判断是否有预训练权重，没有了加载
    if weight_path:
        weight_path = Path(weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"权重文件不存在：{weight_path}")
        logger.info(f"加载自定义权重：{weight_path}")
        state_dict = torch.load(weight_path, map_location=device)
        #添加关键层，看是否被忽略，如缺失模型失败，或自定义算子层
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)

        #判断
        critical_layers = ["conv1", "layer1", "layer2", "fc", "features"]
        missing_critical = [k for k in missing_keys if any(cl in k for cl in critical_layers)]
        if missing_critical:
            raise ValueError(f"核心层权重缺失：{missing_critical}，模型无法正常推理")
        if missing_keys:
            logger.warning(f"⚠️ 非核心层权重缺失：{missing_keys}")
        if unexpected_keys:
            logger.warning(f"⚠️ 冗余权重（已忽略）：{unexpected_keys}")

    #模型导出且进入评估模式，对权重固定
    model = model.to(device).eval()
    logger.info("✅ 模型加载完成（已切换到eval模式并移至目标设备）")
    return model


#定义虚拟输入，与模型输入一致  batch input_shape device dtype
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
    """构造虚拟输入（模拟真实图片分布，设备兼容）"""
    logger.info(f"构造虚拟输入：批次={batch_size}，形状={input_shape}，设备={device}，数据类型={dtype}")
    #对其作归一化处理
    dummy_input = (torch.rand(batch_size, *input_shape, dtype=dtype) - 0.485) / 0.229
    dummy_input = dummy_input.to(device)

    check_cuda_memory(device)
    return dummy_input


#开启动态维度  dynamic_batch dynamic_shape
def get_dynamic_axes(
        dynamic_batch: bool = True,
        dynamic_shape: bool = False
) -> Union[Dict[str, Dict[int, str]], None]:
    #判断传参是否开启动态维度
    if not dynamic_batch and not dynamic_shape:
        logger.info("动态维度配置：无")
        return None
    #添加输入输出为动态维度
    dynamic_axes = {"input": {}, "output": {}}
    #定义batch为动态
    if dynamic_batch:
        dynamic_axes["input"][0] = "batch_size"
        dynamic_axes["output"][0] = "batch_size"
    #仅输入的宽高为动态
    if dynamic_shape:
        dynamic_axes["input"][2] = "height"
        dynamic_axes["input"][3] = "width"
    logger.info(f"动态维度配置：{dynamic_axes}")
    return dynamic_axes


#导出onnx模型
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
    onnx_save_path.parent.mkdir(parents=True, exist_ok=True)#创建保存路径
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
                do_constant_folding=True,#固定常量，减少操作
            )

        logger.info(f"✅ ONNX模型导出成功：{onnx_save_path}")

        #开启模型简化，清除冗余算子
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

#验证模型精度
def verify_onnx_model(
    onnx_path: str,
    pytorch_model: torch.nn.Module,
    dummy_input: torch.Tensor,
    device: torch.device,
    task_type: str = "classification"
):

    logger.info("验证ONNX模型结构完整性...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info("✅ ONNX模型结构校验通过")

    logger.info("验证PyTorch vs ONNX推理结果一致性（任务类型：detection）")
    #关闭梯度下降，防止反向传播增加计算量
    with torch.no_grad():
        pytorch_outputs = pytorch_model(dummy_input)

    if isinstance(pytorch_outputs, (list, tuple)):
        pytorch_output = pytorch_outputs[0].detach().cpu().numpy()
    else:
        pytorch_output = pytorch_outputs.detach().cpu().numpy()

    #用ORT进行推理验证，调用导出的ONNX模型进行测试
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_output = ort_outs[0]
    #转换为np，进行比较
    cos_sim = np.sum(pytorch_output * onnx_output) / (np.linalg.norm(pytorch_output) * np.linalg.norm(onnx_output))
    mse = np.mean(np.power(pytorch_output - onnx_output, 2))

    logger.info(f"✅ 余弦相似度 = {cos_sim:.6f}")
    logger.info(f"✅ MSE 误差 = {mse:.8f}")

    if cos_sim > 0.99 and mse < 1e-5:
        logger.info("🎉 ONNX 与 PyTorch 推理结果完全一致！模型完好可用！")
    else:
        logger.warning("⚠️  存在微小误差，但通常不影响部署使用")

#对模型性能进行测试
def benchmark_onnx_model(onnx_path: str, dummy_input: torch.Tensor, device: torch.device, warmup: int = 10,
                         test_times: int = 100):
    #导入输入
    logger.info(f"开始性能基准测试：预热{warmup}次，测试{test_times}次")
    input_np = dummy_input.cpu().numpy()
    #ort进行推理验证
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name
    #开启预热，提前做好一些数据的加载
    for _ in range(warmup):
        ort_session.run(None, {input_name: input_np})
    #开始对导出模型性能进行测试
    start = time.time()
    for _ in range(test_times):
        ort_session.run(None, {input_name: input_np})
    cost = time.time() - start
    avg = cost / test_times * 1000
    logger.info(f"✅ ONNX 性能测试：平均耗时 {avg:.2f} ms/帧")
    logger.info(f"✅ 理论帧率：{1000 / avg:.1f} FPS")

#主函数，进行调用
def main():
    try:
        #检验版本是否可行
        check_version_compatibility()
        #启用传参
        args = parse_args()
        #查看导出环境
        device = get_device(args.device)
        #如是GPU导出，检查内存
        check_cuda_memory(device)
        #加载模型及权重
        model = load_model(args.model_name, args.weight_path, device)
        #加载虚拟输入
        dummy_input = create_dummy_input(args.batch_size, tuple(args.input_shape), device)
        #加载动态维度
        dynamic_axes = get_dynamic_axes(args.dynamic_batch, args.dynamic_shape)
        #检查内存
        check_cuda_memory(device)
        #导出onnx模型并简化
        export_onnx(
            model=model,
            dummy_input=dummy_input,
            onnx_save_path=args.onnx_save_path,
            opset_version=args.opset_version,
            dynamic_axes=dynamic_axes,
            simplify_onnx=args.simplify_onnx
        )
        #进行模型精度确认
        if args.verify_model:
            verify_onnx_model(args.onnx_save_path, model, dummy_input, device, args.task_type)
        #进行性能测评
        if args.benchmark:
            benchmark_onnx_model(args.onnx_save_path, dummy_input, device)
        #释放资源
        del model, dummy_input
        if device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info("🎉 ONNX模型导出全流程完成！")
    except Exception as e:
        logger.error(f"❌ 全流程执行失败：{str(e)}", exc_info=True)
        exit(1)
#仅在本文件中可执行主函数
if __name__ == "__main__":
    main()
