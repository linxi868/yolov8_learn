import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms.functional as TF
import os


class YOLOv8Detector:
    def __init__(self, onnx_model_path, conf_thres=0.25, iou_thres=0.45):
        """
        YOLOv8 ONNX推理器

        Args:
            onnx_model_path: ONNX模型路径
            conf_thres: 置信度阈值
            iou_thres: NMS的IoU阈值
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_shape = (640, 640)

        # COCO 80类
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        print(f"正在加载ONNX模型: {onnx_model_path}")
        self.session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

        print("\n模型输入信息:")
        for inp in self.session.get_inputs():
            print(f"  {inp.name}: {inp.shape}")

        print("\n模型输出信息:")
        for out in self.session.get_outputs():
            print(f"  {out.name}: {out.shape}")

        print("✅ 模型加载成功\n")

    def preprocess(self, image):
        """预处理图片"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        orig_width, orig_height = image.size

        # Letterbox
        scale = min(self.input_shape[0] / orig_width, self.input_shape[1] / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        image_resized = image.resize((new_width, new_height), Image.BILINEAR)
        image_padded = Image.new('RGB', self.input_shape, (114, 114, 114))
        paste_x = (self.input_shape[0] - new_width) // 2
        paste_y = (self.input_shape[1] - new_height) // 2
        image_padded.paste(image_resized, (paste_x, paste_y))

        # 归一化到[0, 1]
        image_tensor = TF.to_tensor(image_padded).unsqueeze(0).float()

        letterbox_info = {
            'scale': scale,
            'pad_w': paste_x,
            'pad_h': paste_y,
            'orig_width': orig_width,
            'orig_height': orig_height
        }

        return image_tensor.numpy(), letterbox_info

    def postprocess(self, outputs, letterbox_info):
        """
        后处理

        Args:
            outputs: ONNX输出，形状 [1, 84, 8400]
            letterbox_info: letterbox信息

        Returns:
            detections: 检测结果列表
        """
        output = outputs[0]  # [1, 84, 8400]
        print(f"ONNX输出形状: {output.shape}")

        # 调试：查看输出的统计信息
        print(f"输出最小值: {output.min():.6f}, 最大值: {output.max():.6f}")
        print(f"前4通道(box)范围: [{output[:, :4, :].min():.6f}, {output[:, :4, :].max():.6f}]")
        print(f"后80通道(cls)范围: [{output[:, 4:, :].min():.6f}, {output[:, 4:, :].max():.6f}]")

        # 转置为 [8400, 84]
        output = output[0].T  # [8400, 84]

        # 分离box和cls
        # ONNX输出: 前4通道是(x1, y1, x2, y2)，后80通道是类别概率
        boxes = output[:, :4]  # [8400, 4]
        scores = output[:, 4:]  # [8400, 80]

        # 过滤低置信度
        scores_max = scores.max(axis=1)  # [8400]
        class_ids = scores.argmax(axis=1)  # [8400]

        print(f"置信度统计 - 最小: {scores_max.min():.6f}, 最大: {scores_max.max():.6f}, 平均: {scores_max.mean():.6f}")
        print(f"高于阈值({self.conf_thres})的数量: {(scores_max >= self.conf_thres).sum()}")

        mask = scores_max >= self.conf_thres
        boxes_filtered = boxes[mask]
        scores_filtered = scores_max[mask]
        class_ids_filtered = class_ids[mask]

        print(f"置信度过滤: {len(boxes)} -> {len(boxes_filtered)}")

        if len(boxes_filtered) == 0:
            return []

        # NMS
        indices = self._nms(boxes_filtered, scores_filtered, class_ids_filtered)
        print(f"NMS后: {len(indices)} 个框")

        detections = []
        for idx in indices:
            box = boxes_filtered[idx]
            conf = float(scores_filtered[idx])
            class_id = int(class_ids_filtered[idx])

            # box是(x1, y1, x2, y2)在640x640尺度上的坐标
            x1, y1, x2, y2 = box

            # 还原到原始图像
            scale = letterbox_info['scale']
            pad_w = letterbox_info['pad_w']
            pad_h = letterbox_info['pad_h']

            # 去除letterbox padding并缩放到原图
            x1 = (x1 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            x2 = (x2 - pad_w) / scale
            y2 = (y2 - pad_h) / scale

            # 裁剪到图像边界
            orig_w = letterbox_info['orig_width']
            orig_h = letterbox_info['orig_height']
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            # 确保框的有效性
            if x2 <= x1 or y2 <= y1:
                continue

            # 检查框的合理性（面积不能太大）
            box_area = (x2 - x1) * (y2 - y1)
            img_area = orig_w * orig_h
            if box_area > img_area * 0.5:  # 单个框不能超过图像的50%
                continue

            detections.append({
                'box': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': conf,
                'class_id': class_id,
                'class_name': self.class_names[class_id]
            })

        return detections

    def _nms(self, boxes, scores, class_ids):
        """NMS - boxes是(x1, y1, x2, y2)格式"""
        boxes_xyxy = boxes.copy()

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            # IoU计算
            xx1 = np.maximum(boxes_xyxy[i, 0], boxes_xyxy[order[1:], 0])
            yy1 = np.maximum(boxes_xyxy[i, 1], boxes_xyxy[order[1:], 1])
            xx2 = np.minimum(boxes_xyxy[i, 2], boxes_xyxy[order[1:], 2])
            yy2 = np.minimum(boxes_xyxy[i, 3], boxes_xyxy[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            area_i = (boxes_xyxy[i, 2] - boxes_xyxy[i, 0]) * (boxes_xyxy[i, 3] - boxes_xyxy[i, 1])
            area_o = (boxes_xyxy[order[1:], 2] - boxes_xyxy[order[1:], 0]) * \
                     (boxes_xyxy[order[1:], 3] - boxes_xyxy[order[1:], 1])

            # 防止除零
            union = area_i + area_o - inter
            union = np.maximum(union, 1e-6)
            iou = inter / union

            # 只对同类别进行NMS
            same_class = class_ids[order[1:]] == class_ids[i]
            mask = (iou < self.iou_thres) | (~same_class)
            order = order[1:][mask]

        return keep

    def detect(self, image_path):
        """检测单张图片"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        print(f"图片尺寸: {image.shape[1]}x{image.shape[0]}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_data, letterbox_info = self.preprocess(image_rgb)

        print("开始推理...")
        outputs = self.session.run(None, {self.input_name: input_data})

        detections = self.postprocess(outputs, letterbox_info)

        return detections, image

    def draw_detections(self, image, detections):
        """绘制检测结果"""
        img_copy = image.copy()

        if len(detections) == 0:
            cv2.putText(img_copy, "No detections", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return img_copy

        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            confidence = det['confidence']
            class_name = det['class_name']

            color = self._get_color(det['class_id'])
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            cv2.rectangle(img_copy, (x1, y1 - text_h - baseline - 5),
                         (x1 + text_w, y1), color, -1)
            cv2.putText(img_copy, label, (x1, y1 - 5),
                       font, font_scale, (255, 255, 255), thickness)

        return img_copy

    def _get_color(self, class_id):
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
        ]
        return colors[class_id % len(colors)]


def main():
    ONNX_MODEL_PATH = "yolov8_bs1_opset17.onnx"
    IMAGE_PATH = "bus.jpg"
    CONF_THRES = 0.25
    NMS_THRES = 0.45

    try:
        detector = YOLOv8Detector(ONNX_MODEL_PATH, CONF_THRES, NMS_THRES)

        print(f"\n开始检测: {IMAGE_PATH}")
        detections, image = detector.detect(IMAGE_PATH)

        print(f"\n检测到 {len(detections)} 个目标:")
        print("-" * 50)
        for i, det in enumerate(detections, 1):
            print(f"{i}. {det['class_name']}: {det['confidence']:.3f}")
            print(f"   Box: [{det['box'][0]:.1f}, {det['box'][1]:.1f}, "
                  f"{det['box'][2]:.1f}, {det['box'][3]:.1f}]")

        result_image = detector.draw_detections(image, detections)

        output_path = "detection_result.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"\n✅ 结果已保存: {os.path.abspath(output_path)}")

        try:
            cv2.imshow("Result", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            pass

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
