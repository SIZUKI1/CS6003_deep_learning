# Task 2: 场景目标检测与视频多目标跟踪

基于 YOLOv8 的道路车辆检测与多目标跟踪系统。使用 Road Vehicle Images Dataset 微调 YOLOv8s 模型，实现视频流中的车辆检测、多目标跟踪、遮挡分析和越线计数功能。

## 环境配置

### 硬件要求
- GPU: NVIDIA GPU (推荐 RTX 3090 或更高)
- 显存: ≥ 8GB
- 磁盘: ≥ 5GB 可用空间

### 软件依赖
```bash
# Python 3.8+
pip install -r requirements.txt
```

核心依赖：
- `ultralytics>=8.2.0` — YOLOv8 框架
- `swanlab>=0.3.0` — 训练可视化
- `opencv-python>=4.8.0` — 视频处理
- `numpy`, `matplotlib`, `Pillow`, `pyyaml`

## 数据集准备

### 下载 Road Vehicle Images Dataset

**方式一：Kaggle API（推荐）**
```bash
# 1. 配置 Kaggle API
# 从 https://www.kaggle.com/settings 下载 kaggle.json
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 2. 下载数据集
python download_dataset.py
```

**方式二：手动下载**
1. 访问 [Road Vehicle Images Dataset](https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset)
2. 下载并解压到 `data/road_vehicle/` 目录
3. 运行验证脚本：`python download_dataset.py`

**数据集结构要求：**
```
data/road_vehicle/
├── data.yaml          # 自动生成的配置文件
├── train/
│   ├── images/        # 训练图像
│   └── labels/        # YOLO格式标注
└── valid/
    ├── images/        # 验证图像
    └── labels/        # YOLO格式标注
```

### 准备测试视频
将一段 10-30 秒的路口/交通视频放入 `data/test_video/test.mp4`。

## 训练

### 基本训练
```bash
python train.py --epochs 100 --batch 32 --device 0
```

### 训练参数说明
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | yolov8s.pt | 预训练模型 |
| `--data` | 自动发现 | data.yaml 路径 |
| `--epochs` | 100 | 训练轮数 |
| `--batch` | 32 | 批大小 |
| `--imgsz` | 640 | 输入图像尺寸 |
| `--optimizer` | SGD | 优化器 (SGD/Adam/AdamW) |
| `--lr0` | 0.01 | 初始学习率 |
| `--lrf` | 0.01 | 最终学习率比例 |
| `--patience` | 20 | 早停 patience |
| `--device` | 0 | GPU 编号 |
| `--no-swanlab` | False | 禁用 SwanLab |

### 恢复训练
```bash
python train.py --resume
```

### 训练配置详情
- **模型**: YOLOv8s (11.2M 参数)
- **优化器**: SGD (momentum=0.937, weight_decay=0.0005)
- **学习率调度**: Cosine Annealing (lr0=0.01 → lrf=0.0001)
- **损失函数**: CIoU Loss + BCE Loss (YOLOv8 默认)
- **数据增强**: Mosaic, MixUp, HSV 增强, 水平翻转
- **评价指标**: mAP@0.5, mAP@0.5:0.95

训练结果保存在 `runs/detect/yolov8s_road_vehicle/`。

## 推理与测试

### 1. 多目标跟踪
```bash
python track_video.py --source data/test_video/test.mp4
```
输出带有 BoundingBox、类别和 Tracking ID 标注的视频。

### 2. 遮挡与 ID 跳变分析
```bash
python occlusion_analysis.py --source data/test_video/test.mp4
```
自动检测遮挡场景，提取连续帧并生成对比分析图。

### 3. 越线计数
```bash
python line_crossing.py --source data/test_video/test.mp4 --line-y 0.5
```
在画面中设定虚拟线，统计越线车辆数量。

### 4. 完整管线（推荐）
```bash
python inference_pipeline.py --source data/test_video/test.mp4 --line-y 0.5
```
一次运行完成所有任务，输出：
- `outputs/tracking_output.mp4` — 跟踪视频
- `outputs/counting_output.mp4` — 越线计数视频
- `outputs/occlusion_frames/` — 遮挡分析帧
- `outputs/full_report.json` — 完整报告

## 模型权重

- 训练好的模型权重: [百度网盘/Google Drive 链接]（待填写）
- 预训练模型: yolov8s.pt (COCO pretrained)

## 项目文件说明

| 文件 | 说明 |
|------|------|
| `train.py` | 训练脚本，集成 SwanLab |
| `track_video.py` | 视频多目标跟踪 |
| `occlusion_analysis.py` | 遮挡与 ID 跳变分析 |
| `line_crossing.py` | 越线计数 |
| `inference_pipeline.py` | 完整推理管线 |
| `download_dataset.py` | 数据集下载与准备 |
| `requirements.txt` | 依赖列表 |

## 技术细节

### YOLOv8 架构
- Backbone: CSPDarknet53 with C2f modules
- Neck: PANet (Path Aggregation Network)
- Head: Decoupled head (分离的分类和回归头)
- Anchor-free 设计

### 多目标跟踪
- 默认使用 ByteTrack 算法
- 也支持 BoT-SORT (带 ReID)
- 基于卡尔曼滤波预测 + 匈牙利算法匹配

### 越线计数原理
利用 Tracking ID 的连续性，记录每个目标的检测框中心坐标，判断中心点是否从虚拟线的一侧穿越到另一侧。
