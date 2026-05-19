# YOLOv8 Traffic Sign Detection

基于 YOLOv8n 的交通标志识别模型，提供两个版本：

| 版本 | 类别数 | 脚本前缀 | mAP50 | mAP50-95 |
|------|--------|----------|-------|----------|
| 完整版 | 10 类 | `prepare_data.py` / `train.py` / `test.py` | 0.988 | 0.988 |
| 精简版 | 4 类 | `prepare_data_4c.py` / `train_4c.py` / `test_4c.py` | 0.991 | 0.969 |

---

## 一、10 类模型

### 类别

| ID | 类别名 | 原始数量 | 增强后 |
|----|--------|---------|--------|
| 0 | Green_forward | 1 | 60 |
| 1 | Green_left | 1 | 60 |
| 2 | Green_right | 1 | 60 |
| 3 | Parking | 1 | 60 |
| 4 | Red_forward | 1 | 60 |
| 5 | Red_left | 1 | 60 |
| 6 | Red_right | 1 | 60 |
| 7 | no_entry | 58 | 60 |
| 8 | turn_left | 8 | 56 |
| 9 | turn_right | 26 | 52 |

### 使用

```bash
python3 src/prepare_data.py      # 准备数据 → dataset/
python3 src/train.py             # 训练模型 → runs/detect/traffic_signs/
python3 src/test.py              # 测试模型（默认测验证集）
python3 src/test.py img.jpg -s   # 测试单张图片并显示
```

### 模型路径

`runs/detect/traffic_signs/weights/best.pt`

---

## 二、4 类模型（推荐）

### 类别

| ID | 类别名 | 原始数量 | 增强后 | mAP50 | mAP50-95 |
|----|--------|---------|--------|-------|----------|
| 0 | Parking | 1 | 60 | 0.995 | 0.908 |
| 1 | no_entry | 58 | 60 | 0.995 | 0.995 |
| 2 | turn_left | 8 | 60 | 0.989 | 0.989 |
| 3 | turn_right | 26 | 60 | 0.984 | 0.984 |

### 使用

```bash
python3 src/prepare_data_4c.py      # 准备数据 → dataset_4c/
python3 src/train_4c.py             # 训练模型 → runs/detect/traffic_signs_4c/
python3 src/test_4c.py              # 测试模型（默认测验证集）
python3 src/test_4c.py img.jpg -s   # 测试单张图片并显示
```

### 模型路径

`runs/detect/traffic_signs_4c/weights/best.pt`

---

## 三、数据增强策略

| 原始图片数 | 增强强度 | 变换内容 |
|-----------|---------|---------|
| ≤10 张 | heavy | 旋转±40°、亮度/对比度、HSV、高斯模糊、噪声、仿射变换（概率混合） |
| >10 张 | light | 旋转±20°、亮度/对比度、HSV（低概率模糊/噪声/仿射） |

增强后按 8:2 分割训练集和验证集。每张图视为完整目标，YOLO 标签为 `class_id 0.5 0.5 1.0 1.0`。

---

## 四、项目结构

```
yolo_opencv2/
├── src/
│   ├── prepare_data.py      # 10类数据准备
│   ├── train.py             # 10类训练
│   ├── test.py              # 10类测试
│   ├── prepare_data_4c.py   # 4类数据准备
│   ├── train_4c.py          # 4类训练
│   └── test_4c.py           # 4类测试
├── image/                   # 原始图片（10个子目录）
├── dataset/                 # 10类数据集
├── dataset_4c/              # 4类数据集
├── runs/
│   ├── detect/traffic_signs/
│   │   └── weights/
│   │       ├── best.pt      # 10类最佳模型
│   │       └── best.onnx
│   └── detect/traffic_signs_4c/
│       └── weights/
│           ├── best.pt      # 4类最佳模型
│           └── best.onnx
└── test_results_4c/         # 4类测试输出
```

---

## 五、ultralytics CLI 直接调用

```bash
# 预测
yolo predict model=runs/detect/traffic_signs_4c/weights/best.pt source=image/Parking/parking.jpg

# 验证
yolo val model=runs/detect/traffic_signs_4c/weights/best.pt data=dataset_4c/data.yaml

# 导出
yolo export model=runs/detect/traffic_signs_4c/weights/best.pt format=onnx
```
