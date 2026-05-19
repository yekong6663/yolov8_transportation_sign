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

| 格式 | 路径 |
|------|------|
| PyTorch | `runs/detect/traffic_signs/weights/best.pt` |
| ONNX | `runs/detect/traffic_signs/weights/best.onnx` |
| K230 kmodel | `runs/detect/traffic_signs/weights/best.kmodel` |

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

| 格式 | 路径 |
|------|------|
| PyTorch | `runs/detect/traffic_signs_4c/weights/best.pt` |
| ONNX | `runs/detect/traffic_signs_4c/weights/best.onnx` |
| K230 kmodel | `runs/detect/traffic_signs_4c/weights/best.kmodel` |

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
│   ├── test_4c.py           # 4类测试
│   └── export_kmodel.py     # ONNX→kmodel 转换
├── image/                   # 原始图片（10个子目录）
├── dataset/                 # 10类数据集
├── dataset_4c/              # 4类数据集
├── runs/
│   ├── detect/traffic_signs/
│   │   └── weights/
│   │       ├── best.pt      # 10类最佳模型
│   │       ├── best.onnx
│   │       └── best.kmodel   # K230 kmodel
│   └── detect/traffic_signs_4c/
│       └── weights/
│           ├── best.pt      # 4类最佳模型
│           ├── best.onnx
│           └── best.kmodel   # K230 kmodel
├── tmp/                     # nncase 编译中间文件（可删除）
└── test_results_4c/         # 4类测试输出
```

---

## 五、导出 K230 kmodel

将训练好的 ONNX 模型转换为 K230 芯片使用的 kmodel 格式。

### 环境准备

nncase 编译器基于 .NET 构建，需要先安装 .NET 运行时：

```bash
# 1. 安装 .NET Runtime 7.0（nncase 2.11 依赖）
wget https://dotnetcli.azureedge.net/dotnet/Runtime/7.0.20/dotnet-runtime-7.0.20-linux-x64.tar.gz
mkdir -p /usr/local/dotnet
tar -xzf dotnet-runtime-7.0.20-linux-x64.tar.gz -C /usr/local/dotnet

# 2. 安装 nncase 编译器（需要 Python 3.10）
pip install nncase==2.11.0 nncase-kpu==2.11.0

# 3. 设置环境变量（建议写入 ~/.bashrc）
export DOTNET_ROOT=/usr/local/dotnet
export NNCASE_PLUGIN_PATH=$(python3 -c "import nncase; print(__import__('pathlib').Path(nncase.__file__).parent / 'modules')")
```

> **注意**：如果 `pip install nncase` 失败，可从 [nncase GitHub Releases](https://github.com/kendryte/nncase/releases) 下载对应平台的 `.whl` 手动安装。

### 转换命令

```bash
# 转换 4 类模型（推荐，自动使用训练集图片做量化校准）
python3 src/export_kmodel.py --preset 4c

# 转换 10 类模型
python3 src/export_kmodel.py --preset 10c

# 手动指定路径
python3 src/export_kmodel.py \
    --onnx runs/detect/traffic_signs_4c/weights/best.onnx \
    --output runs/detect/traffic_signs_4c/weights/best.kmodel \
    --dataset dataset_4c/images/train

# 不使用量化（float32 模型，精度高但推理慢）
python3 src/export_kmodel.py --preset 4c --no-quantize

# 修改输入尺寸
python3 src/export_kmodel.py --preset 4c --input-size 320 320
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--onnx` | 输入 ONNX 模型路径 | 必填（使用 `--preset` 时可选） |
| `--output` | 输出 kmodel 路径 | 与 ONNX 同目录，后缀 .kmodel |
| `--dataset` | 校准图片目录（int8 量化用） | 无 |
| `--input-size` | 模型输入尺寸 H W | 640 640 |
| `--target` | 目标芯片 | k230 |
| `--no-quantize` | 禁用 int8 量化 | False |
| `--preset` | 快捷预设：4c 或 10c | 无 |

### 转换输出

| 模型 | kmodel 路径 | 大小 | 量化 |
|------|------------|------|------|
| 4 类 | `runs/detect/traffic_signs_4c/weights/best.kmodel` | 3.2 MB | int8 |
| 10 类 | `runs/detect/traffic_signs/weights/best.kmodel` | 3.2 MB | int8 |

生成的 `.kmodel` 文件可直接部署到 K230 板卡上运行。

转换过程中会在项目根目录生成 `tmp/` 文件夹，存放 nncase 编译器各阶段的中间输出（IR dump、量化日志、目标代码等），用于调试编译过程。该目录可随时删除，建议加入 `.gitignore`：

```bash
echo "tmp/" >> .gitignore
```

---
## 六、ultralytics CLI 直接调用

```bash
# 预测
yolo predict model=runs/detect/traffic_signs_4c/weights/best.pt source=image/Parking/parking.jpg

# 验证
yolo val model=runs/detect/traffic_signs_4c/weights/best.pt data=dataset_4c/data.yaml

# 导出
yolo export model=runs/detect/traffic_signs_4c/weights/best.pt format=onnx
```
