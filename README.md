# EuroSAT Land Cover Classification (3-Layer MLP from Scratch)

本项目是 CS6003 深度学习与空间智能课程的作业实现。项目完全基于 NumPy 实现了三层多层感知机（MLP），包含手动推导的反向传播机制，用于 EuroSAT 遥感图像的 10 分类任务。

## 环境依赖
项目运行环境建议为 Python 3.8+。核心依赖如下：
- `numpy` (版本建议: 1.21+)
- `Pillow` (用于图像读取)
- `scikit-learn` (仅用于数据集划分与评价指标计算)
- `matplotlib` & `scipy` (用于结果可视化与权重平滑)

安装指令：
```bash
pip install numpy pillow scikit-learn matplotlib scipy
```

## 数据集配置
本项目使用 [EuroSAT RGB](https://github.com/phelber/EuroSAT) 数据集。
1. 下载并解压数据集，确保目录结构如下：
   ```
   EuroSAT_RGB/
   ├── AnnualCrop/
   ├── Forest/
   ├── ... (10个类别文件夹)
   ```
2. 运行脚本时，确保 `data_dir` 指向该文件夹。

## 快速启动

### 1. 训练模型
你可以直接运行默认参数进行训练：
```bash
python3 train.py --epochs 30 --lr 0.01 --hidden_dim 512 --weight_decay 0.001
```

### 2. 参数搜索 (Grid Search)
自动化搜索最佳超参数组合：
```bash
python3 hyperparam_search.py
```

### 3. 测试与可视化 (最重要)
如果你已经有了训练好的权重文件（如 `final_best_model.pkl`），运行以下指令生成报告所需的全部图表：
```bash
python3 eval.py --model final_best_model.pkl
```
运行后将生成：
- `learning_curves.png`: 训练与验证曲线
- `confusion_matrix.png`: 混淆矩阵热力图
- `weight_visualization.png`: 第一层权重可视化图
- `bad_cases/`: 错例样本文件夹