# Oxford 102 Flowers 微调实验

本项目用于完成：在 ImageNet 预训练 CNN 上微调 Oxford 102 Category Flower Dataset，并完成超参数分析、预训练消融、注意力机制对比，以及 wandb/swanlab 曲线可视化。


## 一、下载文件

分类任务需要这三个文件：

1. `102flowers.tgz`：原始图片
2. `imagelabels.mat`：每张图片对应的 102 类标签
3. `setid.mat`：官方 train / validation / test 划分

## 二、环境安装

```bash
pip install -r requirements.txt
```

## 三、下载并解压数据

```bash
python download_data.py --data_root ./data/flowers102
```

下载后目录如下：

```text
data/flowers102/
├── 102flowers.tgz
├── imagelabels.mat
├── setid.mat
└── jpg/
    ├── image_00001.jpg
    ├── image_00002.jpg
    └── ...
```

## 四、运行所有实验

AutoDL：

```bash
bash run_experiments.sh
```

## 五、输出文件说明

每次运行会在 `outputs/运行名/` 下生成：

```text
outputs/xxx/
├── config.json          # 本次实验参数
├── metrics.csv          # 每个 epoch 的 train/val loss、val accuracy、val mAP
├── best.pt              # 验证集 accuracy 最好的模型
├── last.pt              # 最后一轮模型
└── test_metrics.json    # 使用 best.pt 在 test set 上的最终结果
```

## 六、本地画图

```bash
python plot_metrics.py --run_dir ./outputs/baseline_resnet18
```

会生成：

```text
loss_curve.png
val_acc_map_curve.png
```
