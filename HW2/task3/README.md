# U-Net 图像分割训练

基于 PyTorch 从零实现 U-Net 网络，在 Stanford Background Dataset 上对比三种损失函数（Cross-Entropy Loss、Dice Loss、CE+Dice Loss）的分割效果。

## 环境依赖

### 硬件要求
- NVIDIA GPU（推荐，CPU 也可运行但速度较慢）
- 显存建议 ≥ 4GB（batch_size=4 时约占用 2-3GB）

### 软件环境

推荐使用 conda 创建虚拟环境：

```bash
conda create -n unet_seg python=3.8
conda activate unet_seg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install tqdm
pip install segmentation-models-pytorch