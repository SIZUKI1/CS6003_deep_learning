import sys

task1_tex = r"""\documentclass[12pt]{article}
\usepackage{hyperref}
\usepackage[UTF8]{ctex}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{amsmath, amssymb}
\usepackage{booktabs}
\usepackage{float}
\usepackage{subcaption}
\usepackage{array}
\usepackage{multirow}

\geometry{a4paper, margin=2.5cm}

\title{深度学习课程大作业实验报告}
\author{}
\date{\today}

\begin{document}

\maketitle

\section{任务1：Flower102 数据集 CNN 微调实验}

\subsection{实验概述}
本实验针对 Oxford 102 Category Flower Dataset 完成 102 类花卉图像分类任务。实验以在 ImageNet 上预训练的 ResNet-18 为主干网络，将原始分类层替换为 102 类输出层，并采用“主干网络较小学习率、新分类头较大学习率”的微调策略。围绕任务要求，实验共包含三类对比：
\begin{enumerate}
    \item \textbf{Baseline 与超参数分析：}使用 ImageNet 预训练 ResNet-18，比较不同 backbone learning rate 与 head learning rate 组合对收敛速度和分类性能的影响；
    \item \textbf{预训练消融实验：}使用相同的 ResNet-18 结构，但不加载 ImageNet 预训练参数，从随机初始化开始训练，用于分析预训练迁移学习的作用；
    \item \textbf{注意力机制实验：}在 Baseline 的基础上引入 SE-block 与 CBAM 注意力模块，比较注意力机制对 Accuracy 和 mAP 的影响。
\end{enumerate}

实验过程中使用 wandb 记录训练日志，并根据导出的日志绘制训练集/验证集 loss 曲线以及验证集 Accuracy/mAP 曲线。评价指标包括分类准确率 Accuracy 与多类别 mean Average Precision (mAP)。

\subsection{数据集介绍}
本实验使用 Oxford 102 Category Flower Dataset。该数据集包含 102 个花卉类别，总计 8189 张图像。数据集中不同类别的图像数量不完全相同，并且图像存在尺度、姿态、光照和类内差异等变化，因此属于典型的细粒度图像分类任务。与普通分类任务相比，花卉类别之间往往具有相似的颜色和形状特征，因此模型不仅需要学习低层纹理和颜色信息，还需要捕捉更具判别性的局部结构。

实验使用官方给定的数据划分文件 \texttt{setid.mat}，并根据 \texttt{imagelabels.mat} 读取类别标签。具体划分如表~\ref{tab:task1_data_split} 所示。

\begin{table}[H]
\centering
\caption{数据集划分}
\label{tab:task1_data_split}
\begin{tabular}{lccc}
\toprule
数据划分 & 样本数 & 每轮 batch size & 每轮迭代次数 \\
\midrule
训练集 & 1020 & 32 & $\lceil 1020/32\rceil=32$ \\
验证集 & 1020 & 32 & $\lceil 1020/32\rceil=32$ \\
测试集 & 6149 & 32 & $\lceil 6149/32\rceil=193$ \\
\bottomrule
\end{tabular}
\end{table}

由于训练集每类仅约 10 张图像，样本规模较小，因此直接从随机初始化训练深层 CNN 容易出现收敛慢、泛化不足的问题。ImageNet 预训练模型已经学习到边缘、纹理、颜色组合、局部形状等通用视觉特征，因此更适合作为本任务的初始化参数。

\subsubsection{数据预处理与增强}
训练阶段对输入图像进行随机增强，以提高模型的泛化能力。主要处理包括：
\begin{itemize}
    \item 将图像缩放到较大尺寸后进行 \texttt{RandomResizedCrop}，最终输入尺寸为 $224\times224$；
    \item 使用随机水平翻转和颜色扰动增强样本多样性；
    \item 使用 ImageNet 均值和标准差进行归一化，以匹配预训练模型的输入分布。
\end{itemize}
验证和测试阶段不使用随机增强，仅采用 resize、center crop 和归一化，保证评估结果稳定。

\subsection{模型结构}
\subsubsection{Baseline：ResNet-18 微调}
Baseline 模型采用 ResNet-18。ResNet 的核心思想是通过残差连接学习
\[
    y = F(x) + x,
\]
其中 $x$ 为输入特征，$F(x)$ 为若干卷积层学习到的残差映射。残差连接缓解了深层网络训练中的梯度消失问题，使得网络更容易优化。

本实验加载 ImageNet 预训练的 ResNet-18 参数，并将最后的全连接层由原始 ImageNet 的 1000 类输出替换为 102 类输出：
\[
    \mathrm{FC}_{1000} \rightarrow \mathrm{FC}_{102}.
\]
新的输出层从随机初始化开始训练；其余卷积主干参数使用较小学习率微调。这种设置的直观含义是：主干网络保留 ImageNet 上学到的通用视觉特征，而分类头快速适配花卉数据集的 102 个类别。

\subsubsection{SE-ResNet-18}
SE-block（Squeeze-and-Excitation）是一种通道注意力机制。对于输入特征 $X\in\mathbb{R}^{C\times H\times W}$，首先通过全局平均池化得到每个通道的全局响应：
\[
    z_c=\frac{1}{H W}\sum_{i=1}^{H}\sum_{j=1}^{W}X_c(i,j).
\]
随后经过两层全连接网络和 Sigmoid 函数生成通道权重 $s_c$，并对原特征进行重标定：
\[
    \widetilde{X}_c=s_c X_c.
\]
直观来说，SE 模块会自动判断“哪些通道更重要”。例如在花卉识别中，某些通道可能更关注花瓣颜色，某些通道关注花蕊纹理，SE 模块可以增强更有判别力的通道。

\subsubsection{CBAM-ResNet-18}
CBAM（Convolutional Block Attention Module）包含通道注意力和空间注意力两个部分。通道注意力用于判断“哪些特征通道重要”，空间注意力用于判断“图像中哪些位置重要”。其基本形式可写为：
\[
    X' = M_c(X)\otimes X,
\]
\[
    X'' = M_s(X')\otimes X',
\]
其中 $M_c$ 表示通道注意力，$M_s$ 表示空间注意力，$\otimes$ 表示逐元素乘法。相比 SE，CBAM 结构更复杂，理论上可以同时增强关键通道和关键空间区域，但也会增加训练难度和过拟合风险。

\subsection{实验设置}
所有实验均训练 30 个 epoch，batch size 为 32，输入图像大小为 $224\times224$。优化器采用 AdamW，权重衰减为 $10^{-4}$，损失函数采用交叉熵损失：
\[
    \mathcal{L}=-\sum_{k=1}^{102} y_k\log p_k,
\]
其中 $y_k$ 为 one-hot 标签，$p_k$ 为模型预测该类别的概率。学习率采用余弦退火策略逐步衰减。

评价指标包括：
\begin{itemize}
    \item \textbf{Accuracy：}预测类别与真实类别一致的样本比例；
    \item \textbf{mAP：}对 102 个类别分别计算 Average Precision，再取宏平均，用于衡量模型对各类别置信度排序的整体质量。
\end{itemize}

各组实验设置如表~\ref{tab:task1_config} 所示。

\begin{table}[H]
\centering
\caption{实验设置对比}
\label{tab:task1_config}
\begin{tabular}{lccccc}
\toprule
实验名称 & 模型 & 预训练 & Backbone LR & Head LR & Epoch \\
\midrule
Baseline-A & ResNet-18 & ImageNet & $3\times10^{-5}$ & $3\times10^{-4}$ & 30 \\
Baseline-B & ResNet-18 & ImageNet & $1\times10^{-5}$ & $1\times10^{-3}$ & 30 \\
Baseline-C & ResNet-18 & ImageNet & $1\times10^{-4}$ & $1\times10^{-3}$ & 30 \\
SE-ResNet-18 & ResNet-18+SE & ImageNet & $1\times10^{-4}$ & $1\times10^{-3}$ & 30 \\
CBAM-ResNet-18 & ResNet-18+CBAM & ImageNet & $1\times10^{-4}$ & $1\times10^{-3}$ & 30 \\
Random Init & ResNet-18 & 无 & $1\times10^{-3}$ & $1\times10^{-3}$ & 30 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{实验结果}
表~\ref{tab:task1_result} 给出了六组实验在验证集和测试集上的核心结果。最佳模型根据验证集 Accuracy 选择，并在测试集上进行最终评估。

\begin{table}[H]
\centering
\caption{实验结果对比}
\label{tab:task1_result}
\begin{tabular}{lcccc}
\toprule
实验名称 & 最佳 Epoch & 最佳 Val Acc & Test Acc & Test mAP \\
\midrule
Baseline-A ($3e^{-5}/3e^{-4}$) & 27 & 0.9088 & 0.8762 & 0.9409 \\
Baseline-B ($1e^{-5}/1e^{-3}$) & 22 & 0.8990 & 0.8725 & 0.9402 \\
\textbf{Baseline-C ($1e^{-4}/1e^{-3}$)} & \textbf{22} & \textbf{0.9304} & \textbf{0.8949} & \textbf{0.9550} \\
SE-ResNet-18 & 14 & 0.9176 & 0.8824 & 0.9447 \\
CBAM-ResNet-18 & 24 & 0.8716 & 0.8341 & 0.9056 \\
Random Init & 25 & 0.4225 & 0.3420 & 0.3791 \\
\bottomrule
\end{tabular}
\end{table}

从表~\ref{tab:task1_result} 可以看出，综合测试集 Accuracy 和 mAP，最佳结果来自 Baseline-C，即 backbone learning rate 为 $1\times10^{-4}$、head learning rate 为 $1\times10^{-3}$ 的 ImageNet 预训练 ResNet-18。该模型测试集 Accuracy 达到 0.8949，测试集 mAP 达到 0.9550。与随机初始化模型相比，测试集 Accuracy 提升约 55.29 个百分点，mAP 提升约 0.5759，说明 ImageNet 预训练对小样本花卉分类任务具有显著帮助。

\subsection{训练曲线可视化与分析}
本节结合 wandb 记录结果导出的曲线，对训练过程进行分析。每组实验均包含训练集和验证集 loss 曲线，以及训练集 Accuracy、验证集 Accuracy 和验证集 mAP 曲线。

\subsubsection{不同学习率 Baseline 对比}
图~\ref{fig:task1_baseline_curves} 展示了三组预训练 ResNet-18 在不同学习率组合下的训练曲线。

\begin{figure}[H]
\centering
\begin{subfigure}{0.47\textwidth}
    \includegraphics[width=\linewidth]{figures/loss_baseline_bb3e-5_head3e-4.png}
    \caption{Baseline-A Loss}
\end{subfigure}
\hfill
\begin{subfigure}{0.47\textwidth}
    \includegraphics[width=\linewidth]{figures/accmap_baseline_bb3e-5_head3e-4.png}
    \caption{Baseline-A Acc/mAP}
\end{subfigure}

\begin{subfigure}{0.47\textwidth}
    \includegraphics[width=\linewidth]{figures/loss_baseline_bb1e-5_head1e-3.png}
    \caption{Baseline-B Loss}
\end{subfigure}
\hfill
\begin{subfigure}{0.47\textwidth}
    \includegraphics[width=\linewidth]{figures/accmap_baseline_bb1e-5_head1e-3.png}
    \caption{Baseline-B Acc/mAP}
\end{subfigure}

\begin{subfigure}{0.47\textwidth}
    \includegraphics[width=\linewidth]{figures/loss_baseline_bb1e-4_head1e-3.png}
    \caption{Baseline-C Loss}
\end{subfigure}
\hfill
\begin{subfigure}{0.47\textwidth}
    \includegraphics[width=\linewidth]{figures/accmap_baseline_bb1e-4_head1e-3.png}
    \caption{Baseline-C Acc/mAP}
\end{subfigure}
\caption{不同学习率组合下的 ResNet-18 Baseline 曲线}
\label{fig:task1_baseline_curves}
\end{figure}

三组 Baseline 的训练 loss 都快速下降，说明 ImageNet 预训练特征能够较快适应花卉分类任务。Baseline-A 的 backbone 和 head 学习率都较小，训练过程较稳定，但验证集 Accuracy 最终约在 0.90 附近波动，测试集 Accuracy 为 0.8762。Baseline-B 使用更小的 backbone 学习率 $1\times10^{-5}$，虽然 head 学习率较大，但主干网络更新幅度过小，导致模型对花卉数据集的适配能力不足，最终测试集 Accuracy 为 0.8725，略低于 Baseline-A。

Baseline-C 的 backbone 学习率提升至 $1\times10^{-4}$，head 学习率保持 $1\times10^{-3}$。从曲线看，该设置前几个 epoch 收敛最快，验证集 Accuracy 很快超过 0.90，并在后期稳定在 0.92--0.93 左右。其验证 loss 也低于另外两组 Baseline，最终取得最高的测试集 Accuracy 和 mAP。这表明，在本任务中，backbone 不能完全冻结或更新过慢；适当增大 backbone 学习率可以让预训练特征更好地迁移到细粒度花卉分类任务。

同时，三组 Baseline 均出现训练 Accuracy 接近 1.0 而验证 Accuracy 不再明显提升的现象，说明模型在后期已经基本拟合训练集。由于训练集只有 1020 张图像，后续若进一步提升性能，可以考虑更强的数据增强、label smoothing、mixup/cutmix 或 early stopping。

\subsubsection{注意力机制对比}
图~\ref{fig:task1_attention_curves} 展示了 SE-ResNet-18 与 CBAM-ResNet-18 的训练曲线。

\begin{figure}[H]
\centering
\begin{subfigure}{0.47\textwidth}
    \includegraphics[width=\linewidth]{figures/loss_se_resnet18.png}
    \caption{SE-ResNet-18 Loss}
\end{subfigure}
\hfill
\begin{subfigure}{0.47\textwidth}
    \includegraphics[width=\linewidth]{figures/accmap_se_resnet18.png}
    \caption{SE-ResNet-18 Acc/mAP}
\end{subfigure}

\begin{subfigure}{0.47\textwidth}
    \includegraphics[width=\linewidth]{figures/loss_cbam_resnet18.png}
    \caption{CBAM-ResNet-18 Loss}
\end{subfigure}
\hfill
\begin{subfigure}{0.47\textwidth}
    \includegraphics[width=\linewidth]{figures/accmap_cbam_resnet18.png}
    \caption{CBAM-ResNet-18 Acc/mAP}
\end{subfigure}
\caption{注意力机制模型训练曲线}
\label{fig:task1_attention_curves}
\end{figure}

SE-ResNet-18 的收敛速度较快，验证集 Accuracy 在前 10 个 epoch 内已经达到较高水平，并在第 14 个 epoch 取得最佳验证集 Accuracy 0.9176。其测试集 Accuracy 为 0.8824，mAP 为 0.9447，明显优于随机初始化模型，也优于 CBAM 模型。但与最佳 Baseline-C 相比，SE-ResNet-18 的测试集 Accuracy 低约 1.25 个百分点。这说明 SE 通道注意力能够增强通道选择能力，但在当前训练设置下，并没有超过调参后的 ResNet-18 Baseline。

CBAM-ResNet-18 的训练曲线显示，其早期收敛明显慢于 SE 和 Baseline-C。尤其在第 1 个 epoch 附近，验证 Accuracy 很低，随后逐渐恢复并提升到 0.87 左右。最终 CBAM 测试集 Accuracy 为 0.8341，mAP 为 0.9056，低于 SE 和所有预训练 Baseline。可能原因是 CBAM 同时引入通道注意力和空间注意力，结构更复杂，会改变预训练残差特征的分布；在训练集较小且未使用 warmup 的情况下，模型更难稳定优化。对于 CBAM，后续可以尝试降低注意力模块学习率、增加 warmup、冻结前几层，或延长训练轮数。

\subsubsection{预训练消融实验}
图~\ref{fig:task1_random_curves} 展示了从随机初始化开始训练的 ResNet-18 曲线。

\begin{figure}[H]
\centering
\begin{subfigure}{0.47\textwidth}
    \includegraphics[width=\linewidth]{figures/loss_random_resnet18.png}
    \caption{Random Init Loss}
\end{subfigure}
\hfill
\begin{subfigure}{0.47\textwidth}
    \includegraphics[width=\linewidth]{figures/accmap_random_resnet18.png}
    \caption{Random Init Acc/mAP}
\end{subfigure}
\caption{随机初始化 ResNet-18 训练曲线}
\label{fig:task1_random_curves}
\end{figure}

随机初始化模型的曲线与预训练模型差异非常明显。其训练 loss 虽然逐步下降，但下降速度远慢于预训练模型；验证 loss 长期保持在较高水平，并且 Accuracy/mAP 上升较慢。最终随机初始化模型的测试集 Accuracy 仅为 0.3420，mAP 为 0.3791。该结果说明，在只有 1020 张训练图片的情况下，模型很难从零学习到稳定且泛化良好的视觉特征。

与随机初始化相比，最佳预训练 Baseline-C 的测试集 Accuracy 从 0.3420 提升到 0.8949，提升约 55.29 个百分点。这一消融实验充分说明：对于小样本细粒度分类任务，ImageNet 预训练提供的通用视觉表征是模型性能提升的关键因素。

\subsection{总结分析}
\subsubsection{学习率的影响}
三组 Baseline 表明，学习率组合直接影响微调效果。若 backbone 学习率过小，模型主干特征更新不足，难以充分适配花卉类别；若学习率适中，则既能保留 ImageNet 预训练知识，又能根据目标数据集调整高层语义特征。本实验中 $1\times10^{-4}/1\times10^{-3}$ 是最优组合，说明新分类头需要较快学习，而主干网络也需要一定幅度的微调。

\subsubsection{注意力模块的影响}
SE 和 CBAM 的结果说明，注意力机制并不一定在所有设置下都能带来提升。SE 模块结构较轻量，只对通道进行重标定，因此对预训练 ResNet 的扰动较小，最终效果接近 Baseline。CBAM 额外加入空间注意力，表达能力更强，但在小训练集上可能更难优化，也更容易受到学习率、初始化和正则化策略影响。因此，在本实验条件下，直接加入 CBAM 并未获得性能提升。

\subsubsection{预训练的作用}
随机初始化模型与预训练模型的差距最大。预训练模型在前几个 epoch 内即可取得较高验证 Accuracy，而随机初始化模型到第 30 个 epoch 仍未达到预训练模型早期水平。这说明 ImageNet 预训练不仅提高了最终性能，也显著加快了收敛速度。

\subsubsection{结论}
本实验完成了基于 ImageNet 预训练 CNN 的 Oxford 102 类花卉识别任务，并比较了不同学习率、预训练消融和注意力机制的影响。主要结论如下：
\begin{enumerate}
    \item 最优模型为 ImageNet 预训练 ResNet-18，学习率组合为 backbone $1\times10^{-4}$、head $1\times10^{-3}$，测试集 Accuracy 为 0.8949，测试集 mAP 为 0.9550；
    \item 预训练带来的提升非常显著。与随机初始化相比，最佳预训练模型测试集 Accuracy 提升约 55.29 个百分点，说明迁移学习非常适合小样本细粒度分类任务；
    \item SE 注意力模块取得了较好的结果，但未超过最佳 Baseline；CBAM 在当前设置下表现较弱，可能需要更细致的学习率、warmup 或正则化策略；
    \item 从训练曲线看，预训练模型收敛快，训练 loss 很快接近 0，而验证集指标在中后期趋于平台，说明后续可以通过 early stopping、更强数据增强和正则化进一步提升泛化能力。
\end{enumerate}

"""

task1_bib = r"""
\begin{thebibliography}{9}
\bibitem{flowers102}
Nilsback, M.-E. and Zisserman, A. \textit{Automated Flower Classification over a Large Number of Classes}. Indian Conference on Computer Vision, Graphics and Image Processing, 2008.
\bibitem{resnet}
He, K., Zhang, X., Ren, S., and Sun, J. \textit{Deep Residual Learning for Image Recognition}. CVPR, 2016.
\bibitem{se}
Hu, J., Shen, L., and Sun, G. \textit{Squeeze-and-Excitation Networks}. CVPR, 2018.
\bibitem{cbam}
Woo, S., Park, J., Lee, J.-Y., and Kweon, I. S. \textit{CBAM: Convolutional Block Attention Module}. ECCV, 2018.
\end{thebibliography}

\end{document}
"""

with open("/mnt/miah209/yili126/CS6003_deep_learning/HW2/task2/report_task2.tex", "r") as f:
    task23 = f.read()

# combine all
full = task1_tex + "\n" + task23 + "\n" + task1_bib

with open("/mnt/miah209/yili126/CS6003_deep_learning/HW2/task2/report_task2.tex", "w") as f:
    f.write(full)

print("Successfully merged task1, task2, and task3 into report_task2.tex")
