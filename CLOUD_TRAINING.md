# LoveDA ResNet50 云端重跑说明

这个仓库是独立于原项目的云端实验归档，记录了本地 RTX 4060 8GB 上因显存不足
而失败的完整 ResNet50 + U-Net 3+ 实验，以及随后在 NVIDIA A10 上完成的正式训练。

## 仓库包含什么

- ResNet50 ImageNet预训练编码器
- U-Net 3+ Full-scale Skip Connections
- U-Net 3+ Deep Supervision
- 512×512随机裁块
- CE + Dice
- SGD + poly学习率
- 原始 `15000` optimizer updates 设计，以及实际完成的 `25200` updates 记录
- 16GB与24GB云GPU配置
- LoveDA数据集Release分卷下载脚本
- 本地OOM记录和后续小显存试验指标快照

最终云端实验的 `best_model.pth`、`last_checkpoint.pth`、指标、曲线和预测样本已经
归档；`.pth` 文件通过 Git LFS 管理。更早的本地历史 checkpoint 和 LoveDA 数据集
不在仓库中。

## 实际完成的最终训练

最终结论只来自 `runs/cloud_resnet50_24gb_eff4_40ep/`。该目录中的
`config.json`、`metrics.csv`、`summary.json` 和 `confusion_matrix.csv` 是实验原始记录：

- 环境：NVIDIA A10，PyTorch `2.3.1+cu121`，CUDA runtime `12.1`
- 模型：ImageNet 预训练 ResNet50 Encoder + U-Net 3+ Decoder
- 结构：完整 Full-scale Skip Connections 和 Deep Supervision，`cat_channels=64`
- stem：`standard_resnet_stem=false`，预训练权重加载后将 `conv1.stride` 改为 1
- 归一化：Encoder FrozenBatchNorm，Decoder GroupNorm
- 数据：LoveDA 七类，512×512 crop，单尺度 `1.0`
- Loss：CE + Dice；Focal 和 class-aware crop 均关闭
- 优化：SGD，momentum `0.9`，weight decay `1e-4`
- 调度：Poly，power `0.9`
- batch：physical `2`，accumulation `2`，effective `4`
- 学习率：按 `0.01 × 4 / 16` 线性缩放，实际为 `0.0025`
- 训练长度：`25200` optimizer updates，覆盖约 40 个数据轮次
- runtime：AMP 和 channels-last 开启
- Full Validation：每 8 个 epoch 一次，单尺度评估

减少 accumulation 是为了降低串行 micro-batch 带来的训练时间；effective batch 从
原计划的 16 改为 4 后，学习率同步线性缩小，并把 optimizer updates 增加到 25200。
最佳结果出现在 epoch 24：mIoU `0.4867344561`、mean Dice `0.6502697252`、
Val loss `1.7310836376`。训练在 epoch 41 / 25200 updates 结束，最终 mIoU 为
`0.4764585641`。使用 `best_model.pth` 生成的 LoveDA 单尺度 Test 提交取得隐藏测试
mIoU `0.469813`；仓库未记录该次提交 ID。

`runs/cloud_resnet50_24gb/` 仅保存原始计划。它的 `metrics.csv` 只有表头，未形成
完整 epoch 指标，因此不能作为最终 ResNet50 实验。该目录和
`configs/cloud/resnet50_24gb.yaml` 中的 `batch 2 + accumulation 8 = effective batch 16`、
`lr=0.01`、`15000 updates` 均应按“原始计划配置”理解。

## 原始云端方案与复现入口

    git clone https://github.com/KXCC-cc/loveda-unet3plus-resnet50-cloud-final.git
    cd loveda-unet3plus-resnet50-cloud-final
    bash scripts/cloud_prepare.sh

正式脚本会先执行 CUDA 与数据集预检。只有检测到 GPU，且 Train/Val 分别为
2522/1669 张并通过 image/mask 文件名核对后才会进入训练。

16GB GPU先运行：

    bash scripts/cloud_train_resnet50_16gb.sh

24GB及以上GPU可运行：

    bash scripts/cloud_train_resnet50_24gb.sh

这条命令是原设计阶段的第一轮 baseline 入口。它保留用于追溯配置历史，但并非
已经完成的最终实验入口。

在原计划中，若 24GB 配置 OOM，则改用 16GB 配置，并保持 crop 和 cat_channels
不变。两套原始云配置都设计为 effective batch 16、`lr=0.01`；这些数值不代表
后来实际完成的 eff4_40ep 实验。

## 输出位置

训练结果保存在：

    runs/cloud_resnet50_16gb/
    runs/cloud_resnet50_24gb/
    runs/cloud_resnet50_24gb_eff4_40ep/

输出可包括 best_model.pth、last_checkpoint.pth、metrics.csv、summary.json、曲线、
预测样本和混淆矩阵。只有 `cloud_resnet50_24gb_eff4_40ep/` 形成了本仓库所归档的
完整正式结果；`cloud_resnet50_24gb/metrics.csv` 只有表头。

## 原始 40GB 及以上 GPU 设计

A100 40GB等更大显存GPU可减少串行梯度累积：

    bash scripts/cloud_train_resnet50_40gb.sh

原计划的三套配置都保持 effective batch 16。由于 batch=1 + accumulation=16 会
执行更多串行前后向计算，16GB 配置最省显存但最慢。这一时间成本正是最终实验改用
effective batch 4 的原因之一。

原计划中每 8 个 epoch 约对应 1260 次 optimizer update；实际 eff4_40ep 配置的
`updates_per_epoch=630`，仍每 8 个 epoch 验证一次。两者不能混用。

无论本轮是否执行 Full Validation，每个完整 epoch 结束都会原子更新
`runs/<run>/last_checkpoint.pth`。非验证 epoch 只保存恢复点，不额外运行验证。
云实例中断后继续同一个 24GB baseline：

```bash
bash scripts/cloud_train_resnet50_24gb.sh \
  --resume runs/cloud_resnet50_24gb/last_checkpoint.pth
```

恢复会从下一个 epoch 开始，并恢复 optimizer update、poly scheduler、AMP scaler、
best mIoU 与随机状态。

## 可选受控消融（未形成正式完整结果）

以下脚本已实现，可在完成 baseline 后按顺序运行，但本仓库没有足以证明它们提升
性能的完整正式实验结果：

```bash
bash scripts/cloud_train_resnet50_24gb_diff_lr.sh
bash scripts/cloud_train_resnet50_24gb_classaware.sh
bash scripts/cloud_train_resnet50_24gb_multiscale.sh
bash scripts/cloud_train_resnet50_24gb_focal.sh
bash scripts/cloud_train_resnet50_24gb_warmup.sh
```

只有多尺度与类别感知单项都有效时，再运行：

```bash
bash scripts/cloud_train_resnet50_24gb_multiscale_classaware.sh
```

各项原理、类别影响和资源代价见 [CHANGELOG_MIOU.md](CHANGELOG_MIOU.md)。

## 显存预检

下面的命令模拟 ResNet50、cat64、512 crop、batch2、Deep Supervision、
CE+Dice、AMP、forward/backward/optimizer step，不会启动长训练：

```bash
python -m tools.benchmark_memory \
  --backbone resnet50 --batch-size 2 --image-size 512 \
  --cat-channels 64 --loss-profile ce_dice
```

如果 batch2 OOM，改用 `configs/cloud/resnet50_16gb.yaml`，保持 cat64 与 512 crop。
