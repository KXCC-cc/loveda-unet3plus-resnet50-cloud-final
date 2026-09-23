# ResNet50 最终实验权威记录

本文档固定 LoveDA ResNet50 + U-Net 3+ 最终实验的真实配置与结果。参数以
`runs/cloud_resnet50_24gb_eff4_40ep/config.json` 为准，验证结果以同目录的
`summary.json`、`metrics.csv` 和 `confusion_matrix.csv` 为准；训练日志用于交叉核对
训练进度。文档描述与原始记录冲突时，以这些原始文件为准。

## 实验目的与背景

项目任务是 LoveDA 七类遥感语义分割。初始 scratch U-Net 3+ baseline 的历史最佳
Full Validation mIoU 约为 0.3457；第一阶段使用 ImageNet 预训练 ResNet34 Encoder
后，Full Validation mIoU 达到 0.4669311579969285。

下一步计划在不破坏 U-Net 3+ Full-scale Skip Connections 和 Deep Supervision 的
前提下，将 Encoder 升级为 ResNet50。本地 RTX 4060 Laptop 8GB 在完整配置下显存
余量不足：已有显存记录显示 ResNet50、512 输入、physical batch 1 已接近容量上限，
batch 2 会 OOM，因此正式完整实验转移到云端 NVIDIA A10。

## 原始 24GB 方案为何不是最终实验

`runs/cloud_resnet50_24gb/` 保存最初设计的云端训练方案：

- physical batch：2
- accumulation steps：8
- effective batch：16
- resolved learning rate：0.01
- max optimizer steps：15000

但是该目录的 `metrics.csv` 只有表头，没有完整 epoch 指标，也没有可供正式结论使用的
完整结果。因此它只能作为“原始计划/未完成方案”保留。
`configs/cloud/resnet50_24gb.yaml` 同样是历史设计配置，不是最终实际训练配置。

原方案的梯度累积需要较多串行 micro-batch。为缩短实际云端训练时间，最终将
accumulation 从 8 降为 2，使 effective batch 从 16 降为 4；学习率按 batch size
线性缩放为：

```text
0.01 × 4 / 16 = 0.0025
```

同时把 optimizer updates 增加到 25200，使训练覆盖约 40 个数据轮次。

## 最终实验目录与环境

正式实验目录：

```text
runs/cloud_resnet50_24gb_eff4_40ep/
```

实际环境由该目录的 `config.json` 记录：

| 项目 | 实际值 |
|---|---|
| GPU | NVIDIA A10 |
| PyTorch | 2.3.1+cu121 |
| CUDA runtime | 12.1 |
| cuDNN | 8902 |
| AMP | 开启 |
| channels_last | 开启 |
| compile | 关闭 |
| seed | 42 |

## 模型结构

- 任务：LoveDA 七类语义分割，`num_classes=7`
- Encoder：ImageNet `IMAGENET1K_V1` 预训练 ResNet50
- Decoder：U-Net 3+
- Full-scale Skip Connections：完整保留
- Deep Supervision：完整保留，主输出加 4 个辅助输出
- `cat_channels=64`
- `standard_resnet_stem=false`
- ResNet stem 的 7×7 `conv1` 保留预训练权重，但 stride 从 2 改为 1
- Encoder BatchNorm 转换为 FrozenBatchNorm
- Decoder 使用 GroupNorm

## 数据、损失与训练参数

| 项目 | 最终实际值 |
|---|---|
| Dataset | LoveDA |
| 类别数 | 7 |
| crop | 512×512 |
| 训练尺度 | 单尺度 1.0 |
| class-aware crop | 关闭，probability=0.0 |
| Loss | CE 1.0 + Dice 1.0 |
| Focal | 关闭，weight=0.0 |
| Deep Supervision 辅助权重 | 0.5 / 0.25 / 0.125 / 0.0625 |
| optimizer | SGD |
| momentum | 0.9 |
| weight decay | 1e-4 |
| scheduler | Poly |
| poly power | 0.9 |
| physical batch | 2 |
| accumulation steps | 2 |
| effective batch | 4 |
| resolved learning rate | 0.0025 |
| differential LR | 关闭；Encoder/Decoder 均为 0.0025 |
| max optimizer steps | 25200 |
| updates per epoch | 630 |
| gradient clip | 1.0 |
| Full Validation interval | 每 8 epochs |
| Validation scales | 单尺度 1.0 |
| horizontal flip TTA | 关闭 |

Differential LR、class-aware crop、multi-scale training、Focal 和 warmup 虽然已有可选
配置或实现，但没有用于本次最终实验，也没有完整正式结果证明它们带来提升。

## 最佳验证结果

`summary.json` 和 `metrics.csv` 一致记录最佳结果出现在 epoch 24：

| 指标 | 数值 |
|---|---:|
| Full Validation mIoU | **0.48673445612346994** |
| Mean Dice | **0.6502697251948383** |
| Val loss | **1.7310836375866772** |

epoch 24 的七类 IoU：

| 类别 | IoU |
|---|---:|
| background | 0.5300868907 |
| building | 0.6054462719 |
| road | 0.5075436849 |
| water | 0.5262558001 |
| barren | 0.3261346581 |
| forest | 0.4121493904 |
| agricultural | 0.4995244967 |

训练日志记录 epoch 24 完成时 optimizer step 为 15112/25200，随后进行完整
Validation。训练最终在 epoch 41、optimizer step 25200 结束；最后一次验证结果为：

- Final mIoU：`0.4764585640736871`
- Final mean Dice：`0.6381809247102189`
- Final Val loss：`1.795865305602872`

最终 epoch 指标低于 epoch 24 的最佳值，因此不能用 `last_checkpoint.pth` 替代
`best_model.pth` 进行正式评估或提交。

## ResNet34 与 ResNet50 对比

| 模型 | Full Val mIoU | Mean Dice | Hidden Test mIoU |
|---|---:|---:|---:|
| ResNet34 + U-Net 3+ | 0.4669311579969285 | 0.632850971221246 | 0.457311 |
| ResNet50 + U-Net 3+ | **0.48673445612346994** | **0.6502697251948383** | **0.469813** |
| ResNet50 提升 | **+0.019803298126541413** | +0.0174187539735923 | **+0.012502** |

因此，ResNet50 相比 ResNet34：

- Full Validation mIoU 提升约 **1.98 个百分点**。
- LoveDA Hidden Test mIoU 提升约 **1.25 个百分点**。

ResNet50 Hidden Test `0.469813` 来自使用最终 `best_model.pth` 生成的 LoveDA Test
单尺度提交。仓库没有保存该次 Test submission ID，因此只归档实际得分，不填写或
推测 submission ID。

## Checkpoint 角色

两个最终权重均由 Git LFS 管理，不得删除或改写：

- `best_model.pth`：epoch 24 达到最高 Full Validation mIoU 时保存，是最终验证、
  推理和 LoveDA Test 提交应使用的权重。
- `last_checkpoint.pth`：epoch 41 / 25200 optimizer updates 的最终训练状态，主要用于
  断点恢复、训练状态审计和复现训练结束位置；它不是性能最好的 checkpoint。

最终结论与正式提交统一使用：

```text
runs/cloud_resnet50_24gb_eff4_40ep/best_model.pth
```

## 原始证据索引

- 最终解析配置：`runs/cloud_resnet50_24gb_eff4_40ep/config.json`
- 最终汇总：`runs/cloud_resnet50_24gb_eff4_40ep/summary.json`
- 各验证点指标：`runs/cloud_resnet50_24gb_eff4_40ep/metrics.csv`
- 最佳点混淆矩阵：`runs/cloud_resnet50_24gb_eff4_40ep/confusion_matrix.csv`
- 原始未完成方案：`runs/cloud_resnet50_24gb/config.json` 与 `metrics.csv`
- ResNet34 对照：`docs/resnet34_reference.md`
- 分段训练日志：`train_resnet50_eff4_40ep.log`、
  `train_resnet50_eff4_40ep_resume2.log`、`train_resnet50_eff4_40ep_resume3.log`
