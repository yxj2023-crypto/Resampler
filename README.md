# Resampler

独立的 WSI patch 重采样与 slot 实验项目。模型从 Survival_Predict 提取，保留算法、参数名称和 forward 返回格式；不依赖原项目即可导入。包括 Slot、Perceiver orth、Perceiver orth+rec，以及 Kmeans 基线。未附带特征、WSI 或训练权重。

- `resampler.py`：所有模型和 CrossAttn / Perceiver 依赖。
- `train.py`：便携训练入口，按病例拼接全部切片、使用全部 patch；Adam、batch=1、梯度累积默认8，尾组按实际病例数平均。
- `plot_occupancy.py`：从验证结果导出 slot 占比、空置频率及逐病例空槽图。
- `original_reference/`：三个原始训练入口和原始空槽分析代码，仅作完整实验逻辑参考，依赖原工程，不作为独立运行入口。
- `source_manifest.json`：模型来源及 SHA256。

## 数据格式

创建 `manifest.json`。每个 `.pt` 是形状 `[N,1024]` 的 Tensor；路径相对于 manifest 所在目录，也可填写绝对路径。病例在 train/val 间不得重叠。

```json
{"train": [{"case_id": "train_case", "paths": ["features/slide1.pt", "features/slide2.pt"]}],
 "val": [{"case_id": "val_case", "paths": ["features/slide3.pt"]}]}
```

## 运行

```bash
pip install -r requirements.txt
python train.py --manifest manifest.json --model slot --output runs/slot
python train.py --manifest manifest.json --model orth --iterations 1 --output runs/orth
python train.py --manifest manifest.json --model rec --iterations 1 --output runs/rec
# 原始四次迭代 orth
python train.py --manifest manifest.json --model orth --iterations 4 --output runs/orth4
python plot_occupancy.py runs/rec/validation_epoch_0030.json --output rec_occupancy.png
```

`rec` 实际为 orth+rec，两个损失默认权重均为1，使用独立 decoder。slot 默认4次竞争更新，orth/rec 训练入口默认1次。直接实例化 OrthogonalPerceiver / ReconstructionPerceiver 时，构造器默认迭代数仍保留原实现的4，需显式传入 `iterations=1`。

训练保存初始化(epoch 0)与每轮验证统计、best.pt、last.pt、history.json。Slot/rec 的 best 按验证 decoder_ratio 选择，orth 按 query_orthogonality 选择。导出入口简化了原训练脚本的 TensorBoard、WSI 空间可视化与多种辅助 best checkpoint；不保证重新训练得到逐位相同权重。模型结构与原实现保持一致，可以 `model.load_state_dict(torch.load(path,map_location='cpu')['model'])` 加载对应原权重。

## 模型接口

```python
import torch
from resampler import SlotAttnResampler, OrthogonalPerceiver, ReconstructionPerceiver
model = ReconstructionPerceiver(iterations=1)
r = model(torch.randn(1, 200, 1024), return_slots=True, return_details=True)
loss = r['loss_orth'] + r['loss_rec']
loss.backward()
# slots: [B,50,256]; prototypes: [B,50,1024]
# details.assignment: [B,N,50]，仅诊断，已 detach
counts = torch.bincount(r['details']['assignment'][0].argmax(-1), minlength=50)
```

注意：Perceiver 两组 prototypes/mass 在原实验中属于无梯度诊断输出；训练使用 hidden slots、orth 和 rec。空槽指硬分配计数为0，非零 soft mass 不意味着非空。正交损失是 refined queries 的 Gram 矩阵与单位阵之差的 Frobenius 范数。

## 项目来源与复现范围

初始代码来自 `Survival_Predict/exports/resampler_experiments_20261029`，原始模型文件校验值见 `source_manifest.json`。`original_reference/` 保留原实验入口供追溯，不是独立运行入口。示例数据清单见 `manifest.example.json`；请自行准备特征数据并修改路径。验证记录见 `VALIDATION.md`。大型数据、训练权重和运行输出不纳入 Git。
