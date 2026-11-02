验证通过：三组默认模型与原实现的 state_dict 兼容，前向 slots/prototypes/mass/loss/assignment 和参数梯度逐位一致（CPU，随机输入 [1,31,1024]）。参数量分别为 867840 / 867840 / 1131520。三个便携训练入口均完成 CPU 小规模一轮训练、验证、checkpoint 保存；空槽绘图完成。未重新执行全量数据训练。

## 独立项目验证（2026-11-02）

在 Mamba_py38（Python 3.8.20）中，使用临时合成特征 [17,16]、4 个 slots，在 CPU 上分别运行 slot、orth、rec 各一轮训练和验证；确认 best.pt、last.pt、history.json 和验证统计生成，并完成 occupancy 绘图。临时数据及权重未加入仓库。此检查验证独立入口可运行，不代表全量实验复现。
