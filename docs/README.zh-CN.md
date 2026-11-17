# PocketGPT 中文说明

[返回项目主页](../README.md)

本项目对应论文 *Structure–function conditioned molecular generation with PocketGPT enables prospective discovery of PARP1 inhibitors*，发表于 *The Innovation Drug Discovery* **1**(2), 100028 (2026)。

PocketGPT 结合蛋白口袋、分子性质、单片段和蛋白–配体相互作用指纹，自回归生成分子 SMILES，不直接构建配体三维坐标。论文报告了前瞻性的 PARP1 抑制剂设计、合成与实验验证，其中一个化合物的结合亲和力为 Kd = 4.88 nM。

- [论文](https://doi.org/10.59717/j.xinn-drugdisc.2026.100028)
- [项目展示页](https://qianrong0709.github.io/PocketGPT_Page/)
- [详细输入说明（英文）](INPUTS.md)
- [训练与续训指南（英文）](USAGE.md)
- [环境配置（CPU / CUDA）](INSTALLATION.md)
- [生成示例与条件组合](GENERATION.md)
- [模型说明](MODEL_CARD.md)

## 安装与生成

训练好的模型位于 [`models/pocketgpt-v1.pt`](../models/pocketgpt-v1.pt)，下载仓库时会一并取得，无需另外下载权重。

在 Python 3.11 的独立环境中安装。macOS 示例：

```bash
git clone https://github.com/qianrong0709/PocketGPT.git
cd PocketGPT
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 numpy==2.2.6 rdkit==2026.3.1
python -m pip install -e . --no-deps
python examples/generate.py --count 10 --output outputs/example.csv
```

示例默认读取 `models/pocketgpt-v1.pt`，指定 QED = 0.7、LogP = 2.5 和苯环片段，使用 CPU 生成分子。输出 CSV 包含 SMILES、合法性、计算性质和片段匹配结果；同名 `.summary.json` 记录生成统计。使用 NVIDIA GPU 时添加 `--device cuda`。Linux CPU、CUDA 与 Conda 的安装方式见[环境配置](INSTALLATION.md)。

权重文件内已包含模型配置和词表。上面的性质＋片段示例不需要安装 DeeplyTough。使用四类条件时，需要再提供兼容的口袋向量和六类 IFP 目标，见[四条件示例](GENERATION.md#four-condition-generation)。

## 训练分为两个阶段

1. **性质＋单片段条件预训练**：使用母分子 SMILES、LogP/QED/TPSA/SAS 和单片段，不使用口袋或 IFP。
2. **结构–功能条件微调**：使用已有 CSV 中的口袋向量，加上性质、单片段和六类 IFP，不需要重新计算全量口袋向量，也不重新训练 DeeplyTough。

## 生成条件

生成时可按需要选择以下条件：

| 条件 | 用户提供的内容 |
| --- | --- |
| 口袋 | 与训练特征空间兼容的 128 维口袋向量；原始 PDB 接口还需要匹配的编码器及明确口袋 |
| 性质 | LogP、QED、TPSA、SAS 中一个或多个目标值 |
| 片段 | 一个片段 SMILES |
| IFP | 氢键、疏水、π–π、盐桥、π–阳离子、卤键六类二进制目标 |

当前模型使用历史缓存口袋向量，来源标记为 `legacy-precomputed-unverified`。新计算的 DeeplyTough 向量与训练向量的兼容性尚未验证；使用新口袋前需核对编码器与预处理流程，详见[口袋兼容性](MODEL_CARD.md#pocket-compatibility)。DeeplyTough 参数保持冻结。

从结构提取 IFP 需要配体及其结合姿势；提取片段需要参考分子及相应规则。性质和 IFP 目标也可以直接指定。

具体输入格式和运行命令见[输入说明](INPUTS.md)与[生成指南](GENERATION.md)。

## 依赖与引用

生成模型依赖 PyTorch、NumPy 和 RDKit，口袋编码基于 [DeeplyTough](https://github.com/BenevolentAI/DeeplyTough)。口袋预处理工具使用独立环境，第三方依赖遵循各自的许可证。

论文引用信息见主 README 和 `CITATION.cff`。
