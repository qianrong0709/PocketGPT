# PocketGPT 中文说明

[返回项目主页](../README.md)

PocketGPT 将蛋白口袋、分子性质、单片段和蛋白–配体相互作用指纹结合到自回归分子生成中，输出分子的 SMILES，不直接构建配体三维坐标。

- [论文](https://doi.org/10.59717/j.xinn-drugdisc.2026.100028)
- [项目展示页](https://qianrong0709.github.io/PocketGPT_Page/)
- [详细输入说明（英文）](INPUTS.md)
- [安装与使用指南（英文）](USAGE.md)
- [环境配置（CPU / CUDA）](INSTALLATION.md)
- [生成示例与条件组合](GENERATION.md)
- [模型说明](MODEL_CARD.md)

## 安装与生成

在 Python 3.11 的独立环境中安装。macOS 的简洁示例：

```bash
git clone https://github.com/qianrong0709/PocketGPT.git
cd PocketGPT
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 numpy==2.2.6 rdkit==2026.3.1
python -m pip install -e . --no-deps
python scripts/download_model.py
python examples/generate.py --count 10 --output outputs/example.csv
```

示例指定 QED=0.7、LogP=2.5 和苯环片段，使用 CPU 本地生成。输出 CSV 包含 SMILES、合法性、计算性质和片段匹配结果；同名 `.summary.json` 记录生成统计。推理不需要训练数据，也不要求用户准备 JSONL。Linux CPU、CUDA 与 Conda 的安装方式见[环境配置](INSTALLATION.md)。

权重通过 GitHub Release 提供，下载脚本校验 SHA-256；模型配置和词表已包含在权重文件中。CPU 示例不需要安装 DeeplyTough。希望使用四类条件时，另提供自己兼容的口袋向量和六类 IFP 目标，见[四条件示例](GENERATION.md#four-condition-generation)。

## 训练分为两个阶段

1. **性质＋单片段条件预训练**：使用母分子 SMILES、LogP/QED/TPSA/SAS 和单片段，不使用口袋或 IFP。
2. **结构–功能条件微调**：使用已有 CSV 中的口袋向量，加上性质、单片段和六类 IFP，不需要重新计算全量口袋向量，也不重新训练 DeeplyTough。

## 生成条件

生成时使用兼容的模型检查点及其分词词表，并按需要选择条件：

| 条件 | 用户提供的内容 |
| --- | --- |
| 口袋 | 与训练特征空间兼容的 128 维口袋向量；原始 PDB 接口还需要匹配的编码器及明确口袋 |
| 性质 | LogP、QED、TPSA、SAS 中一个或多个目标值 |
| 片段 | 一个片段 SMILES |
| IFP | 氢键、疏水、π–π、盐桥、π–阳离子、卤键六类二进制目标 |

新口袋可通过单独配置的 DeeplyTough 接口编码并缓存。当前发布权重使用历史缓存向量，其编码器来源标记为 `legacy-precomputed-unverified`；新算向量与历史向量的数值兼容性需要先核验，不能仅凭 128 维或修改来源标记就混用。DeeplyTough 参数保持冻结。

从结构提取 IFP 需要配体及其结合姿势；提取片段需要参考分子及相应规则。性质和 IFP 目标也可以直接指定。

具体输入格式和运行命令见[输入说明](INPUTS.md)与[使用指南](USAGE.md)。

## 依赖与引用

生成模型依赖 PyTorch、NumPy 和 RDKit，口袋编码基于 [DeeplyTough](https://github.com/BenevolentAI/DeeplyTough)。口袋预处理工具使用独立环境，第三方依赖遵循各自的许可证。

论文引用信息见主 README 和 `CITATION.cff`。
