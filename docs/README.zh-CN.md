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

## 从口袋 PDB 生成分子

训练好的模型位于 [`models/pocketgpt-v1.pt`](../models/pocketgpt-v1.pt)，下载仓库时会一并取得，无需另外下载权重。

### 1. 安装 Linux 环境

完整口袋流程使用 **Linux x86_64、glibc 2.28 或更新版本**，需要 Conda、Git、GCC/G++、Make 和 zlib 开发头文件。缺少这些工具时，先按[环境准备说明](INSTALLATION.md#linux-prerequisites)安装。示例使用 CPU，不要求显卡。

以下真实 PDB 流程已在 Ubuntu 22.04 的 CPU 环境实测，包括“完整蛋白＋口袋”和“仅口袋 PDB”两种输入。

```bash
git clone --depth 1 https://github.com/qianrong0709/PocketGPT.git
cd PocketGPT
export PG_RUNTIME="$PWD/.pocketgpt-runtime"
```

先阅读[依赖许可说明](INSTALLATION.md#licenses)，其中提供了 HTMD、DeeplyTough 和 MGLTools 1.5.6 对应版本的许可。MGLTools 的许可在该版本安装包的 `info/LICENSE.txt` 中，不能用新版网页条款替代；文档提供了只下载、校验并查看该文件的命令。阅读并接受适用条款后，再执行：

```bash
bash scripts/setup_linux.sh --prefix "$PG_RUNTIME" \
  --accept-htmd-license --accept-mgltools-license
export PG_PYTHON="$PG_RUNTIME/envs/generator/bin/python"
"$PG_PYTHON" scripts/download_model.py --verify-only
```

脚本会安装生成模型、DeeplyTough 和 MGLTools 的独立环境，并自动写好 `encoder.local.json`，无需手动填写解释器和权重路径。它们使用不同 Python 版本，因此不能把全部依赖混装进同一个环境。安装中断时，查看 `.pocketgpt-runtime/logs/setup.log`，解决问题后在原命令末尾加 `--resume` 继续。

### 2. 输入自己的口袋 PDB

把下面的 `/absolute/path/pocket.pdb` 换成自己的口袋 PDB 路径：

```bash
"$PG_PYTHON" -m pocketgpt generate \
  --checkpoint models/pocketgpt-v1.pt \
  --pocket /absolute/path/pocket.pdb \
  --encoder-config "$PG_RUNTIME/encoder.local.json" \
  --allow-unverified-pocket \
  --conditions pocket --count 10 --seed 42 --device cpu \
  --output outputs/pocket.csv
```

程序会依次提取 **128 维向量 → 缓存向量 → 输入 PocketGPT → 生成分子**。不需要重新训练，也不需要准备训练 CSV 或 JSONL 文件。

- `outputs/pocket.csv`：SMILES、合法性和计算性质，包含所有尝试生成的样本。
- `outputs/pocket.summary.json`：合法数量、去重数量及运行参数等统计。
- `.pocketgpt-runtime/pocket_cache/`：计算得到的口袋向量与编码日志。

### 3. 添加蛋白结构或其他条件

上面的命令只使用口袋文件中已有的原子。如果还有对应蛋白 PDB，可以添加 `--protein /absolute/path/protein.pdb`，保留口袋附近的额外结构信息。两个文件需要使用相同坐标系。口袋 PDB 应包含真实蛋白原子，不能只是标记空腔的虚拟点。

新开终端后，回到仓库目录并重新设置 `PG_RUNTIME` 和 `PG_PYTHON` 即可。每次运行使用新的 `--output` 文件名。需要组合性质、片段和 IFP 时，按[四条件示例](GENERATION.md#four-condition-generation)添加目标。

## 不使用口袋的生成示例

只使用性质、片段或 IFP 时，不必安装 DeeplyTough，按[仅安装生成器](INSTALLATION.md#generator-only-installation)配置即可。已经完成上面安装的用户可直接运行：

```bash
"$PG_PYTHON" examples/generate.py --count 10 --output outputs/example.csv
```

这个示例指定 QED = 0.7、LogP = 2.5 和苯环片段。权重文件内已经包含模型配置和词表。NVIDIA GPU 的配置见[环境说明](INSTALLATION.md#nvidia-gpu-for-the-generator)；配置好后可使用 `--device cuda`。

## 训练分为两个阶段

1. **性质＋单片段条件预训练**：使用母分子 SMILES、LogP/QED/TPSA/SAS 和单片段，不使用口袋或 IFP。
2. **结构–功能条件微调**：使用已有 CSV 中的口袋向量，加上性质、单片段和六类 IFP，不需要重新计算全量口袋向量，也不重新训练 DeeplyTough。

## 生成条件

生成时可按需要选择以下条件：

| 条件 | 用户提供的内容 |
| --- | --- |
| 口袋 | 128 维口袋向量，或口袋 PDB 和 DeeplyTough 配置；也可提供对应完整蛋白 PDB |
| 性质 | LogP、QED、TPSA、SAS 中一个或多个目标值 |
| 片段 | 一个片段 SMILES |
| IFP | 氢键、疏水、π–π、盐桥、π–阳离子、卤键六类二进制目标 |

当前模型使用历史缓存的 DeeplyTough 口袋向量。`legacy-precomputed-unverified` 表示当时没有记录完整的编码器身份，并不是提取向量时没有使用权重。“编码器签名”只是来源信息，不是模型的新输入；模型接收的仍然是 128 维向量，再经过学习到的口袋投影层。

新口袋的使用示例：

```bash
"$PG_PYTHON" -m pocketgpt generate \
  --checkpoint models/pocketgpt-v1.pt \
  --protein /absolute/path/protein.pdb \
  --pocket /absolute/path/pocket.pdb \
  --encoder-config "$PG_RUNTIME/encoder.local.json" \
  --allow-unverified-pocket \
  --conditions pocket --count 10 --device cpu \
  --output outputs/pocket.csv
```

已有向量时，把 `--protein`、`--pocket`、`--encoder-config` 换成 `--pocket-vector /absolute/path/vector.json` 即可。向量 JSON 可以是 128 个数的数组，或包含 `pocket_embedding` 和已记录的 `pocket_encoder_signature` 的对象。

对于当前历史模型，新编码器与训练向量的数值对应关系尚未核对。`--allow-unverified-pocket` 表示明确选择在这种情况下继续生成，程序会提示并在结果摘要中记录，而不会把新向量改名为旧向量。完成[原始向量对照](GENERATION.md#verify-an-encoder-against-training-vectors)后，可以改用 `--pocket-compatibility` 指定验证记录，无需改动生成模型参数。验证记录由维护者准备，普通用户不需要提供原训练数据。

从结构提取 IFP 需要配体及其结合姿势；提取片段需要参考分子及相应规则。性质和 IFP 目标也可以直接指定。

具体输入格式和运行命令见[输入说明](INPUTS.md)与[生成指南](GENERATION.md)。

## 依赖与引用

生成模型依赖 PyTorch、NumPy 和 RDKit，口袋编码基于 [DeeplyTough](https://github.com/BenevolentAI/DeeplyTough)。口袋预处理工具使用独立环境，第三方依赖遵循各自的许可证。

论文引用信息见主 README 和 `CITATION.cff`。
