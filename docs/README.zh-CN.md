# PocketGPT 中文说明

[返回项目主页](../README.md)

PocketGPT 将蛋白口袋、分子性质、单片段和蛋白–配体相互作用指纹结合到自回归分子生成中，输出分子的 SMILES，不直接构建配体三维坐标。

- [论文](https://doi.org/10.59717/j.xinn-drugdisc.2026.100028)
- [项目展示页](https://qianrong0709.github.io/PocketGPT_Page/)
- [详细输入说明（英文）](INPUTS.md)
- [安装与使用指南（英文）](USAGE.md)

## 训练分为两个阶段

1. **性质＋单片段条件预训练**：使用母分子 SMILES、LogP/QED/TPSA/SAS 和单片段，不使用口袋或 IFP。
2. **结构–功能条件微调**：使用已有 CSV 中的口袋向量，加上性质、单片段和六类 IFP，不需要重新计算全量口袋向量，也不重新训练 DeeplyTough。

## 生成条件

生成时使用兼容的模型检查点及其分词词表，并按需要选择条件：

| 条件 | 用户提供的内容 |
| --- | --- |
| 口袋 | 蛋白 PDB 和对应的口袋 PDB；整个蛋白文件本身不能确定结合位点 |
| 性质 | LogP、QED、TPSA、SAS 中一个或多个目标值 |
| 片段 | 一个片段 SMILES |
| IFP | 氢键、疏水、π–π、盐桥、π–阳离子、卤键六类二进制目标 |

新口袋通过配置好的 DeeplyTough 接口编码并缓存，其他条件分别编码后参与融合和生成。DeeplyTough 口袋编码器的参数保持冻结。

从结构提取 IFP 需要配体及其结合姿势；提取片段需要参考分子及相应规则。性质和 IFP 目标也可以直接指定。

具体输入格式和运行命令见[输入说明](INPUTS.md)与[使用指南](USAGE.md)。

## 依赖与引用

生成模型依赖 PyTorch、NumPy 和 RDKit，口袋编码基于 [DeeplyTough](https://github.com/BenevolentAI/DeeplyTough)。口袋预处理工具使用独立环境，第三方依赖遵循各自的许可证。

论文引用信息见主 README 和 `CITATION.cff`。
