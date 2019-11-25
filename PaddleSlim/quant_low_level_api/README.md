<div align="center">
  <h3>
    <a href=".README.md">
      模型量化概述
    </a>
    <span> | </span>
    <a href="../docs/tutorial.md">
      模型量化原理
    </a>
    <span> | </span>
    <a href=".quantization_aware_training.md">
      QAT量化Low-Level API使用方法和示例
    </a>
    <span> | </span>
    <a href=".post_training_quantization.md">
      离线量化Low-Level API使用方法和示例
    </a>
  </h3>
</div>

---
模型量化是使用更少的比特数表示神经网络的权重和激活的方法，具有加快推理速度、减小存储大小、降低功耗等优点。

目前，模型量化主要分为离线量化（Post Training Quantization）和QAT量化（Quantization Aware Training）。离线量化是基于采样数据，采用KL散度等方法计算量化比例因子的方法。它具有无需重新训练、快速获得量化模型的方法。QAT量化是在训练过程中对量化进行建模以确定量化参数，具有为复杂模型提供更高的精度的优点。

模型量化的原理和Low-Level API使用方法可以参考如下文档：
* [模型量化原理](../docs/tutorial.md)
* [量化训练Low-Level API使用方法和示例](./quantization_aware_training.md)
* [离线量化Low-Level API使用方法和示例](./post_training_quantization.md)
