# First-Order Error Matters: Accurate Compensation for Quantized Large Language Models.

This is the implementation repository for FOEM [AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40123).

This repository has not been updated for some time because we have completed the integration with [GPTQModel](https://github.com/Xingyu-Zheng/GPTQModel). The implementation is now fully usable and is currently being merged into the main branch. See: https://github.com/ModelCloud/GPTQModel/issues/1678 and https://github.com/ModelCloud/GPTQModel/pull/2639.

```python
from datasets import load_dataset, load_from_disk
from gptqmodel import GPTQModel, QuantizeConfig, FOEMConfig

size = "8B"
model_id = f"models/Qwen3/Qwen3-{size}"
quant_path = f"models/gptqmodel/Qwen3-{size}-foem-4bit"

calibration_dataset = load_from_disk("datasets/allenai/c4/allenai--c4/train").select(range(256))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128, foem=FOEMConfig(alpha=0, beta=0.2, device="auto"))

model = GPTQModel.load(model_id, quant_config)

model.quantize(calibration_dataset, batch_size=4)

model.save(quant_path)

```

| Model      | Method           | Bits | Hyperparameters      | Wikitext PPL |
| ---------- | ---------------- | ---- | -------------------- | ------------ |
| Qwen3-0.6B | GPTQ             | 4    | —                    | 30.0372      |
|            | GPTAQ            | 4    | alpha=0.25           | 30.5776      |
|            | FOEM (w/o GPTAQ) | 4    | alpha=0, beta=0.2    | 29.6199      |
|            | FOEM (w/ GPTAQ)  | 4    | alpha=0.25, beta=0.2 | 29.3823      |
| Qwen3-8B   | GPTQ             | 4    | —                    | 12.5488      |
|            | GPTAQ            | 4    | alpha=0.25           | 12.7152      |
|            | FOEM (w/o GPTAQ) | 4    | alpha=0, beta=0.2    | 12.5128      |
|            | FOEM (w/ GPTAQ)  | 4    | alpha=0.25, beta=0.2 | 12.6172      |



------

### Citation

If you find this work useful, please cite:

```
@inproceedings{zheng2026first,
  title={First-order error matters: Accurate compensation for quantized large language models},
  author={Zheng, Xingyu and Qin, Haotong and Li, Yuye and Chu, Haoran and Wang, Jiakai and Guo, Jinyang and Magno, Michele and Liu, Xianglong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={34},
  pages={28883--28891},
  year={2026}
}
```


