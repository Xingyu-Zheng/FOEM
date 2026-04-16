# First-Order Error Matters: Accurate Compensation for Quantized Large Language Models.

<div align="center">
  <a href=https://ojs.aaai.org/index.php/AAAI/article/view/40123 target="_blank"><img src=https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage height=22px></a>
  <a href=https://huggingface.co/collections/Xingyu-Zheng/foem-quantization target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href=https://github.com/Xingyu-Zheng/FOEM target="_blank"><img src= https://img.shields.io/badge/Page-bb8a2e.svg?logo=github height=22px></a>
  <a href=https://arxiv.org/abs/2507.11017 target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>
  <a href=https://modelscope.cn/models/XingyuZheng target="_blank"><img src=https://img.shields.io/badge/ModelScope-Models-624aff.svg height=22px></a>
</div>

FOEM has been accepted at [AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40123).

We have completed the integration with [GPTQModel](https://github.com/ModelCloud/GPTQModel). 

Parts of this repository are now outdated, but we keep it available for developers who wish to debug or experiment with the algorithm.

The code snippets and results below are all obtained using GPTQModel.


## Quant

```python
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig, FOEMConfig

size = "8B"
model_id = f"Qwen/Qwen3-{size}"
quant_path = f"models/gptqmodel/Qwen3-{size}-foem-4bit"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
  ).select(range(256))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128, foem=FOEMConfig(alpha=0, beta=0.2, device="auto"))

model = GPTQModel.load(model_id, quant_config)

model.quantize(calibration_dataset, batch_size=4)

model.save(quant_path)

```

## Eval

```bash
lm-eval --model vllm --model_args pretrained=models/gptqmodel/Qwen3-8B-foem-4bit,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.6 --tasks wikitext --batch_size auto
```

## Result

Note: The PPL evaluation on WikiText using lm-eval differs from that reported in our original paper.

| Model      | Method           | Bits | Hyperparameters      | Wikitext PPL |
| ---------- | ---------------- | ---- | -------------------- | ------------ |
| Qwen3-0.6B | GPTQ             | 4    | \                    | 30.0372      |
|            | GPTAQ            | 4    | alpha=0.25           | 30.5776      |
|            | FOEM (w/o GPTAQ) | 4    | alpha=0, beta=0.2    | 29.6199      |
|            | FOEM (w/ GPTAQ)  | 4    | alpha=0.25, beta=0.2 | 29.3823      |
| Qwen3-8B   | GPTQ             | 4    | \                    | 12.5488      |
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


