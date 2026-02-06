# LMK > CLS: Landmark Pooling for Dense Embeddings

This repository provides the implementation for **Landmark Pooling (LMK)**. Landmark pooling replaces traditional `CLS` or `mean` pooling by aggregating representations over learned *landmark tokens*, leading to improved **long-context dense retrieval**.

This implementation is built upon the [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) codebase.

---

## Setup

### Environment Configuration

Set the following variables before running training or evaluation:

```bash
export OUTPUT_DIR=/path/to/save/models
export DATA_DIR=/path/to/datasets
```

### Training Data

Training utilizes the [BGE-full-data](https://huggingface.co/datasets/cfli/bge-full-data) and [BGE-M3](https://huggingface.co/datasets/Shitao/bge-m3-data) datasets.

---

## Fine-tuning

### Training Command

```bash
bash ft_scripts/msmarco_distil.sh <PREFIX> <MODEL_PATH> <N_PSG> <POOLING> <GRANULARITY>
```

| Parameter | Description |
| --- | --- |
| `PREFIX` | Experiment prefix string for naming |
| `MODEL_PATH` | Base model (e.g., `answerdotai/ModernBERT-base`) |
| `N_PSG` | Number of passages per query (1 + #negatives) |
| `POOLING` | Strategy: `lmk_var`, `lmk_fixed`, `lmk_en`, `mean`, `cls` or `latent_attn` |
| `GRANULARITY` | LMK splitter size (e.g., `32`, `64`, `128`) for fixed or `32_64_128_256` for variable |

---

## Evaluation

Run inference using `landmark_index.py`.

### Example: Long-context Retrieval (MLDR)

```bash
python eval_scripts/landmark_index.py \
    --model $OUTPUT_DIR/my_lmk_model \
    --pool landmark \
    --max_seq_len 8192 \
    --fixed_splitter True \
    --split_size 32 \
    --task_type mldr \
    --lang eng
```

---

## Citation

```bibtex
@misc{doshi2026lmkclslandmark,
      title={LMK > CLS: Landmark Pooling for Dense Embeddings}, 
      author={Meet Doshi and Aashka Trivedi and Vishwajeet Kumar and Parul Awasthy and Yulong Li and Jaydeep Sen and Radu Florian and Sachindra Joshi},
      year={2026},
      eprint={2601.21525},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.21525}, 
}
```

---

## Contact

* **Meet** ([meet@ibm.com](mailto:meet@ibm.com))
* **Aashka Trivedi** ([aashka.trivedi@ibm.com](mailto:aashka.trivedi@ibm.com))
