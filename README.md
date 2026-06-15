# Learning Structured Scene Representations for Visual Question Answering

This repository provides the implementation of the **Structured Scene Reasoning (SSR)** framework for dense structured visual question answering. SSR converts unordered visual detections into a learnable row-column topological representation and performs missing-region reasoning for structured shelf scenes.

The project targets **structured VQA for missing-region analysis and replenishment**, rather than open-ended caption generation.

## Overview

SSR follows a structured pipeline:

1. Open-vocabulary object detection and local semantic-spatial aggregation.
2. Sinusoidal Spatial Embedding (SSE) for spatial representation.
3. Neural Grid Inference (NGI) for row-column topology learning.
4. Directional Structure Induction (DSI) for missing-region reasoning.
5. Slot-based structured answer synthesis for controlled VQA queries.

The repository contains the source code used for data processing, learnable grid completion, baseline evaluation, ablation studies, cross-domain evaluation, structured answer-quality evaluation, and visualization.

## Repository Structure

```text
configs/            Configuration templates.
scripts/            Helper scripts for running experiments.
src/                Source code for preprocessing, training, evaluation, and visualization.
paper_results/      Small CSV result files used in the paper.
docs/               File manifests and reproducibility notes.
requirements.txt    Python dependencies.
README.md           Usage instructions.
```

Large datasets, processed images, model checkpoints, model weights, and full output folders are not stored in this repository.

## Environment Setup

Create a Python environment and install dependencies:

```bash
conda create -n ssr-vqa python=3.10 -y
conda activate ssr-vqa
pip install -r requirements.txt
```

The main experiments were run on an AutoDL node with an NVIDIA RTX 3090 GPU.

## Data Preparation

This repository does not include raw datasets due to size and license restrictions.

Expected data structure:

```text
data/
├── SKU-110K/
├── GapDetection/
└── processed/
```

The main experiments use SKU-110K with a controlled topological masking protocol. External Gap Detection subsets are used for cross-domain evaluation.

The controlled masking protocol constructs missing-object samples by removing annotated product boxes with sufficient neighboring context. The removed object boxes are retained as ground-truth missing regions. This enables structured missing-region localization and structured VQA reference construction without additional manual vacancy annotations.

## Main Experimental Pipeline

The complete experiment pipeline includes:

1. Preparing controlled missing-object samples from SKU-110K.
2. Training the learnable grid-completion branch.
3. Evaluating localization baselines and the full SSR model.
4. Running cross-domain evaluation on Gap Detection subsets.
5. Running ablation studies.
6. Evaluating structured answer quality.
7. Generating paper tables and figures.

Please check the scripts under `scripts/` and modules under `src/` for the corresponding implementation files.

## Paper Result Files

Small result files used in the paper are provided under:

```text
paper_results/tables/
```

Expected files include:

```text
table2_localization_final.csv
table3_cross_domain_per_domain_final.csv
table4_ablation_prf_final.csv
table5_vqa_answer_quality_final_3seed.csv
```

These files summarize the main paper results. Full intermediate outputs are excluded from GitHub and should be stored separately in the experiment backup archive.

## Structured VQA Answer Quality Evaluation

SKU-110K does not provide human-written VQA annotations. Therefore, structured VQA references are automatically constructed from the controlled topological masking protocol. Each reference answer is derived from the ground-truth missing region, row-column position, and neighboring structural context.

Run the structured answer-quality evaluation for three independent runs:

```bash
for s in 42 2024 3407
do
  python -m src.eval.eval_vqa_answer_quality \
    --qa-jsonl data/processed/vqa/test_qa.jsonl \
    --ours-json outputs/results/ours_ssr_grid_test_seed_${s}.json \
    --out-summary outputs/tables/table5_vqa_answer_quality_seed_${s}.csv \
    --out-detail outputs/results/table5_vqa_answer_quality_detail_seed_${s}.csv
done
```

The reported metrics include:

* normalized Answer Exact Match;
* Token-F1;
* BLEU-1.

BLEU-4 and CIDEr are not used because the generated answers are short structured responses rather than long free-form captions with multiple human-written references.

The final three-seed result is summarized in:

```text
paper_results/tables/table5_vqa_answer_quality_final_3seed.csv
```

## Main Paper Results

The main localization comparison is summarized in:

```text
paper_results/tables/table2_localization_final.csv
```

The cross-domain evaluation is summarized in:

```text
paper_results/tables/table3_cross_domain_per_domain_final.csv
```

The ablation study is summarized in:

```text
paper_results/tables/table4_ablation_prf_final.csv
```

The structured answer-quality evaluation is summarized in:

```text
paper_results/tables/table5_vqa_answer_quality_final_3seed.csv
```

## Notes on Reproducibility

To reproduce the experiments from scratch, users need to prepare the required datasets and follow the same preprocessing and masking protocol.

This repository is designed for code release and paper-level reproducibility. It does not include the full 25GB AutoDL experiment snapshot. For exact recovery of the original AutoDL experiment state, the full local experiment directory or a full recovery archive is required. The separately saved paper-result archive contains the key paper tables, figures, selected result JSON files, answer-quality details, and environment records.

## Citation

If you use this code, please cite the corresponding paper:

```bibtex
@article{ssr_vqa_2026,
  title={Learning Structured Scene Representations for Visual Question Answering},
  author={Fan, Miao and Zhu, Shiyu and Ma, Chen and Xiong, Haoyi},
  year={2026}
}
```
