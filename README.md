# 🧠 LLM Text Classification Project (with Ray + MLflow + Transformers)

This project demonstrates how to build and scale a text classification pipeline using **SciBERT**, **Ray Train**, **MLflow**, and **Ray Tune**. It represents a foundational step into **LLMOps**, covering the full training and tuning lifecycle.

---

## 🚀 Project Overview

We fine-tune a pretrained BERT-based language model to classify short text descriptions into tags such as:

- `natural-language-processing`
- `computer-vision`
- `mlops`
- `other`

The workflow covers:

- Data loading and stratified splitting
- Preprocessing and featurization
- Distributed training and evaluation with Ray
- Hyperparameter tuning
- Experiment tracking with MLflow
- Prediction from the best run

---

## ✅ What’s Done So Far

### 1. 🔧 Training

```bash
export EXPERIMENT_NAME="llm"
export DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'

python scripts/train.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --dataset-loc "$DATASET_LOC" \
    --train-loop-config "$TRAIN_LOOP_CONFIG" \
    --num-workers 1 \
    --cpu-per-worker 1 \
    --gpu-per-worker 0 \
    --num-epochs 10 \
    --batch-size 256 \
    --results-fp results/training_results.json

export EXPERIMENT_NAME="llm"
export DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
export INITIAL_PARAMS="[{\"train_loop_config\": $TRAIN_LOOP_CONFIG}]"

python scripts/tune.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --dataset-loc "$DATASET_LOC" \
    --initial-params "$INITIAL_PARAMS" \
    --num-runs 2 \
    --num-workers 1 \
    --cpu-per-worker 1 \
    --gpu-per-worker 0 \
    --num-epochs 10 \
    --batch-size 256 \
    --results-fp results/tuning_results.json

export EXPERIMENT_NAME="llm"
export RUN_ID=$(python madewithml/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)

python scripts/predict.py predict \
    --run-id $RUN_ID \
    --title "Transfer learning with transformers" \
    --description "Using transformers for transfer learning on text classification tasks."
```
