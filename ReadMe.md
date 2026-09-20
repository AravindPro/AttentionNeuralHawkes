# Attention Neural Hawkes for LOB

## Overview

This repository adapts and extends the Neural Hawkes model for limit order book (LOB) event modeling, adding attention-based improvements and utilities for working with LOB data preprocessed to 1-second resolution.

## Dataset

- Primary dataset (WSELOB 2017) used (preprocessed LOB at 1s resolution) was obtained from: https://data.mendeley.com/datasets/3g4mhdp899/1
-I have preprocessed this LOB data to 1sec market data using `orderbook2.py` code within WSELOB link above. The resulting market data is present in: https://www.kaggle.com/datasets/aravindproml/lob-1sec-dataset
- If you use the Kaggle dataset, update any example paths (e.g. `/kaggle/input/lob-1sec-dataset/PZU`) to point to your local dataset location.

## Requirements

- Python 3.8+ (3.10 recommended)
- See `requirements.txt` for exact dependency versions

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running / Setup

- GPU is not required; experiments were run on CPU. Runtimes depend on dataset slice, chosen thresholds, and hyperparameters. Example timings (informal): a 60-day training slice could take ~5 hours for certain hyperparameter settings used in experiments.

## How to run the notebook

1. Open a terminal in the repository root.
2. Create and activate a virtual environment if desired.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download or place the dataset in a local folder and update the dataset path in the notebook if needed. The notebook currently uses a Kaggle-style path:

```python
root_dir = '/kaggle/input/lob-1sec-dataset/PZU/'
```

Replace it with your local dataset directory, for example:

```python
root_dir = 'C:/path/to/lob-1sec-dataset/PZU/'
```

5. Prefeably move the notebook to Kaggle.

6. Open `neuralhawkesforlob.ipynb` and run the cells from top to bottom in order.
7. The training section is executed in the later cells of the notebook. You can adjust hyperparameters such as `lr`, `seq_len`, `batch_size`, `num_epochs`, and `num_files` near the top of the notebook before running the training cells.

> If you are using VS Code, you can also open the notebook directly and use the "Run All" command after installing the Python and Jupyter extensions.

## Model sections and execution order

The notebook is organized into three main experiment sections, and each one should be run independently:

1. Original model section
   - Uses the base `Conttime` model.
   - It reads the data, pads the event sequences, builds a DataLoader, and trains the standard CTLSTM-based Neural Hawkes model.
   - This is the simplest baseline and is the most direct reference implementation.

2. Feature-aware LSTM section
   - Uses `ConttimeFeat`.
   - The key difference is the addition of per-event feature tensors (`extra_feats`) alongside event type and duration sequences.
   - This section uses `padding_full_feats` or `padding_seq_len_feats` and passes feature tensors into `train_batch` and `forward`.
   - This section is a natural extension of the original model and is internally consistent.

3. Attention LSTM section
   - Uses `ConttimeFeatAttnLSTM`.
   - It augments the feature-aware model with `FeatureAttention`, then concatenates the attention output back into the feature vector before the CTLSTM update.
   - This is the most advanced variant and is conceptually coherent, but it is also the most environment-specific block because it uses a hard-coded Kaggle path for a saved model and writes to the same shared output artifacts.

Important: run each section separately. Do not execute all three training cells in sequence as one continuous workflow. The notebook reuses global variables and file names such as `model.pt`, `model_old.pt`, and the training log outputs, so running all three sections together can overwrite earlier results and mix model states.

## Flow review of each model

### 1) Original model
The flow is: preprocess LOB data -> create event times and types -> split train/test -> pad sequences -> build `DataLoader` -> call `Conttime.train_batch()` -> compute likelihood via `conttime_loss()` -> update optimizer.

### 2) Feature-aware model
The flow is: preprocess -> normalize features -> split -> pad both event sequences and feature tensors -> pass `(types, dtime, extra_feats)` into `ConttimeFeat.train_batch()` -> run `forward()` with feature concatenation -> compute the same likelihood objective with auxiliary features.

### 3) Attention model
The flow is: preprocess -> normalize features -> split -> create attention-enriched feature vectors via `FeatureAttention` -> concatenate attention output with original features -> feed into CTLSTM -> compute the same likelihood objective.

## Notebook: neuralhawkesforlob.ipynb

This repository includes `neuralhawkesforlob.ipynb`, a Jupyter notebook that demonstrates data preparation, model training, and evaluation. The notebook contains:

- Data loading: instructions and example code to load the preprocessed 1s LOB dataset and expected file structure.
- Preprocessing: normalization, event filtering, jump/threshold computation, and how slices are created for experiments.
- Model: the attention-augmented Neural Hawkes model architecture and the key implementation details used in training.
- Training: an example training run with sample hyperparameters and how to reproduce the main plots.
- Evaluation & plots: loss curves and the main result plots (notably the epoch-wise graphs referenced in the paper).

## Notes

- Much of the implementation is adapted from NeuralHawkes: https://github.com/Hongrui24/NeuralHawkesPytorch
- Results and runtimes depend on the number of days used, jump threshold (e.g. 0.05% vs 0.01%), and epochs.

## References

- Neural Hawkes implementation: https://github.com/Hongrui24/NeuralHawkesPytorch
- Dataset (primary): https://data.mendeley.com/datasets/3g4mhdp899/1
- Kaggle LOB 1s dataset: https://www.kaggle.com/datasets/aravindproml/lob-1sec-dataset
