# Insider Threat Detection - Scripts

This directory contains the main pipeline scripts for training and evaluating the LSTM-Autoencoder model.

## Pipeline Overview

```
01_prepare_data.py → 02_feature_engineering.py → 03_train.py → 04_evaluate.py → 05_plot.py
```

## Scripts

### 1. Data Preparation (`01_prepare_data.py`)

Load raw CERT dataset and import into DuckDB database.

```bash
python scripts/01_prepare_data.py
```

**Outputs:**
- `data/insider.duckdb` - Database with all events

---

### 2. Feature Engineering (`02_feature_engineering.py`)

Sessionize events, extract features, and create train/val/test splits.

```bash
# Default: StandardScaler (recommended)
python scripts/02_feature_engineering.py

# Options:
python scripts/02_feature_engineering.py --scaler standard  # StandardScaler (zero-mean, unit-variance)
python scripts/02_feature_engineering.py --scaler minmax    # MinMaxScaler (0-1 range)
python scripts/02_feature_engineering.py --scaler none      # No scaling (raw values)
```

**Outputs:**
- `data/processed/X_train.npy`, `X_val.npy`, `X_test.npy`
- `data/processed/y_train.npy`, `y_val.npy`, `y_test.npy`
- `data/processed/sessions.parquet`
- `data/processed/artifacts/` (encoders, scaler)

---

### 3. Training (`03_train.py`)

Train the LSTM-Autoencoder model.

```bash
python scripts/03_train.py
```

**Outputs:**
- `outputs/models/checkpoint_best.pt` - Best model checkpoint
- `outputs/training_history.npy` - Loss history
- `outputs/logs/` - TensorBoard logs

**Monitor training:**
```bash
tensorboard --logdir outputs/logs
```

---

### 4. Evaluation (`04_evaluate.py`)

Evaluate model and generate metrics.

```bash
# Default: evaluate with both positive class perspectives
python scripts/04_evaluate.py

# Options:
python scripts/04_evaluate.py --positive-class insider   # Insider as positive class only
python scripts/04_evaluate.py --positive-class normal    # Normal as positive class only
python scripts/04_evaluate.py --threshold 1.0            # Use fixed threshold (instead of F1-optimal)
python scripts/04_evaluate.py --exclude-scenarios 3      # Exclude specific scenarios
python scripts/04_evaluate.py --exclude-scenarios 2,3    # Exclude multiple scenarios

# Combined example:
python scripts/04_evaluate.py --positive-class insider --threshold 1.0 --exclude-scenarios 3
```

**Outputs:**
- `outputs/evaluation/insider_positive/` - Metrics with Insider=positive
  - `metrics.npy` - All metrics
  - `confusion_matrix.png`
  - `roc_curve.png`
  - `pr_curve.png`
  - `error_distribution.png`
  - `per_scenario.json` - Per-scenario breakdown
- `outputs/evaluation/normal_positive/` - Metrics with Normal=positive

---

### 5. Plotting (`05_plot.py`)

Generate visualization plots.

```bash
python scripts/05_plot.py
```

**Outputs:**
- `outputs/plots/loss_curve.png` - Train/Val loss over epochs
- `outputs/plots/reconstruction_error_scatter.png` - Error distribution by class

---

## Full Pipeline

Run the complete pipeline:

```bash
# 1. Prepare data (one-time)
python scripts/01_prepare_data.py

# 2. Feature engineering
python scripts/02_feature_engineering.py --scaler standard

# 3. Train model
python scripts/03_train.py

# 4. Evaluate
python scripts/04_evaluate.py

# 5. Generate plots
python scripts/05_plot.py
```

---

## Configuration

Model and training parameters are configured in `config/config.yaml`.

### Model Architecture
```yaml
model:
  lookback: 20          # Sequence length (timesteps)
  lstm_units: [32, 16]  # Encoder: 32→16, Decoder: 16→32
  n_features: 12        # Input feature dimension
```

### Training Hyperparameters
```yaml
training:
  epochs: 200
  batch_size: 64
  learning_rate: 0.0001
  optimizer: "adam"
  loss: "mse"
```

### Data Splitting
```yaml
split:
  train: 0.7
  val: 0.1
  test: 0.2
```

### Feature Vector (12 features per session)
- `logon_time`, `logoff_time` (1-24)
- `day` (0-6)
- `user_id`, `pc` (encoded)
- `http_count`, `email_count`, `file_count`, `device_count`
- `user_role`, `functional_unit`, `department`

