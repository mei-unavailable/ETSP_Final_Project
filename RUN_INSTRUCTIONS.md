# Brain-to-Text Training & Evaluation Guide

This guide provides command-line instructions for training and evaluating the Brain-to-Text models, including the new Conformer CTC architecture and Masked Autoencoder (MAE) pre-training. This is suitable for WSL or standard terminal environments.

## 1. Environment Setup

Ensure you have activated the project environment:

```bash
conda activate b2txt25
```

If you haven't set up the environment yet, run:
```bash
bash setup.sh
```

## 2. Data Preparation

Ensure the dataset is downloaded and extracted to `data/hdf5_data_final`.

```bash
python download_data.py
```

## 3. Model Configuration

The training configuration is located in `model_training/rnn_args.yaml`.

### Switching Architectures
To switch between the **RNN Baseline** and the **Conformer CTC** model, edit the `model.type` field in `model_training/rnn_args.yaml`:

**For RNN (Default):**
```yaml
model:
  type: 'rnn'
  # ... other parameters
```

**For Conformer:**
```yaml
model:
  type: 'conformer'
  n_heads: 4 # Conformer specific parameter
  # ... other parameters
```

You can also adjust `gpu_number` in this file to select which GPU to use (e.g., `'0'`).

## 4. Training the Model

To start training the selected model (RNN or Conformer), run:

```bash
cd model_training
python train_model.py
```

*   **Logs**: Training logs will be saved in `model_training/trained_models/baseline_rnn/training_log`.
*   **Checkpoints**: Checkpoints are saved in `model_training/trained_models/baseline_rnn/checkpoint`.

## 5. Self-Supervised Pre-training (MAE U-Net)

To train the U-Net using the Masked Autoencoder (MAE) strategy for feature learning/denoising:

```bash
cd model_training
python train_ssl.py
```

*   This will train a U-Net to reconstruct masked neural signals.
*   Models are saved in `model_training/trained_models/unet_ssl`.

## 6. Evaluation Pipeline

Evaluation requires running the Redis server and the Language Model (LM) in parallel with the evaluation script.

### Step 1: Start Redis Server
Open a **new terminal**, activate the environment, and start redis:
```bash
redis-server
```

### Step 2: Start Language Model
Open a **second terminal**, activate the LM environment (if different, otherwise use `b2txt25`), and run the LM:

```bash
# From the project root directory
conda activate b2txt25
python language_model/language-model-standalone.py \
    --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil \
    --do_opt \
    --nbest 100 \
    --acoustic_scale 0.325 \
    --blank_penalty 90 \
    --alpha 0.55 \
    --redis_ip localhost \
    --gpu_number 0
```
*Wait until you see "Successfully connected to the redis server".*

### Step 3: Run Evaluation
Open a **third terminal** (or use the original one), navigate to `model_training`, and run the evaluation script:

```bash
cd model_training
python evaluate_model.py \
    --model_path trained_models/baseline_rnn \
    --data_dir ../data/hdf5_data_final \
    --eval_type test \
    --gpu_number 0
```

*   **Note**: Replace `trained_models/baseline_rnn` with the actual path to your trained model if you changed the output directory.
*   **Output**: Predictions will be saved as a CSV file in the model directory.

### Step 4: Cleanup
When finished, you can shut down the Redis server:
```bash
redis-cli shutdown
```
