# ETSP Project - Brain-to-Text 25

This repository contains the implementation for the "Brain-to-Text" project, also an opening contest held at Kaggle ([Kaggle Contest Link](https://www.kaggle.com/competitions/brain-to-text-25/))
. The project focuses on decoding text from neural signals using advanced neural network architectures and language models.

## Project Structure

```
.
├── analyses/           # Notebooks for analyzing results 
├── data/               # Dataset directory (Download required)
├── language_model/     # N-gram language model and decoding logic
├── model_training/     # Code for training and evaluating the neural networks
├── nejm_b2txt_utils/   # Utility functions
├── setup.sh            # Setup script for the main environment
├── setup_lm.sh         # Setup script for the language model environment
└── README.md           # This file
```

## Setup & Installation

This project requires **Linux** (or WSL on Windows) and **Conda**.

### 1. Environment Setup
There are two separate environments required: one for the main model training and one for the language model decoder.

```bash
# 1. Create the main environment (b2txt25)
bash setup.sh

# 2. Create the language model environment (b2txt25_lm)
bash setup_lm.sh

# 3. （Optional) Probable fixes required for langurage model setup, should rerun the setup.py in language_model\runtime\server\x86
cd language_model/runtime/server/x86
python setup.py build


```

### 2. Dependencies
*   **Redis**: Required for communication between the decoder and the language model.
    *   Ubuntu/WSL: `sudo apt-get install redis`
*   **C/C++ Tools**: Required for compiling the language model tools.
    *   Ubuntu/WSL: `sudo apt-get install cmake build-essential`

## Data Preparation

The dataset is hosted on Dryad.

1.  **Download**: [Dryad Dataset Link](https://datadryad.org/stash/dataset/doi:10.5061/dryad.dncjsxm85)
2.  **Extract**: Unzip `t15_copyTask_neuralData.zip` and `t15_pretrained_rnn_baseline.zip`.
3.  **Organize**: Place files in the `data/` directory so it looks like this:

```
data/
├── t15_copyTask.pkl
├── t15_personalUse.pkl
├── hdf5_data_final/          # Extracted from t15_copyTask_neuralData.zip
│   ├── t15.2023.08.11/
│   └── ...
└── t15_pretrained_rnn_baseline/
```

## Usage

### 1. Model Training

You can train either the baseline RNN or the Conformer model.

1.  **Configure**: Edit `model_training/rnn_args.yaml` to select the model type (`rnn` or `conformer`) and hyperparameters.
2.  **Train**:
    ```bash
    conda activate b2txt25
    cd model_training
    python train_model.py
    ```
    *   Logs are saved to `model_training/trained_models/`.

3.  **(Optional) Self-Supervised Pre-training**:
    To train the U-Net using Masked Autoencoder (MAE) strategy:
    ```bash
    python train_ssl.py
    ```

### 2. Evaluation

Evaluation requires running three separate processes in parallel (use 3 terminal windows).

**Terminal 1: Redis Server**
```bash
redis-server
```

**Terminal 2: Language Model**
```bash
conda activate b2txt25_lm
# Run the 1-gram language model
python language_model/language-model-standalone.py \
    --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil \
    --do_opt --nbest 100 --acoustic_scale 0.325 --blank_penalty 90 --alpha 0.55 \
    --redis_ip localhost --gpu_number 0
```
*Wait until you see "Successfully connected to the redis server".*

**Terminal 3: Evaluator**
```bash
conda activate b2txt25
cd model_training
python evaluate_model.py \
    --model_path ../data/t15_pretrained_rnn_baseline \
    --data_dir ../data/hdf5_data_final \
    --eval_type test \
    --gpu_number 0
```
*   Replace `--model_path` with trained model path
*   Results are saved as a CSV file in the model directory.

### 3. Shutdown redis
When you're done, you can shutdown the redis server from any terminal using `redis-cli shutdown`.


### 4. Larger Models

After downloading the n-gram language models from Dryad ([Dryad LM Link](https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq)), you can run the 3gram and 5gram models as follows:

#### run a 3gram model
To run the 3gram language model from the root directory of this repository (requires ~60GB RAM):
```bash
conda activate b2txt25_lm
python language_model/language-model-standalone.py --lm_path language_model/pretrained_language_models/openwebtext_3gram_lm_sil --do_opt --nbest 100 --acoustic_scale 0.325 --blank_penalty 90 --alpha 0.55 --redis_ip localhost --gpu_number 0
```

#### run a 5gram model
To run the 5gram language model from the root directory of this repository (requires ~300GB of RAM):
```bash
conda activate b2txt25_lm
python language_model/language-model-standalone.py --lm_path language_model/pretrained_language_models/openwebtext_5gram_lm_sil --rescore --do_opt --nbest 100 --acoustic_scale 0.325 --blank_penalty 90 --alpha 0.55 --redis_ip localhost --gpu_number 0
```