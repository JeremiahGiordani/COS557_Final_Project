# TAR Revision Prediction Model

This repository contains code for predicting the likelihood of **Total Ankle Replacement (TAR) revision surgery** using patient X-rays and structured metadata.

Contributors:
- Jeremiah Giordani
- Tess Marvin
- Isabel Armour-Garb

Principal Investigator:
- Prof Olga Troyanskaya

## Problem Overview

Patients who undergo TAR may eventually require revision surgery. Accurately predicting which patients are at higher risk of revision can help physicians counsel patients more effectively and optimize post-op care strategies.

## Our Approach

We build a binary classification model that predicts whether a patient will require revision surgery based on:

- **X-rays** taken at various time points (pre- and post-op)
- **Patient metadata**, such as age, sex, BMI, time to/after surgery, and more

### Architecture Details

- All X-rays for a given patient encounter are passed through a pretrained ResNet-18 model.
- We extract feature maps for each image and apply mean pooling across the image set.
- The pooled image features are concatenated with normalized structured metadata.
- This combined representation is passed through a fully connected classifier to predict revision likelihood.

---

## How to Use

Clone this repository onto the Rothman server, and update the file paths to point to your local dataset.

### 1. **Installation**
Make sure you have Python 3.9+ and pip installed. Then, install all required packages using:

```bash
pip install -r requirements.txt
```

If you're working in a virtual environment (recommended), activate it first before running the above command.

---

### 2. **Evaluate a Pretrained Model**

Open `eval_model.py` and update these lines to your dataset location:

```python
image_dir = "/your/path/to/images"
csv_path = "/your/path/to/parsed_xray_files_log.csv"
patient_info_path = "/your/path/to/patient_info.csv"
```

Then run:

```bash
python eval_model.py
```

This will load a pretrained model and print evaluation metrics on the test set.

---

### 3. **Train a New Model**

Open `train_model.py` and update the same path variables to match your data:

```python
image_dir = "/your/path/to/images"
csv_path = "/your/path/to/parsed_xray_files_log.csv"
patient_info_path = "/your/path/to/patient_info.csv"
```

You can also change the seed of the train/val/test split:
```python
seed = <some_integer>
```

Then run:

```bash
python train_model.py
```

**Note**: The dataset is relatively small and highly imbalanced. If you modify the random seed to change the train/test/val split, you may unintentionally create significant class imbalance across splits. Do so cautiously.

---

## Results

You can check previous model training and evaluation logs in the `run_logs/` directory:

- **Training Logs**:
  - `run_logs/15_iterations_training.log`
  - `run_logs/15_iterations_training_replicated.log`

- **Evaluation Logs**:
  - `run_logs/eval.log`
  - `run_logs/eval_replicated.log`

These logs include final evaluation metrics like accuracy, F1 score, and precision-recall statistics.
