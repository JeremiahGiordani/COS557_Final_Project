# TAR Revision Prediction Model

This repository contains code for predicting the likelihood of **Total Ankle Replacement (TAR) revision surgery** using patient X-rays and structured metadata.

Contributors:
- Jeremiah Giordani
- Gia Musslewhite
- Braiden Aaronson
- Alex Borengasser

TAs:
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

Run the following script:

```bash
python eval_model.py
```

To direct output into a logfile, run:

```bash
python eval_model.py > eval_output.log 2>&1
```

This will load a pretrained model and print/log evaluation metrics on the test set.

---

### 3. **Train a New Model**

Run:

```bash
python train_model.py
```

To direct output into a logfile, run:

```bash
python train_model.py > train_output.log 2>&1
```

Note that you can also open `train_model.py` and change the seed of the train/val/test split:
```python
seed = <some_integer>
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
