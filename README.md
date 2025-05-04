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

These logs include final evaluation metrics like accuracy, F1 score, and precision-recall statistics.

### Extensions (optional)
When training a new model using train_model.py, several optional command-line arguments are available to customize training behavior.

If you run the script without any arguments:

```bash
python train_model.py
```
you’ll get the default settings:

- Data split by patient (--split=by-patient)
- Binary cross-entropy loss (--loss=bce)
- No image augmentation

You can customize training with the following options:

Choose how the dataset is split:
`--split`

- by-patient (default): Ensures that no patient appears in more than one dataset split (train/val/test). This is highly recommended when using metadata to prevent information leakage.
- random: Randomly splits datapoints, which may include multiple appointments from the same patient across different splits.

Specify the loss function:
`--loss`

- bce (default): Standard binary cross-entropy with class weighting for imbalance.
- focal: Focal loss, which emphasizes harder-to-classify examples and may help in highly imbalanced settings.

Enable data augmentation:
`--augment`

- If specified, each datapoint (i.e., appointment) is duplicated with 7 different image transformations (e.g., flipping, rotation, inversion). This significantly increases dataset size and may improve generalization.

Example usage:

```bash
python train_model.py --split=by-patient --loss=focal --augment
```
This will train the model using a patient-independent split, focal loss, and augmented training data.