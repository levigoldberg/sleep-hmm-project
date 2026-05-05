# Sleep HMM Project

This project uses a Gaussian Hidden Markov Model (HMM) to learn sleep-stage-like states from Sleep-EDF EEG recordings. It extracts EEG band-power features from 30-second epochs, trains an unsupervised HMM, and validates the learned states against expert hypnogram labels.

Current model:

```text
Wake, NREM, REM
```

Important: the HMM is unsupervised, so `State 0`, `State 1`, and `State 2` do not automatically mean Wake, NREM, and REM. After training, user must inspect the learned means/transitions and manually map the raw states in `constants.py`.

---

## Setup

From the project root:

```bash
python3 -m venv myenv
source myenv/bin/activate
```

Install requirements with `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Put the Sleep-EDF files here:

```text
data/sleep-cassette/
```

This folder should contain both PSG files and matching Hypnogram files.

---

## Important settings

Most settings are in:

```text
src/constants.py
```

Key values:

```python
DATA_DIR = "data/sleep-cassette"
RESULTS_DIR = "results"

K = 3
CHANNEL = "EEG Fpz-Cz"
EPOCH_SECONDS = 30

FEATURE_EXTRACTION_METHOD = "relative"
DECODING_METHOD = "posterior"
```

For validation, the raw HMM states must be mapped manually:

```python
HMM_STATE_TO_VALIDATION_LABEL = {
    0: "Wake",
    1: "NREM",
    2: "REM",
}
```

---

## Run training

From the project root:

```bash
python src/main.py
```

This will:

1. Find PSG files in `data/sleep-cassette/`
2. Split them into training and validation sets
3. Save the split to `results/train_validation_split.csv`
4. Extract EEG features from training participants
5. Train the HMM with Baum-Welch
6. Save model parameters to `results/`
7. Print the final transition matrix and Gaussian means

Saved model files:

```text
results/initial_prob.npy
results/transition.npy
results/means.npy
results/variances.npy
```

---

## Interpret training output

After training, look at:

```text
Final Transition Matrix
Final Gaussian Means
```

Transition matrix:

```text
Rows = current state
Columns = next state
```

Example:

```text
State 1 -> 0.014  0.850  0.137
```

This means State 1 usually stays State 1 because the self-transition probability is `0.850`.

Gaussian means:

```text
Columns = delta_rel, theta_rel, alpha_rel, beta_rel
```

General interpretation:

```text
Wake: lower delta, higher alpha/beta
NREM: higher delta/theta
REM: mixed-frequency and harder to identify with EEG alone
```

Use the means and transitions to update this dictionary in `src/constants.py`:

```python
HMM_STATE_TO_VALIDATION_LABEL = {
    0: "Wake",
    1: "NREM",
    2: "REM",
}
```

You do not need to rerun training after changing this mapping.

---

## Run validation

After training and updating the mapping:

```bash
python src/validation.py
```

This will:

1. Load the trained model
2. Load the saved validation split
3. Extract features for held-out validation participants
4. Decode raw HMM states
5. Map raw states to Wake, NREM, and REM
6. Compare predictions against expert hypnogram labels
7. Save validation results

Validation output files:

```text
results/validation_predictions.csv
results/validation_summary_by_participant.csv
results/validation_confusion_matrix.csv
```

---

## Interpret validation results

The confusion matrix should be read as:

```text
Rows = true expert labels
Columns = predicted HMM labels
Diagonal = correct predictions
```

Example:

```text
Predicted   NREM    REM   Wake
True
NREM        3687   5169   3721
REM          231   1117   1954
Wake       14433  16352   4692
```

Correct predictions are on the diagonal:

```text
NREM correct = 3687
REM correct = 1117
Wake correct = 4692
```

Overall accuracy is:

```text
correct predictions / total predictions
```

A better model should have larger numbers on the diagonal and smaller numbers off the diagonal.

If the confusion matrix is poor, try a different manual mapping in `constants.py` and rerun:

```bash
python src/validation.py
```

You only need to rerun `main.py` if you changed the model, features, training data, or initialization.

---

### MNE filter warnings

You may see warnings about different highpass or lowpass filters. These come from EDF metadata and are OK because the code later selects one EEG channel to work with

---
