# Sleep HMM Project

This project trains and validates an unsupervised Gaussian HMM for sleep staging using EEG bandpower features from the Sleep-EDF dataset.

## Project Structure

```text
sleep-hmm-project/
├── data/
│   └── sleep-cassette/
├── results/
├── src/
│   ├── constants.py
│   ├── data_split.py
│   ├── feature_extraction.py
│   ├── training.py
│   ├── validation.py
│   ├── main.py
│   └── run_experiments.py
├── requirements.txt
└── README.md
````

## Setup

From the project root:

```bash
python3 -m venv myenv
source myenv/bin/activate
python -m pip install -r requirements.txt
```

## Data

Put the Sleep-EDF cassette files here:

```text
data/sleep-cassette/
```

The folder should contain both PSG files and matching hypnogram files:

```text
*-PSG.edf
*-Hypnogram.edf
```

## How to Run

All training and validation should be run through:

```bash
python src/run_experiments.py
```

This script will:

1. Find the PSG files in `data/sleep-cassette/`.
2. Split participants into training and validation sets.
3. Extract EEG bandpower features.
4. Train the HMM.
5. Automatically map the learned HMM states to Wake, NREM, and REM.
6. Validate predictions against the Sleep-EDF hypnogram labels.
7. Save results under `results/experiments/`.

## Outputs

Experiment results are saved in:

```text
results/experiments/
```

A summary file is also saved:

```text
experiment_summary.csv
```

## Main Settings

Most settings are edited in:

```text
src/constants.py
```

The experiment specific settings are in:

```text
src/run_experiments.py
```
