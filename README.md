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
```

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


### Train the model

```bash
python src/main.py
```

This trains the HMM and saves model files in `results/`

The user then must manually label the states based on the outputted results in `src/constants.py`

### Validate the model

```bash
python src/validation.py
```

This loads the trained model, uses same pariticpant split, compares predictions to the annotated hypnogram labels, and saves validation results in `results/`.

### Run experiments

```bash
python src/run_experiments.py
```

This runs multiple models specified in run_experiments.py and the outputs under `results/experiments/`.

## Main Settings

Most settings are edited in:

```text
src/constants.py
```
