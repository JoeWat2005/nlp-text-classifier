> [!WARNING]
> **Deprecated: no longer maintained.** Last updated in December 2025; kept for reference.

# NLP text classifier

A small sentiment classifier that labels text as positive or negative, trained on the IMDB movie review dataset with TF-IDF features and logistic regression.

## Features

- Downloads the IMDB review dataset from Hugging Face and stores it locally as a tab-separated file.
- Trains a TF-IDF vectoriser (the 5,000 most frequent terms, English stop words removed) and a logistic regression classifier.
- Prints test-set accuracy and a per-class classification report.
- Saves a confusion matrix and an ROC curve (with AUC) as PNG images.
- Interactive prediction loop that shows the predicted label and the model's confidence.

## Tech stack

- Python
- scikit-learn: TF-IDF, logistic regression and evaluation metrics
- Hugging Face `datasets`: dataset download
- pandas and NumPy
- Matplotlib and seaborn: plots

## Getting started

### Prerequisites

- Python 3.11 or newer. The repository does not state a minimum version, but `requirements.txt` pins `numpy==2.4.0`, which needs Python 3.11+.
- An internet connection for the first dataset download.

### Installation

Clone the repository and create a virtual environment called `env`:

```bash
git clone https://github.com/JoeWat2005/nlp-text-classifier.git
cd nlp-text-classifier
python -m venv env
```

Activate the environment.

Windows (PowerShell):

```powershell
.\env\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in the same window and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Windows (Command Prompt):

```bat
env\Scripts\activate
```

macOS or Linux:

```bash
source env/bin/activate
```

Install the pinned dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Usage

Run the three scripts in order from the repository root.

1. Download the dataset to `data/data.csv`. This needs an internet connection and the `datasets` package.

   ```bash
   python download_data.py
   ```

2. Train and evaluate the model. This prints the accuracy and a classification report, saves `models/confusion_matrix.png` and `models/roc_curve.png`, and writes `models/model.pkl` and `models/vectorizer.pkl`.

   ```bash
   python train.py
   ```

3. Classify your own sentences. Type `quit` to exit.

   ```bash
   python predict.py
   ```

The `data/` and `models/` folders are created on first use and are ignored by Git.

## How it works

1. `download_data.py` loads the `imdb` dataset with `datasets.load_dataset`, joins its train and test splits, and writes them to `data/data.csv` as tab-separated `text` and `label` columns with no header row.
2. `utils/preprocessing.py` reads the file back with pandas and skips any rows that fail to parse.
3. `train.py` fits the TF-IDF vectoriser on all reviews, makes a random 80/20 train/test split (`random_state=42`), trains `LogisticRegression(max_iter=500)`, evaluates it on the held-out 20%, then saves the plots and the pickled vectoriser and model.
4. `predict.py` loads the pickled vectoriser and model, transforms each sentence you type, and reports the class with the highest predicted probability (label 1 is positive, label 0 is negative).

| File | Purpose |
|---|---|
| `download_data.py` | Fetches the IMDB dataset and writes `data/data.csv` |
| `train.py` | Vectorises, splits, trains, evaluates and saves the model |
| `predict.py` | Interactive sentiment prediction |
| `utils/preprocessing.py` | Loads the tab-separated dataset |
| `utils/model_utils.py` | Builds the vectoriser and model, and saves and loads pickles |
| `utils/evaluation.py` | Plots the confusion matrix and ROC curve |

Note: pickle files can run code when they are loaded, so only load model files that you trained yourself.

## Project structure

```
.
├── download_data.py   # dataset download
├── train.py           # training and evaluation
├── predict.py         # interactive prediction
├── utils/             # preprocessing, model and plotting helpers
├── requirements.txt   # pinned dependencies
├── data/              # created by download_data.py (ignored by Git)
└── models/            # created by train.py (ignored by Git)
```

## Licence

MIT. See [LICENSE](LICENSE).
