# NLP Text Classifier

A simple sentiment analysis project using the IMDB movie review dataset.
It classifies text as **Positive** or **Negative** using **TF‑IDF** + **Logistic Regression**. fileciteturn1file0

## Prerequisites
- **Python** (recommended: 3.10+)
- **Git** (optional, only needed if you want to clone with `git clone`)

## Setup (clone + virtual environment)

### 1) Clone the repository
```bash
git clone https://github.com/JoeWat2005/nlp-text-classifier.git
cd nlp-text-classifier
```

### 2) Create a virtual environment called `env`
```bash
python -m venv env
```

### 3) Activate the environment

**Windows (PowerShell):**
```powershell
.\env\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once (in the same PowerShell window) and try again:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Windows (CMD):**
```bat
env\Scripts\activate
```

**macOS / Linux:**
```bash
source env/bin/activate
```

### 4) Install dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
The dependency list is pinned in `requirements.txt` (includes `scikit-learn`, `datasets`, `pandas`, `matplotlib`, `seaborn`, etc.). fileciteturn1file1

## Run the project

### 1) Download the dataset
Downloads the IMDB dataset from Hugging Face and writes it to `data/data.csv`. fileciteturn1file4

```bash
python download_data.py
```

### 2) Train the model
Trains a TF‑IDF vectorizer + Logistic Regression model, prints evaluation metrics, and saves artifacts under `models/`. fileciteturn1file2

```bash
python train.py
```

### 3) Make predictions (interactive)
Loads `models/model.pkl` + `models/vectorizer.pkl` and lets you type sentences to classify. fileciteturn1file3

```bash
python predict.py
```

Type `quit` to exit.

## Folder structure
- `data/` — downloaded dataset (`data.csv`)
- `models/` — saved model/vectorizer + plots
- `utils/` — helper modules

## Troubleshooting

### `FileNotFoundError: ... models/confusion_matrix.png`
If you see an error about saving plots into `models/`, create the directory and rerun:
```bash
mkdir models
python train.py
```

### Dataset download issues
`download_data.py` requires an internet connection and the Hugging Face `datasets` package. fileciteturn1file4

## License
MIT License

