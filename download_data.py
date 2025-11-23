from datasets import load_dataset
import pandas as pd
import os

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "data.csv")

def ensure_directory():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def download_and_save():
    print("Loading IMDB dataset (HuggingFace)…")
    dataset = load_dataset("imdb")

    print("Converting to pandas DataFrames…")
    train_df = dataset["train"].to_pandas()
    test_df  = dataset["test"].to_pandas()

    print("Combining splits…")
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    print("Saving to CSV…")
    full_df.to_csv(OUTPUT_FILE, sep="\t", header=False, index=False)

    print(f"Saved {len(full_df)} samples → {OUTPUT_FILE}")

if __name__ == "__main__":
    ensure_directory()
    download_and_save()
