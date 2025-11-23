import pandas as pd

def load_dataset(path):
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["text", "label"],
        quoting=3,  # ignore quotes
        engine="python",
        on_bad_lines="skip"  # skip corrupted rows
    )

    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str)

    return df["text"], df["label"]
