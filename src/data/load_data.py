import pandas as pd
from pathlib import Path


# Project root directory
ROOT_DIR = Path(__file__).resolve().parents[2]

# Dataset path
DATA_PATH = ROOT_DIR / "data" / "raw" / "students.csv"


def load_data(path=DATA_PATH):
    """
    Load the student dataset from a CSV file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    return df