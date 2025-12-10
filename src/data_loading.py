from pathlib import Path
import pandas as pd


# Centralised logic to load the raw Cars4You train/test data
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_raw_data(
    train_filename: str = "train.csv",
    test_filename: str = "test.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the original train and test files from data/raw/.

    Parameters
    ----------
    train_filename : str
        Name of the training file inside data/raw/.
    test_filename : str
        Name of the test file inside data/raw/.

    Returns
    -------
    train : pd.DataFrame
    test : pd.DataFrame
    """
    train_path = RAW_DATA_DIR / train_filename
    test_path = RAW_DATA_DIR / test_filename

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test
