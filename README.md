# Cars4You - Used Car Price Prediction

This project predicts used car prices for the Cars4You platform using supervised machine learning (regression).

## Project structure

- `data/raw/` – original Kaggle data (`train.csv`, `test.csv`)
- `data/processed/` – (optional) processed datasets or feature matrices
- `notebooks/` – Jupyter notebooks for EDA, modelling and evaluation
- `src/` – Python modules for preprocessing, training and inference
- `.gitignore` – files and folders that are excluded from version control
- `requirements.txt` – Python dependencies

## How to run

1. Make sure the files `train.csv`, `test.csv` and `sample_submission.csv` are in `data/raw/`.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt

3. Open the notebooks in `notebooks/` to reproduce the analysis.

A final model is trained on the Cars4You training data and evaluated using regression metrics (e.g. MAE, RMSE, R²) with cross-validation.


