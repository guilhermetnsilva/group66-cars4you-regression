# Cars4You - Used Car Price Prediction (Group 66)

This repository contains the final project for Group 66, focused on predicting used car prices for the Cars4You platform using supervised machine learning (regression).

## Project structure

- `data/raw/` – original Kaggle data (`train.csv`, `test.csv`)
- `data/processed/` – (optional) processed datasets or feature matrices
- `notebooks/` – Jupyter notebooks for EDA, modelling and evaluation  
  - main notebook: `ML_project.ipynb`
- `src/` – (reserved) Python modules for preprocessing, training and inference (future refactor)
- `submissions/` – final prediction files for Kaggle/assessment (e.g. `group66_rf_submission.csv`)
- `models/` – local model artifacts (e.g. `car_price_bundle.joblib`, ignored in version control)
- `.gitignore` – files and folders that are excluded from version control
- `requirements.txt` – Python dependencies

## How to run

1. Make sure the files `train.csv` and `test.csv` are in `data/raw/`.

2. Install dependencies (ideally in a virtual environment):

   ```bash
   pip install -r requirements.txt
   
3. Open the notebook in notebooks/ (e.g. ML_project.ipynb) and run the cells in order.
The notebook assumes the data is available in data/raw/ using relative paths such as:

train_data = pd.read_csv("../data/raw/train.csv")
test_data = pd.read_csv("../data/raw/test.csv")

Final model and predictions

The final regression model is trained on the Cars4You training data and evaluated using cross-validation with standard regression metrics (e.g. MAE, RMSE, R²).

Test predictions are exported to the submissions/ folder in the required format:

carID,price
...


For example, the notebook currently writes:

output_path = "../submissions/group66_rf_submission.csv"


A trained model bundle (including the estimator and key preprocessing objects such as scaler, feature selector and encodings) is stored locally using Joblib:

import joblib
joblib.dump(bundle, "../models/car_price_bundle.joblib")


The models/ folder and Joblib artifacts are ignored by Git and are therefore not included in the repository, but the saving/loading logic is documented in the notebook.

