# Cars4You - Used Car Price Prediction (Group 66)

This repository contains the final project for Group 66, focused on predicting used car prices for the Cars4You platform using supervised machine learning (regression).

## Project structure

- `data/raw/` – original data (`train.csv`, `test.csv`)
- `data/processed/` – (optional) processed datasets or feature matrices
- `notebooks/` – Jupyter notebooks for EDA, modelling and evaluation  
  - main notebook: `ML_project.ipynb`
- `src/` – Python modules for data loading and modelling (e.g. `data_loading.py`)
- `submissions/` – final prediction files for Kaggle/assessment (e.g. `group66_rf_submission.csv`)
- `models/` – trained model bundle used for inference (car_price_bundle.joblib)
- `.gitignore` – files and folders that are excluded from version control
- `requirements.txt` – Python dependencies

## How to run

1. Make sure the files `train.csv` and `test.csv` are in `data/raw/`.

2. Install dependencies (ideally in a virtual environment):

   ```bash
   pip install -r requirements.txt
   
3. Open the notebook in notebooks/ (e.g. ML_project.ipynb) and run the cells in order.
The notebook assumes the data is available in data/raw/ using relative paths such as:

   ```python
   train_data = pd.read_csv("../data/raw/train.csv")
   test_data = pd.read_csv("../data/raw/test.csv")

## Final model and predictions

- The final regression model is trained on the Cars4You training data and evaluated using cross-validation with standard regression metrics (e.g. MAE, RMSE, R²).
- Test predictions are exported to the `submissions/` folder in the required format:

  ```text
  carID,price
  ...


For example, the notebook currently writes:

```python
output_path = "../submissions/group66_rf_submission.csv"
```


A trained model bundle (including the estimator and key preprocessing objects such as scaler, feature selector and encodings) is stored locally using Joblib:

```python 
import joblib
joblib.dump(bundle, "../models/car_price_bundle.joblib")
```

The trained bundle is included in this repository so the Streamlit app can run without retraining.

## Figures

The folder `figures/` contains the main visual outputs generated during the EDA and exploratory feature analysis, exported from the notebook for easier review.

It includes:
- Univariate distributions (e.g. `Histograms for each numerical variable.png`)
- Boxplots for numeric variables (e.g. `Boxplot of price.png`, `Boxplot of mileage.png`, `Boxplot of engineSize.png`)
- Bivariate relationships with price (e.g. `Relationship between Price and mileage.png`, `Relationship between Price and year.png`)
- Pairplot** overview (`Pairplot.png`)
- *Category-level summaries (e.g. frequency vs mean price plots such as `Relationship between Brand Frequency and Mean Price.png`)

## Streamlit demo (Open-ended / Deployment)

A simple Streamlit app is provided in `app/app.py` to predict the price for a custom user-defined car (single row input).  
It loads the pre-trained bundle from `models/car_price_bundle.joblib` and applies the same preprocessing and feature engineering used in training.

Run locally from the repository root:

```bash
pip install -r requirements.txt
streamlit run app/app.py


