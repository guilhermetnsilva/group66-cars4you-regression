# Cars4You - Used Car Price Prediction (Group 66)

This repository contains the final project for Group 66, focused on predicting used car prices for the Cars4You platform using supervised machine learning (regression).

## Project structure

```text
.
├── app/
│   └── app.py                        # Streamlit demo (single-row prediction)
├── data/
│   └── raw/                          # Original provided datasets
│       ├── train.csv                 # Training set
│       └── test.csv                  # Test set
├── figures/                          # Exported plots from EDA / exploratory analysis
├── models/
│   └── car_price_bundle.joblib       # Trained model + preprocessing bundle for inference
├── notebooks/
│   └── group66_notebook.ipynb        # Main notebook (EDA → modeling → final submission)
├── src/                              # Python modules (data loading + modeling utilities)
│   ├── data_loading.py               # Data loading utilities
│   └── modeling.py                   # Training / evaluation / inference utilities
├── submissions/
│   └── group66_submission.csv        # Final predictions in required format
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Files/folders excluded from version control
└── README.md                         # Project documentation
```


## How to run

1. Make sure the files `train.csv` and `test.csv` are in `data/raw/`.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   
3. Open the notebook in notebooks/ (e.g. ML_project.ipynb) and run the cells in order.

## Final model and predictions

- The final regression model is trained on the Cars4You training data and evaluated using cross-validation with standard regression metrics (e.g. MAE, RMSE, R²).
- Test predictions are exported to the `submissions/` folder in the required format:

  ```text
  carID,price
  ...


For example, the notebook currently writes:

```python
output_path = "../submissions/group66_submission.csv"
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

## Streamlit Demo (Open-ended / Deployment)

A simple Streamlit app is provided in `app/app.py` to predict the price for a custom user-defined car (single row input).  
It loads the pre-trained bundle from `models/car_price_bundle.joblib` and applies the same preprocessing and feature engineering used in training.

Run locally from the repository root:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

