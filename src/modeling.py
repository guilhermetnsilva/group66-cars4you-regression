import numpy as np
import pandas as pd

from collections import Counter
from typing import Iterable, Optional, Tuple, Dict, Any

from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_model(
    X, y,
    scaler=None,
    model=None,
    fill_method='median',
    selector=None,                         # optional: SelectKBest/SelectFromModel/RFE
    # encoding of the categorical variables
    fe_freq_cols=('model'),        # columns for frequency encoding
    ohe_cols=('fuelType','transmission', 'Brand'),   # columns for OHE + mode
    use_log_target=False,
    clip_cols=('mileage', 'tax', 'mpg'),          # columns to apply clipping of extreme values
    clip_quantile=0.995          # quantile used in clipping
):

    '''
    Fit a regression model on a given training set, learning all preprocessing
    steps using only (X, y) from the current fold.
    This function encompasses the full pipeline: imputation of categorical features, encoding
    (frequency and OHE), imputation of numerical features (controlled by fill_method), scaling,
    feature selection and model fitting.
    This function is intended to be called inside a cross-validation loop. All preprocessing
    parameters are learned inside each fold.
    '''

    X_p = X.copy()

    # sanity check to clean tokens "na/none/nan" and replace with real NaN in OHE columns
    for c in ohe_cols:
        if c in X_p.columns:
            X_p[c] = X_p[c].astype('string').str.strip().str.lower()
            X_p[c] = X_p[c].replace({'nan': np.nan, 'none': np.nan, 'na': np.nan, '': np.nan, 'unknown':'other','unk':'other'})

    # 1) Categorical imputation with mode: only for all categorical variables
    cat_modes = {}
    for c in ohe_cols:
      if c in X_p.columns:
          if c == "Brand":
              cat_modes[c] = "Unknown"
              X_p[c] = X_p[c].fillna("Unknown")
          else:
              vc = X_p[c].dropna()
              cat_modes[c] = vc.mode().iloc[0] if not vc.empty else '__MISSING__'
              X_p[c] = X_p[c].fillna(cat_modes[c])


    # 2) Frequency encoding, on high cardinality categorical variables
    freq_maps = {}
    for c in fe_freq_cols:
        if c in X_p.columns:
            counts = X_p[c].value_counts(dropna=False)
            freq_maps[c] = counts.to_dict()
            X_p[f'{c}_freq'] = X_p[c].map(freq_maps[c]).astype(float).fillna(0)

    # drop the old categorical columns, where we used frequency encoding
    X_p = X_p.drop(columns=[c for c in fe_freq_cols if c in X_p.columns], errors='ignore')

    # 3) One-Hot encoding in low cardinality features, after imputing with the mode
    dummies_cols = []
    present = [c for c in ohe_cols if c in X_p.columns]
    if present:
        X_cat = pd.get_dummies(X_p[present], drop_first=False)
        dummies_cols = X_cat.columns.tolist()
        X_p = pd.concat([X_p.drop(columns=present), X_cat], axis=1)

    # 4) Clipping outliers with a predefined treshold (quantile 99,2)
    clip_info = {}
    if clip_cols is not None:
      for col in clip_cols:
          if col in X_p.columns:
            cap = X_p[col].quantile(clip_quantile)
            X_p[col] = np.minimum(X_p[col], cap)
            clip_info[col] = cap

    # Applying a log transformation to mileage, to help center the distribution
    if "mileage" in X_p.columns:
        X_p["log_mileage"] = np.log1p(X_p["mileage"].clip(lower=0))
        X_p = X_p.drop(columns=["mileage"],errors='ignore')

    # 5) Imputation with the median for the numerical columns
    fill_values = None
    if fill_method is not None:
        num_cols = X_p.select_dtypes(include=['number']).columns
        if fill_method == 'median':
            fill_values = X_p[num_cols].median()
        elif fill_method == 'mean':
            fill_values = X_p[num_cols].mean()
        else:
            raise ValueError("fill_method must be 'median', 'mean' or None")
        X_p[num_cols] = X_p[num_cols].fillna(fill_values)

    # 6) Saving the names of the features before scaler/selector
    feature_names = X_p.columns.tolist()
    selected_feature_names = None

    # 7) Scaling
    if scaler is not None:
        X_p = scaler.fit_transform(X_p)

    # 8) Feature selection
    # we need to clone the selector in order to "restart" it; ensures ensure each fold uses a fresh, independent feature selector
    sel_local = clone(selector) if selector is not None else None
    if sel_local is not None:
        X_p = sel_local.fit_transform(X_p, y)
        if hasattr(sel_local, "get_support"):
            mask = sel_local.get_support() #recover the names of the selected features for inspection
            selected_feature_names = [n for n, keep in zip(feature_names, mask) if bool(keep)]

    # 9) Model
    # If no model is provided, fall back to a simple LinearRegression as default
    if model is None:
        model = LinearRegression()

    if use_log_target: # to do log of price (target)
        y_fit = np.log1p(y.to_numpy().ravel())
    else: #if not, use y as it is
        y_fit = y.to_numpy().ravel()

    model.fit(X_p, y_fit)

    return (model, scaler, fill_values, sel_local,
            cat_modes, freq_maps, dummies_cols, feature_names, selected_feature_names, clip_info)


def transform_X(
    X,
    scaler=None,
    fill_values=None,
    selector=None,
    cat_modes=None,
    freq_maps=None,
    dummies_cols=None,
    feature_names=None,
    fe_freq_cols=('model'),
    ohe_cols=('fuelType','transmission', 'Brand'),
    clip_info=None
):
    '''
    Applies all preprocessing steps learned (fitted) on training data. This function reproduces (only
    transforming) the same steps used during training: imputation of categorical missing values (using the
    mode of the training fold - cat_modes), frequency encoding (using frequency maps learning in training
    folds), OHE for the same categorical values (aligned with dummie columns of training), numerical
    imputation (fill_values with the median of training data columns), winsorization of columns with outliers
    (using clip_info from train),scaling and feature selection (feature names selected using training data).
    '''

    X_p = X.copy()

    # 0) sanity check in OHE columns, to prevent creating wrong dummt columns
    for c in ohe_cols:
        if c in X_p.columns:
            X_p[c] = X_p[c].astype('string').str.strip().str.lower()
            X_p[c] = X_p[c].replace(
                {'nan': np.nan, 'none': np.nan, 'na': np.nan,
                 'unknown': 'other', 'unk': 'other'}
            )

    # 1) Categorical imputation with cat_modes (from train)
    if cat_modes:
        for c, m in cat_modes.items():
            if c in X_p.columns:
                X_p[c] = X_p[c].fillna(m)

    # 2) Frequency encoding with freq_maps from train
    if freq_maps:
        for c in fe_freq_cols:
            if c in X_p.columns and c in freq_maps:
                X_p[f'{c}_freq'] = (
                    X_p[c].map(freq_maps[c]).astype(float).fillna(0.0)
                )

    # Drop the the original categorical columns (brand and model) which were freq-encoded
    X_p = X_p.drop(columns=[c for c in fe_freq_cols if c in X_p.columns],
                   errors='ignore')

    # 3) One Hot Encoding, aligned (reindexed) with dummies_cols
    if dummies_cols:
        present = [c for c in ohe_cols if c in X_p.columns]
        X_cat = (pd.get_dummies(X_p[present], drop_first=False)
                 if present else pd.DataFrame(index=X_p.index))
        X_cat = X_cat.reindex(columns=dummies_cols, fill_value=0)
        X_p = pd.concat([X_p.drop(columns=present, errors='ignore'), X_cat], axis=1)

    # 4) Winsorization of outliers using clip_info (values to clip from train)
    if clip_info:
        for col, cap in clip_info.items():
            if col in X_p.columns:
                X_p[col] = np.minimum(X_p[col], cap)

    # Applying a log transformation to mileage, to help center the distribution
    if "mileage" in X_p.columns:
        X_p["log_mileage"] = np.log1p(X_p["mileage"]).clip(lower=0)
        X_p = X_p.drop(columns=["mileage"],errors='ignore')

    # 5) Numerical imputation with fill_values (medians) from train
    if fill_values is not None:
        num_cols = X_p.select_dtypes(include=['number']).columns
        X_p[num_cols] = X_p[num_cols].fillna(fill_values)

    # 6) Aligning all columns with feature_names of train
    if feature_names is not None:
        X_p = X_p.reindex(columns=feature_names, fill_value=0.0)

    # 7) Scaling
    if scaler is not None:
        X_p = scaler.transform(X_p)

    # 8) Feature selection (only transform)
    if selector is not None:
        X_p = selector.transform(X_p)

    return X_p


def evaluate_model(
    X, y, model,
    scaler=None, fill_values=None, selector=None,
    cat_modes=None, freq_maps=None, dummies_cols=None,
    feature_names=None,
    fe_freq_cols= ('Brand','model'),
    ohe_cols=('fuelType','transmission'),
    use_log_target=False,
    clip_info=None,
    brand_mean=None,
    global_mean=None):

    '''
    This function is meant to be used inside cross-validation: for each fold
    you pass the model and all preprocessing objects that were fitted on the
    training split of that fold.

    It applies the preprocessing learned in `run_model` to a new dataset through the function transform_X,
    that replicates all preprocessing steps, predicts and evaluates models using the preprocessed data.

    Returns a dictionaire with regression metrics on the original scale (libras)
    '''

    X_p = X.copy()

    X_p = transform_X(
        X,
        scaler=scaler,
        fill_values=fill_values,
        selector=selector,
        cat_modes=cat_modes,
        freq_maps=freq_maps,
        dummies_cols=dummies_cols,
        feature_names=feature_names,
        fe_freq_cols=fe_freq_cols,
        ohe_cols=ohe_cols,
        clip_info=clip_info
      )

    # 7) Evaluation metrics
    y_pred_raw = model.predict(X_p)
    y_pred_raw = np.asarray(y_pred_raw).ravel()

    if use_log_target:
        # model used log(price), we have to convert to the original scale
        y_pred = np.expm1(y_pred_raw)
    else:
        y_pred = y_pred_raw

    y_true = np.asarray(y).ravel()
    n, p = X_p.shape

    r2  = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan

    return {"R2": r2, "Adj_R2": adj_r2, "RMSE": rmse, "MAE": mae}


def avg_scores(
    method, X, y,
    scaler=None, model=None, fill_method='median',
    selector=None,
    fe_freq_cols= ('Brand','model'),
    ohe_cols=('fuelType','transmission'),
    use_log_target=True, # we define wether to work with log(price) or price
    clip_cols=('mileage', 'tax','mpg', 'engineSize'),
    clip_quantile=0.992
):

    '''
    Evaluates a given model and preprocessing configuration using K-fold cross-validation and
    compute average metrics across folds.
    By each split produced by 'method' (kfold) this function splits X, y into train and validation sets,
    calls run_model on the training split to learn all preprocessing steps and fit the model. Consequently,
    it evaluate_model is called on both traning and validation to obtain the scores of predictions.
    Stores the training and validation metrics for each fold and the list of selected features if a
    selector is used.
    Returns the mean of each metric across folds for train an validation through a summary table with all
    metrics and prints the seletion frequency of each feature across folds.
    '''

    metrics_train = []
    metrics_val = []
    selection_bag =[] # list of selected features per fold

    first = True

    # splitting the data
    for tr_idx, va_idx in method.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]


        # fit to train in each fold
        (mdl, sc, fills, sel, cat_modes, freq_maps,
        dummies, feat_names, sel_feat_names, clip_info) = run_model(
            X_tr, y_tr, scaler=scaler, model=model,
            fill_method=fill_method, selector=selector,
            fe_freq_cols=fe_freq_cols, ohe_cols=ohe_cols, use_log_target=use_log_target,
            clip_cols=clip_cols, clip_quantile=clip_quantile
        )

        if first:
          print(f"Nº features after preprocessing: {len(feat_names)}")
          if sel_feat_names is not None:
            print(f"Nº features after feature selection: {len(sel_feat_names)}")
          else:
            print("Without feature selection (selector=None).")
          first = False

        if sel_feat_names is not None:
            selection_bag.append(sel_feat_names)

        # metrics train and validation
        mtr = evaluate_model(
            X_tr, y_tr, mdl,
            scaler=sc, fill_values=fills, selector=sel,
            cat_modes=cat_modes, freq_maps=freq_maps, dummies_cols=dummies,
            fe_freq_cols=fe_freq_cols, ohe_cols=ohe_cols, feature_names=feat_names,
            use_log_target=use_log_target, clip_info=clip_info
        )
        mva = evaluate_model(
            X_va, y_va, mdl,
            scaler=sc, fill_values=fills, selector=sel,
            cat_modes=cat_modes, freq_maps=freq_maps, dummies_cols=dummies,
            fe_freq_cols=fe_freq_cols, ohe_cols=ohe_cols, feature_names=feat_names,
            use_log_target=use_log_target, clip_info=clip_info)

        metrics_train.append(mtr)
        metrics_val.append(mva)

    # averaging the metrics
    df_tr = pd.DataFrame(metrics_train).mean().to_frame("Train")
    df_va = pd.DataFrame(metrics_val).mean().to_frame("Validation")
    summary = pd.concat([df_tr, df_va], axis=1).sort_index()
    print("\n=== Metrics (Mean K-Fold) ===")
    print(summary.round(4))

    # Frequency of selection, when using a selector, for feature selection
    selection_freq = None
    if selection_bag:
        cnt = Counter([f for fold_list in selection_bag for f in fold_list])
        total_folds = len(selection_bag)
        selection_freq = (pd.Series(cnt).sort_values(ascending=False) / total_folds)
        print("\n=== Feature Selection frequency (proportion of folds) ===")
        print(selection_freq.head(30).round(2))


    return summary, selection_freq


def predict_on_test(
    X, model,
    scaler=None, fill_values=None, selector=None,
    cat_modes=None, freq_maps=None, dummies_cols=None,
    feature_names=None,
    fe_freq_cols=('model',),
    ohe_cols=('fuelType','transmission','Brand'),
    use_log_target=False,
    clip_info=None
):

    '''
    Apply the final trained preprocessing and model to the test set and return predictions.
    This function is the "deployment" counterpart of evaluate_model: it takes
    the raw test features and applies exactly the same transformations that
    were learned on the full training data when fitting the final model.
    Imputes missing values in categorical columns using the modes learned on train (cat_modes),
    applies frequency encoding using the frequency maps learned on train (freq_maps), applies one-hot encoding,
    imputes missing numerical features with the train-fold statistics (fill_values),
    applies the scaler (if any) and feature selector (if any) and calls `model.predict` on the
    final preprocessed test
    '''

    X_p = X.copy()

    # double check to make sure no ids are included in the model
    X_p = X_p.drop(columns=[c for c in ['car_ID','carID'] if c in X_p.columns],
                 errors='ignore')

    X_p = transform_X(
        X_p,
        scaler=scaler,
        fill_values=fill_values,
        selector=selector,
        cat_modes=cat_modes,
        freq_maps=freq_maps,
        dummies_cols=dummies_cols,
        feature_names=feature_names,
        fe_freq_cols=fe_freq_cols,
        ohe_cols=ohe_cols,
        clip_info=clip_info
    )

    # predict using log or level
    y_pred_raw = model.predict(X_p)
    y_pred = np.asarray(y_pred_raw).ravel()

    if use_log_target:
        y_pred = np.expm1(y_pred_raw)   # revert log(price+1) back to price

    return y_pred

