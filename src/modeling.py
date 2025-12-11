import numpy as np
import pandas as pd

from collections import Counter
from typing import Iterable, Optional, Tuple, Dict, Any

from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def run_model(
    X: pd.DataFrame,
    y: pd.Series,
    scaler=None,
    model=None,
    fill_method: Optional[str] = "median",
    selector=None,  # optional: SelectKBest/SelectFromModel/RFE
    # encoding of the categorical variables
    fe_freq_cols: Iterable[str] = ("Brand", "model"),       # columns for frequency encoding
    ohe_cols: Iterable[str] = ("fuelType", "transmission"), # columns for OHE + mode
):
    """
    Fit a regression model on a given training set, learning all preprocessing
    steps using only (X, y) from the current fold.

    This function encompasses the full pipeline: imputation of categorical features, encoding
    (frequency and OHE), imputation of numerical features (controlled by fill_method), scaling,
    feature selection and model fitting.

    Intended to be called inside a cross-validation loop. All preprocessing
    parameters are learned inside each fold.

    Returns
    -------
    model, scaler, fill_values, selector_local, cat_modes, freq_maps,
    dummies_cols, feature_names, selected_feature_names
    """
    X_p = X.copy()

    # sanity check to clean tokens "na/none/nan" and replace with real NaN in OHE columns
    for c in ohe_cols:
        if c in X_p.columns:
            X_p[c] = X_p[c].astype("string").str.strip().str.lower()
            X_p[c] = X_p[c].replace(
                {
                    "nan": np.nan,
                    "none": np.nan,
                    "na": np.nan,
                    "": np.nan,
                    "unknown": "other",
                    "unk": "other",
                }
            )

    # 1) Categorical imputation with mode
    cat_modes: Dict[str, Any] = {}
    for c in ohe_cols:
        if c in X_p.columns:
            vc = X_p[c].dropna()
            cat_modes[c] = vc.mode().iloc[0] if not vc.empty else "__MISSING__"
            X_p[c] = X_p[c].fillna(cat_modes[c])

    for c in fe_freq_cols:
        if c in X_p.columns:
            vc = X_p[c].dropna()
            cat_modes[c] = vc.mode().iloc[0] if not vc.empty else "__MISSING__"
            X_p[c] = X_p[c].fillna(cat_modes[c])

    # 2) Frequency encoding
    freq_maps: Dict[str, Dict[Any, float]] = {}
    for c in fe_freq_cols:
        if c in X_p.columns:
            counts = X_p[c].value_counts(dropna=False)
            freq_maps[c] = counts.to_dict()
            X_p[f"{c}_freq"] = X_p[c].map(freq_maps[c]).astype(float)

    # drop the original categorical columns where we used frequency encoding
    X_p = X_p.drop(columns=[c for c in fe_freq_cols if c in X_p.columns], errors="ignore")

    # 3) One-Hot encoding
    dummies_cols = []
    present = [c for c in ohe_cols if c in X_p.columns]
    if present:
        X_cat = pd.get_dummies(X_p[present], drop_first=False)
        dummies_cols = X_cat.columns.tolist()
        X_p = pd.concat([X_p.drop(columns=present), X_cat], axis=1)

    # 5) Imputation with median/mean for numerical columns
    fill_values = None
    if fill_method is not None:
        num_cols = X_p.select_dtypes(include=["number"]).columns
        if fill_method == "median":
            fill_values = X_p[num_cols].median()
        elif fill_method == "mean":
            fill_values = X_p[num_cols].mean()
        else:
            raise ValueError("fill_method must be 'median', 'mean' or None")
        X_p[num_cols] = X_p[num_cols].fillna(fill_values)

    # 6) Save names of features before scaler/selector
    feature_names = X_p.columns.tolist()
    selected_feature_names = None

    # 7) Scaling
    if scaler is not None:
        X_p = scaler.fit_transform(X_p)

    # 8) Feature selection
    sel_local = clone(selector) if selector is not None else None
    if sel_local is not None:
        X_p = sel_local.fit_transform(X_p, y)
        if hasattr(sel_local, "get_support"):
            mask = sel_local.get_support()
            selected_feature_names = [
                n for n, keep in zip(feature_names, mask) if bool(keep)
            ]

    # 9) Model
    if model is None:
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()

    model.fit(X_p, y)

    return (
        model,
        scaler,
        fill_values,
        sel_local,
        cat_modes,
        freq_maps,
        dummies_cols,
        feature_names,
        selected_feature_names,
    )


def evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    scaler=None,
    fill_values=None,
    selector=None,
    cat_modes=None,
    freq_maps=None,
    dummies_cols=None,
    feature_names=None,  # not used but kept for API symmetry
    fe_freq_cols: Iterable[str] = ("Brand", "model"),
    ohe_cols: Iterable[str] = ("fuelType", "transmission"),
):
    """
    Apply the preprocessing learned in `run_model` to a new dataset and
    evaluate the fitted model on it.
    """
    X_p = X.copy()

    # Clean tokens "na/none/nan"
    for c in ohe_cols:
        if c in X_p.columns:
            X_p[c] = X_p[c].astype("string").str.strip().str.lower()
            X_p[c] = X_p[c].replace(
                {
                    "nan": np.nan,
                    "none": np.nan,
                    "na": np.nan,
                    "": np.nan,
                    "unknown": "other",
                    "unk": "other",
                }
            )

    # 1) Categorical imputation using train modes
    if cat_modes:
        for c, m in cat_modes.items():
            if c in X_p.columns:
                X_p[c] = X_p[c].fillna(m)

    # 2) Frequency encoding using train freq_maps
    if freq_maps:
        for c in fe_freq_cols:
            if c in X_p.columns and c in freq_maps:
                X_p[f"{c}_freq"] = (
                    X_p[c].map(freq_maps[c]).astype(float).fillna(0.0)
                )

    # drop original freq-encoded columns
    X_p = X_p.drop(columns=[c for c in fe_freq_cols if c in X_p.columns], errors="ignore")

    # 3) OHE aligned with train dummies
    if dummies_cols:
        present = [c for c in ohe_cols if c in X_p.columns]
        X_cat = (
            pd.get_dummies(X_p[present], drop_first=False)
            if present
            else pd.DataFrame(index=X_p.index)
        )
        X_cat = X_cat.reindex(columns=dummies_cols, fill_value=0)
        X_p = pd.concat([X_p.drop(columns=present), X_cat], axis=1)

    # 4) Numerical imputation using train fill_values
    if fill_values is not None:
        num_cols = X_p.select_dtypes(include=["number"]).columns
        X_p[num_cols] = X_p[num_cols].fillna(fill_values)

    # 5) Scaler (transform only)
    if scaler is not None:
        X_p = scaler.transform(X_p)

    # 6) Selector (transform only)
    if selector is not None:
        X_p = selector.transform(X_p)

    # 7) Metrics
    y_pred = model.predict(X_p)
    y_pred = np.asarray(y_pred).ravel()
    n, p = X_p.shape
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan

    return {"R2": r2, "Adj_R2": adj_r2, "RMSE": rmse, "MAE": mae, "MSE": mse}


def avg_scores(
    method,
    X: pd.DataFrame,
    y: pd.Series,
    scaler=None,
    model=None,
    fill_method: str = "median",
    selector=None,
    fe_freq_cols: Iterable[str] = ("Brand", "model"),
    ohe_cols: Iterable[str] = ("fuelType", "transmission"),
):
    """
    Evaluate a given model + preprocessing configuration using K-fold CV
    and compute average metrics across folds.
    """
    metrics_train = []
    metrics_val = []
    selection_bag = []

    first = True

    for tr_idx, va_idx in method.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        (
            mdl,
            sc,
            fills,
            sel,
            cat_modes,
            freq_maps,
            dummies,
            feat_names,
            sel_feat_names,
        ) = run_model(
            X_tr,
            y_tr,
            scaler=scaler,
            model=model,
            fill_method=fill_method,
            selector=selector,
            fe_freq_cols=fe_freq_cols,
            ohe_cols=ohe_cols,
        )

        if first:
            print(f"Nº features after preprocessing: {len(feat_names)}")
            if sel_feat_names is not None:
                print(
                    f"Nº features after feature selection: {len(sel_feat_names)}"
                )
            else:
                print("Without feature selection (selector=None).")
            first = False

        if sel_feat_names is not None:
            selection_bag.append(sel_feat_names)

        mtr = evaluate_model(
            X_tr,
            y_tr,
            mdl,
            scaler=sc,
            fill_values=fills,
            selector=sel,
            cat_modes=cat_modes,
            freq_maps=freq_maps,
            dummies_cols=dummies,
            fe_freq_cols=fe_freq_cols,
            ohe_cols=ohe_cols,
            feature_names=feat_names,
        )
        mva = evaluate_model(
            X_va,
            y_va,
            mdl,
            scaler=sc,
            fill_values=fills,
            selector=sel,
            cat_modes=cat_modes,
            freq_maps=freq_maps,
            dummies_cols=dummies,
            fe_freq_cols=fe_freq_cols,
            ohe_cols=ohe_cols,
            feature_names=feat_names,
        )

        metrics_train.append(mtr)
        metrics_val.append(mva)

    df_tr = pd.DataFrame(metrics_train).mean().to_frame("Train")
    df_va = pd.DataFrame(metrics_val).mean().to_frame("Validation")
    summary = pd.concat([df_tr, df_va], axis=1).sort_index()
    print("\n=== Metrics (Mean K-Fold) ===")
    print(summary.round(4))

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
    fe_freq_cols=('Brand','model'),
    ohe_cols=('fuelType','transmission'),
    use_log_target=False,
    clip_info=None
):
    """
    Apply the final trained preprocessing and model to the test set and return predictions.
    This function is the "deployment" counterpart of evaluate_model: it takes
    the raw test features and applies exactly the same transformations that
    were learned on the full training data when fitting the final model.
    Imputes missing values in categorical columns using the modes learned on train (cat_modes),
    applies frequency encoding using the frequency maps learned on train (freq_maps), applies one-hot encoding,
    imputes missing numerical features with the train-fold statistics (fill_values),
    applies the scaler (if any) and feature selector (if any) and calls `model.predict` on the
    final preprocessed test.
    """

    X_p = X.copy()

    # double check to make sure no ids are included in the model
    X_p = X_p.drop(
        columns=[c for c in ['car_ID', 'carID'] if c in X_p.columns],
        errors='ignore'
    )

    # Categorical imputation with the mode for the categorical columns
    if cat_modes:
        for c, m in cat_modes.items():
            if c in X_p.columns:
                X_p[c] = X_p[c].fillna(m)

    # Frequency encoding
    for c in fe_freq_cols:
        if c in X_p.columns and c in freq_maps:
            X_p[f'{c}_freq'] = X_p[c].map(freq_maps[c]).astype(float).fillna(0.0)
    X_p = X_p.drop(columns=[c for c in fe_freq_cols if c in X_p.columns], errors='ignore')

    # One-hot encoding
    present = [c for c in ohe_cols if c in X_p.columns]
    if dummies_cols:
        X_cat = (
            pd.get_dummies(X_p[present], drop_first=False)
            if present else pd.DataFrame(index=X_p.index)
        )
        X_cat = X_cat.reindex(columns=dummies_cols, fill_value=0)
        X_p = pd.concat([X_p.drop(columns=present), X_cat], axis=1)

    # Imputation of numerical columns
    if fill_values is not None:
        num_cols = X_p.select_dtypes(include=['number']).columns
        X_p[num_cols] = X_p[num_cols].fillna(fill_values)

    # Align with training feature order (safety)
    if feature_names is not None:
        X_p = X_p.reindex(columns=feature_names, fill_value=0.0)

    # Winsorization of some numerical columns
    if clip_info:
        for col, cap in clip_info.items():
            if col in X_p.columns:
                X_p[col] = np.minimum(X_p[col], cap)

    # scaler / selector
    if scaler is not None:
        X_p = scaler.transform(X_p)
    if selector is not None:
        X_p = selector.transform(X_p)

    # Predict (log or level)
    y_pred_raw = model.predict(X_p)
    y_pred = np.asarray(y_pred_raw).ravel()

    if use_log_target:
        y_pred = np.expm1(y_pred_raw)   # revert log(price+1) back to price

    return y_pred




