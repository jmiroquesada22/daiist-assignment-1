"""Assignment 1: training pipeline (Telco customer churn)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
DATA_PATH = "data/telco.csv"
ARTIFACTS = Path("artifacts")

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]
NUM_COLS = ["tenure_log", "MonthlyCharges", "SeniorCitizen", "is_echeck", "contract_ord"]
CAT_COLS = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService"] + ADDONS + ["PaperlessBilling"]


def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path)

    # customerID is an identifier, so it carries no signal
    df = df.drop(columns=["customerID"])

    # TotalCharges is text because a few rows hold blank strings.
    # Check that those are all customers with tenure == 0 before filling.
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank = total.isna()
    assert (df.loc[blank, "tenure"] == 0).all(), "blank TotalCharges with tenure > 0"
    df["TotalCharges"] = total.fillna(0.0)

    # Target: 1 = churned, 0 = stayed
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def split_data(df):
    # 60% train / 20% validation / 20% test, stratified on the target
    train_df, temp_df = train_test_split(
        df, test_size=0.4, stratify=df["Churn"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["Churn"], random_state=SEED
    )
    return train_df, val_df, test_df


def engineer_features(df):
    """Row-wise transformations only: nothing here learns from the data."""
    X = pd.DataFrame(index=df.index)
    # log1p: churn falls steeply in the first year then flattens; log(1+0)=0 is safe
    X["tenure_log"] = np.log1p(df["tenure"])
    X["MonthlyCharges"] = df["MonthlyCharges"]
    X["SeniorCitizen"] = df["SeniorCitizen"]
    # Electronic check churns ~45%, every other method 15-19%: one flag captures it
    X["is_echeck"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    # Churn falls monotonically 43% -> 11% -> 3% across contract lengths
    X["contract_ord"] = df["Contract"].map(
        {"Month-to-month": 0, "One year": 1, "Two year": 2})
    for c in ["gender", "Partner", "Dependents", "PhoneService", "InternetService",
              "PaperlessBilling"]:
        X[c] = df[c]
    # "No internet/phone service" duplicates InternetService/PhoneService, so merge into "No"
    for c in ADDONS + ["MultipleLines"]:
        X[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})
    return X[NUM_COLS + CAT_COLS]


def build_preprocessor():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), CAT_COLS),
        ],
        sparse_threshold=0,  # always return a dense array (PyTorch needs dense)
    )


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    df = load_and_clean()
    train_df, val_df, test_df = split_data(df)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name}: {len(part)} rows, churn rate {part['Churn'].mean():.3f}")

    # Fit the scaler/encoder on TRAIN only, then apply to all three splits
    prep = build_preprocessor()
    X_train = prep.fit_transform(engineer_features(train_df)).astype(np.float32)
    X_val = prep.transform(engineer_features(val_df)).astype(np.float32)
    X_test = prep.transform(engineer_features(test_df)).astype(np.float32)
    y_train = train_df["Churn"].to_numpy()
    y_val = val_df["Churn"].to_numpy()
    y_test = test_df["Churn"].to_numpy()
    print("feature matrix shapes:", X_train.shape, X_val.shape, X_test.shape)
    joblib.dump(prep, ARTIFACTS / "preprocessor.joblib")

    # Sanity check only (not the final model): should be near AUC 0.863, log-loss 0.396
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    p = clf.predict_proba(X_val)[:, 1]
    print(f"sanity val AUC {roc_auc_score(y_val, p):.4f}, log-loss {log_loss(y_val, p):.4f}")


if __name__ == "__main__":
    main()
