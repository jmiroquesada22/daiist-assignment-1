"""Assignment 1: training pipeline (Telco customer churn)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
DATA_PATH = "data/telco.csv"


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


def main():
    df = load_and_clean()
    train_df, val_df, test_df = split_data(df)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name}: {len(part)} rows, churn rate {part['Churn'].mean():.3f}")


if __name__ == "__main__":
    main()
