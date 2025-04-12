import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def train_spending_forecast_model(df, user):
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], format='%d-%m-%Y', errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Type"].str.lower() == "expense"]

    if df.empty:
        return None, None

    monthly_expense = df.groupby(df["Date"].dt.to_period("M"))["Amount"].sum()
    monthly_expense.index = monthly_expense.index.to_timestamp()
    monthly_expense = monthly_expense.sort_index()
    monthly_expense_df = monthly_expense.reset_index()
    monthly_expense_df["MonthIndex"] = range(len(monthly_expense_df))

    X = monthly_expense_df[["MonthIndex"]]
    y = monthly_expense_df["Amount"]

    model = LinearRegression()
    model.fit(X, y)

    r2_score = model.score(X, y)
    next_month_index = pd.DataFrame([[len(monthly_expense_df)]], columns=["MonthIndex"])
    forecast = model.predict(next_month_index)[0]

    os.makedirs(f"ml_models/{user}", exist_ok=True)
    model_path = f"ml_models/{user}/spending_forecast_model.pkl"
    joblib.dump(model, model_path)

    return forecast, r2_score
