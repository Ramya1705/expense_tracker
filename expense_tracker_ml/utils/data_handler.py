import os
import pandas as pd

def load_existing_csv(user):
    csv_path = f"user_data/{user}_transactions.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=["Date", "Type", "Category", "Amount", "Comment"])

def save_csv(df, user):
    os.makedirs("user_data", exist_ok=True)
    df.to_csv(f"user_data/{user}_transactions.csv", index=False)
