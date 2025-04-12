import streamlit as st
import pandas as pd
from utils.forecast import train_spending_forecast_model
from utils.data_handler import load_existing_csv, save_csv
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Expense Tracker", layout="centered")

st.title("💸 Smart Expense Tracker")

user = st.text_input("Enter your name").strip().replace(" ", "_")

if user:
    df = load_existing_csv(user)

    with st.form("add_transaction"):
        st.subheader("➕ Add New Transaction")
        date = st.date_input("Date")
        type_ = st.selectbox("Type", ["Expense", "Income"])
        category = st.text_input("Category")
        amount = st.number_input("Amount", min_value=0.0)
        comment = st.text_input("Comment (optional)")

        submitted = st.form_submit_button("Add Transaction")

        if submitted:
            new_row = {
                "Date": date.strftime("%d-%m-%Y"),
                "Type": type_,
                "Category": category,
                "Amount": amount,
                "Comment": comment
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_csv(df, user)
            st.success("Transaction added!")

    st.subheader("📂 All Transactions")
    st.dataframe(df)

    if not df.empty:
        st.subheader("📊 Category-wise Breakdown")
        expense_df = df[df["Type"].str.lower() == "expense"]
        if not expense_df.empty:
            cat_summary = expense_df.groupby("Category")["Amount"].sum()
            st.bar_chart(cat_summary)

        st.subheader("📈 Monthly Forecast")
        forecast, r2_score = train_spending_forecast_model(df, user)
        if forecast:
            st.write(f"📅 **Next Month's Forecast**: ₹{forecast:.2f}")
            st.write(f"📊 **Model Accuracy (R² Score)**: {r2_score:.4f}")
        else:
            st.warning("Not enough data to forecast.")
