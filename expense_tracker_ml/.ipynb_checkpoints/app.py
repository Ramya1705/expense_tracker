import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from utils.data_handler import load_existing_csv, save_transactions
from utils.model_utils import train_spending_forecast_model, show_category_breakdown

st.title("💰 Smart Expense Tracker")

user = st.text_input("Enter your name:", "").strip().replace(" ", "_")

if user:
    df = load_existing_csv(user)

    st.subheader("📥 Add New Transaction")
    with st.form("transaction_form"):
        date = st.date_input("Date")
        type_ = st.selectbox("Type", ["Expense", "Income"])
        category = st.text_input("Category")
        amount = st.number_input("Amount", min_value=0.0, format="%.2f")
        comment = st.text_input("Comment (optional)")
        submitted = st.form_submit_button("Add Transaction")

        if submitted:
            new_transaction = {
                "Date": date.strftime("%d-%m-%Y"),
                "Type": type_,
                "Category": category,
                "Amount": amount,
                "Comment": comment
            }
            df = df.append(new_transaction, ignore_index=True)
            save_transactions(df, user)
            st.success("Transaction added successfully!")

    if not df.empty:
        st.subheader("📊 Category-wise Breakdown")
        show_category_breakdown(df)

        st.subheader("📈 Spending Forecast")
        forecast = train_spending_forecast_model(df, user)
        if forecast:
            st.write(f"📅 Forecast for Next Month: ₹{forecast:.2f}")
