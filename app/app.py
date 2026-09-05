import streamlit as st
import pandas as pd
import joblib
import os
# =========================================================
# LOAD MODEL
# =========================================================
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "fraud_detection_model.pkl"
)
model = joblib.load(MODEL_PATH)
# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)
# =========================================================
# TITLE
# =========================================================
st.title("💳 Credit Card Fraud Detection System")
st.write(
    "AI-powered transaction risk analysis using a trained "
    "Random Forest machine learning model."
)
st.divider()
# =========================================================
# INPUT SECTION
# =========================================================
st.subheader("📝 Transaction Details")
col1, col2 = st.columns(2)
with col1:
    amount = st.number_input(
        "💰 Transaction Amount",
        min_value=0.0,
        value=5000.0
    )
    transaction_hour = st.slider(
        "🕐 Transaction Hour",
        min_value=0,
        max_value=23,
        value=14
    )
    foreign_transaction = st.selectbox(
        "🌍 Foreign Transaction?",
        ["No", "Yes"]
    )
    location_mismatch = st.selectbox(
        "📍 Location Mismatch?",
        ["No", "Yes"]
    )
with col2:
    device_trust_score = st.slider(
        "📱 Device Trust Score",
        min_value=0,
        max_value=100,
        value=80
    )
    velocity_last_24h = st.number_input(
        "🔄 Transactions in Last 24 Hours",
        min_value=0,
        value=2
    )
    cardholder_age = st.number_input(
        "👤 Cardholder Age",
        min_value=18,
        max_value=100,
        value=25
    )
    merchant_category = st.selectbox(
        "🛒 Merchant Category",
        ["Electronics", "Food", "Grocery", "Travel"]
    )
# =========================================================
# CONVERT INPUTS
# =========================================================
foreign_value = 1 if foreign_transaction == "Yes" else 0
location_value = 1 if location_mismatch == "Yes" else 0
electronics = 1 if merchant_category == "Electronics" else 0
food = 1 if merchant_category == "Food" else 0
grocery = 1 if merchant_category == "Grocery" else 0
travel = 1 if merchant_category == "Travel" else 0
# =========================================================
# CREATE TRANSACTION DATA
# =========================================================
transaction = pd.DataFrame([{
    "amount": amount,
    "transaction_hour": transaction_hour,
    "foreign_transaction": foreign_value,
    "location_mismatch": location_value,
    "device_trust_score": device_trust_score,
    "velocity_last_24h": velocity_last_24h,
    "cardholder_age": cardholder_age,
    "merchant_category_Electronics": electronics,
    "merchant_category_Food": food,
    "merchant_category_Grocery": grocery,
    "merchant_category_Travel": travel
}])
# =========================================================
# PREDICTION
# =========================================================
st.divider()
if st.button(
    "🔍 ANALYZE TRANSACTION",
    use_container_width=True
):
    # Match exact model features
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        transaction = transaction.reindex(
            columns=expected_features,
            fill_value=0
        )
    prediction = model.predict(transaction)[0]
    probability = model.predict_proba(
        transaction
    )[0][1] * 100

    # Save transaction history
    history_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "transaction_history.csv"
    )

    history_row = pd.DataFrame([{
        "amount": amount,
        "transaction_hour": transaction_hour,
        "foreign_transaction": foreign_value,
        "location_mismatch": location_value,
        "device_trust_score": device_trust_score,
        "velocity_last_24h": velocity_last_24h,
        "cardholder_age": cardholder_age,
        "merchant_category": merchant_category,
        "fraud_probability": round(probability, 2),
        "prediction": "Fraud" if prediction == 1 else "Genuine"
    }])

    # Add the new transaction to the CSV
    if os.path.exists(history_path):
        history_row.to_csv(
            history_path,
            mode="a",
            header=False,
            index=False
        )
    else:
        history_row.to_csv(
            history_path,
            mode="w",
            header=True,
            index=False
        )
    # =====================================================
    # RESULT
    # =====================================================
    st.subheader("📊 Fraud Detection Result")
    # Risk levels
    if probability < 30:
        risk = "LOW RISK"
        st.success(
            "🟢 GENUINE TRANSACTION"
        )
    elif probability < 70:
        risk = "MEDIUM RISK"
        st.warning(
            "🟠 SUSPICIOUS TRANSACTION"
        )
    else:
        risk = "HIGH RISK"
        st.error(
            "🔴 FRAUDULENT TRANSACTION"
        )
    # =====================================================
    # RESULT METRICS
    # =====================================================
    result_col1, result_col2, result_col3 = st.columns(3)
    with result_col1:
        st.metric(
            "Fraud Probability",
            f"{probability:.2f}%"
        )
    with result_col2:
        st.metric(
            "Risk Level",
            risk
        )
    with result_col3:
        if prediction == 1:
            result = "FRAUD"
        else:
            result = "GENUINE"
        st.metric(
            "Prediction",
            result
        )
    # =====================================================
    # PROBABILITY BAR
    # =====================================================
    st.write("### Fraud Probability")
    st.progress(
        min(int(probability), 100)
    )
    # =====================================================
    # RECOMMENDATION
    # =====================================================
    if probability >= 70:
        st.error(
            "⚠️ Recommendation: Block or manually review "
            "this transaction."
        )
    elif probability >= 30:
        st.warning(
            "⚠️ Recommendation: Perform additional "
            "verification before approving."
        )
    else:
        st.success(
            "✅ Recommendation: Transaction appears "
            "low risk."
        )
