import joblib
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
# ==========================================
# MODEL PERFORMANCE DATA
# ==========================================

ROC_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "roc_data.csv"
)

roc_data = pd.read_csv(ROC_PATH)

y_true = roc_data["actual"]
y_score = roc_data["probability"]

auc_score = roc_auc_score(y_true, y_score)
# Calculate fraud recall
PREDICTIONS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "model_predictions.csv"
)

model_predictions = pd.read_csv(PREDICTIONS_PATH)

cm = confusion_matrix(
    model_predictions["actual"],
    model_predictions["predicted"]
)

TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

recall_score = TP / (TP + FN)

st.set_page_config(
    page_title="Fraud Detection Analytics",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection Analytics")
st.markdown(
    "Real-time transaction monitoring, fraud risk analysis, "
    "and machine learning performance insights."
)
st.info(
    f"🤖 Model: Random Forest with SMOTE | "
   f"ROC-AUC: {auc_score:.2f} | Fraud Recall: {recall_score * 100:.2f}%"
)
st.caption(
    "🎯 Model objective: detect fraudulent transactions while minimizing missed fraud cases."
)

with st.expander("📌 About This Project"):
    st.write(
        "This system uses Machine Learning to detect potentially "
        "fraudulent credit card transactions."
    )
    st.write(
        "The Random Forest model is trained using SMOTE to handle "
        "the highly imbalanced fraud dataset."
    )
    st.write(
        "Transactions are classified based on fraud probability "
        "and categorized into Low, Medium, or High Risk."
    )

st.success("🟢 Fraud Detection System: Online")

# ==========================================
# LOAD TRANSACTION HISTORY
# ==========================================

file_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "transaction_history.csv"
)

if not os.path.exists(file_path):

    st.error("Transaction history file not found.")

    st.stop()


df = pd.read_csv(file_path)
# ==========================================
# CREATE RISK LEVEL
# ==========================================

def get_risk_level(probability):
    if probability >= 70:
        return "High Risk"
    elif probability >= 40:
        return "Medium Risk"
    else:
        return "Low Risk"

df["risk_level"] = df["fraud_probability"].apply(get_risk_level)

# ==========================================
# DASHBOARD FILTERS
# ==========================================

st.sidebar.title("💳 Fraud Detection")
st.sidebar.caption("⚙️ Dashboard Controls")
st.sidebar.markdown("---")
st.sidebar.info(
    "🤖 Random Forest + SMOTE\n\n"
    "Detects potentially fraudulent "
    "credit card transactions using "
    "machine learning."
)
if st.sidebar.button("🔄 Refresh Dashboard"):
    st.rerun()

st.sidebar.header("🔎 Filters")

prediction_filter = st.sidebar.selectbox(
    "Transaction Type",
    ["All", "Fraud", "Genuine"]
)

category_filter = st.sidebar.selectbox(
    "Merchant Category",
    ["All"] + sorted(df["merchant_category"].unique().tolist())
)
risk_filter = st.sidebar.selectbox(
    "Risk Level",
    ["All", "Low Risk", "Medium Risk", "High Risk"]
)

if prediction_filter != "All":
    df = df[df["prediction"] == prediction_filter]

if category_filter != "All":
    df = df[df["merchant_category"] == category_filter]

if risk_filter != "All":
    df = df[df["risk_level"] == risk_filter]

# ==========================================
# KEY METRICS
# ==========================================

total_transactions = len(df)

fraud_transactions = len(
    df[df["prediction"] == "Fraud"]
)

genuine_transactions = len(
    df[df["prediction"] == "Genuine"]
)
total_amount = df["amount"].sum()

if total_transactions > 0:
    fraud_rate = (
        fraud_transactions / total_transactions
    ) * 100
else:
    fraud_rate = 0


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Transactions",
        total_transactions
    )

with col2:
    st.metric(
        "🚨 Fraud Transactions",
        fraud_transactions
    )

with col3:
    st.metric(
        "✅ Genuine Transactions",
        genuine_transactions
    )

with col4:
    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    if fraud_rate >= 10:
        st.warning("⚠️ High fraud activity")
    else:
        st.caption("✅ Fraud activity within monitored level")
with col5:
    st.metric(
        "💰 Total Amount",
        f"₹{total_amount:,.2f}"
    )
st.caption(
    "📌 Metrics above are based on the current transaction history."
)

st.divider()


# ==========================================
# FRAUD VS GENUINE
# ==========================================

st.subheader("📊 Fraud vs Genuine Transactions")

prediction_counts = df["prediction"].value_counts()

prediction_percentage = (
    prediction_counts / prediction_counts.sum()
) * 100

st.bar_chart(
    prediction_percentage,
    height=350
)
st.caption(
    f"📌 Genuine: {prediction_counts.get('Genuine', 0)} | "
    f"Fraud: {prediction_counts.get('Fraud', 0)} transactions"
)

# ==========================================
# TRANSACTIONS BY MERCHANT CATEGORY
# ==========================================

st.subheader("🛒 Transactions by Merchant Category")

category_counts = df["merchant_category"].value_counts()

st.bar_chart(
    category_counts,
    height=350
)

# ==========================================
# FRAUD BY MERCHANT CATEGORY
# ==========================================

st.subheader("🚨 Fraud by Merchant Category")

fraud_category = df[
    df["prediction"] == "Fraud"
]["merchant_category"].value_counts().reindex(
    ["Electronics", "Food", "Grocery", "Travel"],
    fill_value=0
)
st.bar_chart(
    fraud_category,
    height=350
)
# ==========================================
# RISK LEVEL DISTRIBUTION
# ==========================================

st.subheader("⚠️ Risk Level Distribution")

risk_counts = df["risk_level"].value_counts().reindex(
    ["Low Risk", "Medium Risk", "High Risk"],
    fill_value=0
)
st.bar_chart(
    risk_counts,
    height=350
)
st.caption(
    f"📌 Low: {risk_counts.get('Low Risk', 0)} | "
    f"Medium: {risk_counts.get('Medium Risk', 0)} | "
    f"High: {risk_counts.get('High Risk', 0)}"
)

# ==========================================
# TRANSACTIONS BY HOUR
# ==========================================

st.subheader("🕐 Transactions by Hour")

hour_counts = df["transaction_hour"].value_counts().sort_index()

st.line_chart(
    hour_counts,
    height=300
)
st.caption(
    "📌 Shows the number of transactions recorded at each hour of the day."
)

# ==========================================
# FRAUD PROBABILITY
# ==========================================

st.subheader("🤖 Fraud Probability")

st.line_chart(
    df["fraud_probability"],
    height=350
)
st.caption(
    "📌 Shows the fraud probability (%) assigned to each transaction by the model."
)
# ==========================================
# HIGH-RISK TRANSACTIONS
# ==========================================

st.subheader("🚨 High-Risk Transactions")

high_risk = df[
    df["risk_level"]=="High Risk"
]

st.metric(
    "🚨 High-Risk Transaction Count",
    len(high_risk)
)
st.caption(
    "📌 Transactions classified as High Risk require immediate attention."
)
st.write("High-risk transactions requiring attention:")

st.dataframe(
    high_risk[
        [
            "amount",
            "transaction_hour",
            "merchant_category",
            "fraud_probability",
            "prediction"
        ]
    ],
    use_container_width=True
)
# ==========================================
# RECENT TRANSACTIONS
# ==========================================

st.divider()

st.subheader("🧾 Recent Transactions")

st.dataframe(
    df.tail(10)[
        [
            "amount",
            "transaction_hour",
            "merchant_category",
            "fraud_probability",
            "risk_level",
            "prediction"
        ]
    ],
    use_container_width=True,
    hide_index=True
)
st.caption(
    "📌 Displays the 10 most recent transactions recorded by the system."
)
# ==========================================
# DOWNLOAD REPORT
# ==========================================

st.divider()

st.subheader("📥 Download Transaction Report")

csv_data = df.to_csv(index=False)

st.download_button(
    label="⬇️ Download Transaction Report",
    data=csv_data,
    file_name="fraud_transaction_report.csv",
    mime="text/csv",
    use_container_width=True
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

st.divider()
st.subheader("📊 Confusion Matrix")

st.write("Model prediction results:")

# Load actual model predictions
PREDICTIONS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "model_predictions.csv"
)

model_predictions = pd.read_csv(PREDICTIONS_PATH)

cm = confusion_matrix(
    model_predictions["actual"],
    model_predictions["predicted"]
)
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

# ==========================================
# MODEL PERFORMANCE SUMMARY
# ==========================================

st.divider()
st.subheader("🎯 Model Performance")

performance_col1, performance_col2, performance_col3, performance_col4 = st.columns(4)

with performance_col1:
    st.metric("Precision", f"{precision * 100:.2f}%")

with performance_col2:
    st.metric("Recall", f"{recall * 100:.2f}%")

with performance_col3:
    st.metric("F1 Score", f"{f1_score * 100:.2f}%")

with performance_col4:
    st.metric("ROC-AUC", f"{auc_score:.2f}")


fig, ax = plt.subplots(figsize=(8, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Genuine", "Fraud"],
    yticklabels=["Genuine", "Fraud"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - SMOTE Random Forest")

st.pyplot(fig)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

st.divider()
st.subheader("🔍 Feature Importance")

model_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "fraud_detection_model.pkl"
)

with open(model_path, "rb") as file:
    model = joblib.load(file)

importance = model.feature_importances_

if hasattr(model, "feature_names_in_"):
    feature_names = model.feature_names_in_
else:
    feature_names = [
        "Amount",
        "Transaction Hour",
        "Foreign Transaction",
        "Location Mismatch",
        "Device Trust",
        "Velocity Last 24h",
        "Cardholder Age",
        "Merchant Category - Electronics",
        "Merchant Category - Food",
        "Merchant Category - Grocery",
        "Merchant Category - Travel"
    ]

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

st.bar_chart(
    feature_importance.set_index("Feature"),
    height=400
)
# ==========================================
# FRAUD ALERT SYSTEM
# ==========================================

st.divider()
st.subheader("🚨 Fraud Alerts")

high_risk_alerts = df[df["risk_level"]=="High Risk"]
high_risk_amount = high_risk_alerts["amount"].sum()

if len(high_risk_alerts) > 0:
    st.metric(
    "💰 High-Risk Amount",
    f"₹{high_risk_amount:,.2f}"
    )
    st.error(
    f"🚨 ALERT: {len(high_risk_alerts)} high-risk "
    f"transaction(s) detected. "
    f"Immediate verification recommended."
)
else:
    st.success(
        "✅ No high-risk transactions detected."
    )
# ==========================================
# HIGH-RISK TRANSACTION DETAILS
# ==========================================

if len(high_risk_alerts) > 0:
    st.write("### 🔎 High-Risk Transaction Details")

    st.dataframe(
        high_risk_alerts[
            [
                "amount",
                "transaction_hour",
                "merchant_category",
                "fraud_probability",
                "risk_level",
                "prediction"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )
# ==========================================
# ROC CURVE AND AUC
# ==========================================

st.divider()
st.subheader("📈 ROC Curve")

ROC_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "roc_data.csv"
)

roc_data = pd.read_csv(ROC_PATH)

y_true = roc_data["actual"]
y_score = roc_data["probability"]

fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc_score = roc_auc_score(y_true, y_score)

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
ax.plot([0, 1], [0, 1], linestyle="--")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - SMOTE Random Forest")
ax.legend()

st.pyplot(fig)
st.caption(
    f"📌 ROC-AUC Score: {auc_score:.2f} — higher values indicate better model discrimination."
)

st.divider()

st.caption(
    "💳 Credit Card Fraud Detection | Random Forest + SMOTE | "
    "Machine Learning Analytics Dashboard"
)