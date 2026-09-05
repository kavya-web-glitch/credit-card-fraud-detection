import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/fraud_detection_model.pkl")

print("=" * 45)
print("       CREDIT CARD FRAUD DETECTION")
print("=" * 45)

# Get transaction details
amount = float(input("Enter transaction amount: "))

foreign = int(input("Foreign transaction? (0=No, 1=Yes): "))

location = int(input("Location mismatch? (0=No, 1=Yes): "))

device = float(input("Enter device trust score: "))

velocity = int(input("Transactions in last 24 hours: "))

age = int(input("Enter cardholder age: "))

print("\nMerchant Category")
print("1 - Electronics")
print("2 - Food")
print("3 - Grocery")
print("4 - Travel")

merchant = int(input("Choose merchant category (1-4): "))

transaction_hour = int(input("Enter transaction hour (0-23): "))


# Merchant category values
electronics = 0
food = 0
grocery = 0
travel = 0

if merchant == 1:
    electronics = 1
elif merchant == 2:
    food = 1
elif merchant == 3:
    grocery = 1
elif merchant == 4:
    travel = 1
else:
    print("Invalid merchant category.")
    exit()


# Create transaction using the SAME feature names
transaction = pd.DataFrame([{
    "amount": amount,
    "transaction_hour": transaction_hour,
    "foreign_transaction": foreign,
    "location_mismatch": location,
    "device_trust_score": device,
    "velocity_last_24h": velocity,
    "cardholder_age": age,
    "merchant_category_Electronics": electronics,
    "merchant_category_Food": food,
    "merchant_category_Grocery": grocery,
    "merchant_category_Travel": travel
}])


# Make sure columns are in exactly the same order
transaction = transaction[model.feature_names_in_]


# Prediction
prediction = model.predict(transaction)[0]

# Fraud probability
probability = model.predict_proba(transaction)[0][1] * 100


print("\n" + "=" * 45)

if prediction == 1:
    print("🚨 FRAUDULENT TRANSACTION DETECTED")
else:
    print("✅ GENUINE TRANSACTION")

print(f"Fraud Probability: {probability:.2f}%")

print("=" * 45)