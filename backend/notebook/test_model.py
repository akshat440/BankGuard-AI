import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

def main():
    # 1️⃣ Load saved model
    model = joblib.load("random_forest_apk.pkl")
    print("✅ Model loaded successfully!")

    # 2️⃣ Load test dataset
    X_test = pd.read_csv("X_test.csv").values
    y_test = pd.read_csv("y_test.csv")["label"].values
    print(f"✅ Test data loaded: {X_test.shape}, Labels: {y_test.shape}")

    # 3️⃣ Make predictions
    y_pred = model.predict(X_test)

    # 4️⃣ Evaluate performance
    print("\n--- Model Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 5️⃣ Show sample predictions
    print("\n--- Sample Predictions ---")
    print("Predicted:", y_pred[:10])
    print("Actual:   ", y_test[:10])

if __name__ == "__main__":
    main()
