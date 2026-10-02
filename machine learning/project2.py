import pandas as pd
import numpy as np
import gradio as gr
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

MODEL_PATH = "diabetes_model.pkl"
SCALER_PATH = "diabetes_scaler.pkl"
IMPUTER_PATH = "diabetes_imputer.pkl"

def prepare_model():
    if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, IMPUTER_PATH]):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        imputer = joblib.load(IMPUTER_PATH)
        return model, scaler, imputer
    
    df = pd.read_csv("diabetes.csv")
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Model Training Complete")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(imputer, IMPUTER_PATH)
    
    return model, scaler, imputer

model, scaler, imputer = prepare_model()

feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

def predict_diabetes(*inputs):
    input_df = pd.DataFrame([inputs], columns=feature_names)
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    input_df[zero_cols] = input_df[zero_cols].replace(0, np.nan)
    processed = imputer.transform(input_df)
    processed = scaler.transform(processed)
    pred = model.predict(processed)[0]
    prob = model.predict_proba(processed)[0][1]
    result = "Diabetic" if pred == 1 else "Not Diabetic"
    confidence = f"{prob*100:.1f}%"
    return f"Prediction: {result}\nConfidence: {confidence}"

inputs = [gr.Number(label=name) for name in feature_names]
outputs = gr.Textbox(label="Prediction Result")

app = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs=outputs,
    title="Diabetes Prediction System",
    description="Enter patient medical details to predict diabetes risk."
)

if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())