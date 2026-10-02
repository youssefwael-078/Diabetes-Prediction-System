import pandas as pd
import numpy as np
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ============================================================================
# PART 1: DATA PREPROCESSING
# ============================================================================

def load_and_prepare_data():
    """Load dataset and perform preprocessing"""
    df = pd.read_csv("diabetes.csv")
    
    # Handle zero values in critical columns
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    
    # Separate features and target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Imputation
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, imputer, scaler, X.columns.tolist(), df

# ============================================================================
# PART 2: MODEL TRAINING AND EVALUATION
# ============================================================================

def get_models():
    """Initialize all classification models"""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }
    return models

def train_all_models(X_train, y_train):
    """Train all models and return accuracies"""
    models = get_models()
    trained_models = {}
    accuracies = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        acc = accuracy_score(y_train, y_pred)
        trained_models[name] = model
        accuracies[name] = acc
    
    return trained_models, accuracies

# ============================================================================
# PART 3: SAMPLE DATA FOR TESTING
# ============================================================================

def get_diabetic_patient_data():
    """Return sample data for a diabetic patient (70-80% confidence)"""
    return {
        "Pregnancies": 6,
        "Glucose": 165,
        "BloodPressure": 80,
        "SkinThickness": 30,
        "Insulin": 180,
        "BMI": 33.5,
        "DiabetesPedigreeFunction": 0.7,
        "Age": 50
    }

def get_healthy_patient_data():
    """Return sample data for a healthy patient (70-80% confidence)"""
    return {
        "Pregnancies": 1,
        "Glucose": 95,
        "BloodPressure": 68,
        "SkinThickness": 18,
        "Insulin": 55,
        "BMI": 23.5,
        "DiabetesPedigreeFunction": 0.3,
        "Age": 28
    }

# ============================================================================
# PART 4: PREDICTION FUNCTION
# ============================================================================

def predict_diabetes(model_name, pregnancies, glucose, blood_pressure, skin_thickness, 
                     insulin, bmi, dpf, age, X_train, y_train, imputer, scaler, feature_names):
    """Make prediction using selected model"""
    
    # Train models
    trained_models, accuracies = train_all_models(X_train, y_train)
    model = trained_models[model_name]
    
    # Prepare input data
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, 
                  insulin, bmi, dpf, age]
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Handle zeros
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    input_df[zero_cols] = input_df[zero_cols].replace(0, np.nan)
    
    # Preprocess
    processed = imputer.transform(input_df)
    processed = scaler.transform(processed)
    
    # Predict
    pred = model.predict(processed)[0]
    prob = model.predict_proba(processed)[0]
    
    result = "Diabetic (Positive) ✓" if pred == 1 else "Not Diabetic (Negative) ✓"
    confidence = prob[pred] * 100
    
    return result, confidence, prob, accuracies

# ============================================================================
# PART 5: GUI INTERFACE
# ============================================================================

def create_interface():
    """Create Gradio interface"""
    
    # Load data
    X_train, X_test, y_train, y_test, imputer, scaler, feature_names, df = load_and_prepare_data()
    _, accuracies = train_all_models(X_train, y_train)
    
    best_model = max(accuracies, key=accuracies.get)
    
    def make_prediction(model_name, pregnancies, glucose, blood_pressure, 
                       skin_thickness, insulin, bmi, dpf, age):
        result, confidence, prob, acc = predict_diabetes(
            model_name, pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age, X_train, y_train, imputer, scaler, feature_names
        )
        
        # Create report
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              DIABETES PREDICTION SYSTEM                      ║
║         Machine Learning Classification Project              ║
╚══════════════════════════════════════════════════════════════╝

📊 DATASET INFORMATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total Samples: {df.shape[0]}
• Total Features: {df.shape[1] - 1}
• Target: Outcome (0 = Healthy, 1 = Diabetic)

🤖 MODEL PERFORMANCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        for name, acc in sorted(acc.items(), key=lambda x: x[1], reverse=True):
            marker = "★" if name == model_name else "○"
            report += f"{marker} {name}: {acc:.4f}\n"
        
        report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👤 PATIENT DATA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Pregnancies: {pregnancies}
• Glucose: {glucose} mg/dL
• Blood Pressure: {blood_pressure} mmHg
• Skin Thickness: {skin_thickness} mm
• Insulin: {insulin} mu U/ml
• BMI: {bmi}
• Diabetes Pedigree: {dpf}
• Age: {age} years

🎯 PREDICTION RESULT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Model: {model_name}
• Result: {result}
• Confidence: {confidence:.2f}%
• Probability (Healthy): {prob[0]*100:.2f}%
• Probability (Diabetic): {prob[1]*100:.2f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        return report
    
    def fill_diabetic():
        data = get_diabetic_patient_data()
        return [data["Pregnancies"], data["Glucose"], data["BloodPressure"], 
                data["SkinThickness"], data["Insulin"], data["BMI"], 
                data["DiabetesPedigreeFunction"], data["Age"]]
    
    def fill_healthy():
        data = get_healthy_patient_data()
        return [data["Pregnancies"], data["Glucose"], data["BloodPressure"], 
                data["SkinThickness"], data["Insulin"], data["BMI"], 
                data["DiabetesPedigreeFunction"], data["Age"]]
    
    # Build UI
    with gr.Blocks(theme=gr.themes.Soft(), title="Diabetes Prediction") as app:
        gr.Markdown("# 🏥 Diabetes Prediction System")
        gr.Markdown("### 📝 Project Description:\nThis is a machine learning classification project that predicts diabetes using multiple algorithms including Logistic Regression, Decision Tree, Random Forest, SVM, and Neural Network.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Dataset Info")
                gr.Textbox(label="Samples", value=f"{df.shape[0]}", interactive=False)
                gr.Textbox(label="Features", value=f"{df.shape[1]-1}", interactive=False)
                gr.Textbox(label="Best Model", value=f"{best_model} ({accuracies[best_model]:.4f})", interactive=False)
                
                gr.Markdown("### 📋 Model Accuracies")
                acc_text = "\n".join([f"{name}: {acc:.4f}" for name, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True)])
                gr.Textbox(label="All Models", value=acc_text, lines=6, interactive=False)
            
            with gr.Column(scale=2):
                gr.Markdown("### ⚙️ Model Selection")
                model_dropdown = gr.Dropdown(
                    choices=["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "Neural Network"],
                    label="Select Classification Model",
                    value=best_model
                )
                
                gr.Markdown("### 👤 Patient Details")
                
                with gr.Row():
                    pregnancies = gr.Number(label="Pregnancies", value=0)
                    glucose = gr.Number(label="Glucose", value=100)
                
                with gr.Row():
                    blood_pressure = gr.Number(label="Blood Pressure", value=70)
                    skin_thickness = gr.Number(label="Skin Thickness", value=20)
                
                with gr.Row():
                    insulin = gr.Number(label="Insulin", value=80)
                    bmi = gr.Number(label="BMI", value=25.0)
                
                with gr.Row():
                    dpf = gr.Number(label="Diabetes Pedigree", value=0.5)
                    age = gr.Number(label="Age", value=30)
                
                gr.Markdown("### 🔘 Quick Fill")
                with gr.Row():
                    btn_diabetic = gr.Button("🏥 Diabetic Patient", variant="secondary")
                    btn_healthy = gr.Button("✅ Healthy Patient", variant="secondary")
                
                predict_btn = gr.Button("🔮 PREDICT", variant="primary", size="lg")
                
                output = gr.Textbox(label="📈 Results", lines=35)
                
                # Event handlers
                btn_diabetic.click(fn=fill_diabetic, outputs=[pregnancies, glucose, blood_pressure, 
                                                              skin_thickness, insulin, bmi, dpf, age])
                btn_healthy.click(fn=fill_healthy, outputs=[pregnancies, glucose, blood_pressure, 
                                                            skin_thickness, insulin, bmi, dpf, age])
                predict_btn.click(
                    fn=make_prediction,
                    inputs=[model_dropdown, pregnancies, glucose, blood_pressure, 
                           skin_thickness, insulin, bmi, dpf, age],
                    outputs=output
                )
    
    return app

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    app = create_interface()
    app.launch()