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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def load_and_prepare_data():
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
    return X_train_scaled, X_test_scaled, y_train, y_test, imputer, scaler, X.columns.tolist(), df

def get_diabetic_patient_data():
    return [6, 165, 80, 30, 180, 33.5, 0.7, 50]

def get_healthy_patient_data():
    return [1, 95, 68, 18, 55, 23.5, 0.3, 28]

def train_model(model_name, optimizer, loss_fn, X_train, y_train):
    if model_name == "Neural Network":
        try:
        
            model = MLPClassifier(
                hidden_layer_sizes=(64, 32), 
                solver=optimizer, 
                loss=loss_fn, 
                max_iter=500, 
                random_state=42
            )
        except TypeError:

            model = MLPClassifier(
                hidden_layer_sizes=(64, 32), 
                solver=optimizer,
                max_iter=500, 
                random_state=42
            )
    elif model_name == "Logistic Regression":
        model = LogisticRegression(solver=optimizer, max_iter=1000, random_state=42)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "SVM":
        model = SVC(probability=True, kernel='rbf', random_state=42)
    else:
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    
    model.fit(X_train, y_train)
    return model

def predict_and_evaluate(model, X_test, y_test, imputer, scaler, feature_names, input_values):
    input_df = pd.DataFrame([input_values], columns=feature_names)
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    input_df[zero_cols] = input_df[zero_cols].replace(0, np.nan)
    processed = imputer.transform(input_df)
    processed = scaler.transform(processed)
    y_pred = model.predict(X_test)
    y_pred_sample = model.predict(processed)[0]
    prob = model.predict_proba(processed)[0]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    result_label = "Diabetic (Positive)" if y_pred_sample == 1 else "Not Diabetic (Negative)"
    confidence = prob[y_pred_sample] * 100
    return {
        "prediction": result_label,
        "confidence": confidence,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "probabilities": prob
    }

def format_classification_report(report_dict):
    output = "\n" + "="*70 + "\n"
    output += "                    CLASSIFICATION REPORT\n"
    output += "="*70 + "\n\n"
    output += "📊 CLASS 0 - NOT DIABETIC (Healthy Patients):\n"
    output += "-"*70 + "\n"
    output += f"   Precision:  {report_dict['0']['precision']:.4f}\n"
    output += f"   Recall:     {report_dict['0']['recall']:.4f}\n"
    output += f"   F1-Score:   {report_dict['0']['f1-score']:.4f}\n"
    output += f"   Support:    {int(report_dict['0']['support'])} patients\n\n"
    output += "📊 CLASS 1 - DIABETIC (Sick Patients):\n"
    output += "-"*70 + "\n"
    output += f"   Precision:  {report_dict['1']['precision']:.4f}\n"
    output += f"   Recall:     {report_dict['1']['recall']:.4f}\n"
    output += f"   F1-Score:   {report_dict['1']['f1-score']:.4f}\n"
    output += f"   Support:    {int(report_dict['1']['support'])} patients\n\n"
    output += "="*70 + "\n"
    output += "📈 OVERALL METRICS:\n"
    output += "="*70 + "\n"
    output += f"   Accuracy:      {report_dict['accuracy']:.4f}\n"
    output += f"   Macro Avg:     {report_dict['macro avg']['f1-score']:.4f}\n"
    output += f"   Weighted Avg:  {report_dict['weighted avg']['f1-score']:.4f}\n"
    output += f"   Total Samples: {int(report_dict['0']['support']) + int(report_dict['1']['support'])}\n"
    output += "="*70 + "\n"
    return output

model_configs = {
    "Neural Network": {
        "optimizers": ["adam", "sgd", "lbfgs"],
        "losses": ["log_loss", "hinge"],
        "show_optimizer": True,
        "show_loss": True,
        "info": "Neural networks use gradient-based optimizers"
    },
    "Logistic Regression": {
        "optimizers": ["lbfgs", "liblinear", "newton-cg", "saga", "newton-cholesky"],
        "losses": ["log_loss (binary cross-entropy)"],
        "show_optimizer": True,
        "show_loss": False,
        "info": "Uses solvers for optimization, loss is fixed"
    },
    "Decision Tree": {
        "optimizers": ["Not Applicable (uses Gini/Entropy)"],
        "losses": ["Not Applicable"],
        "show_optimizer": False,
        "show_loss": False,
        "info": "Tree-based methods don't use traditional optimizers"
    },
    "Random Forest": {
        "optimizers": ["Not Applicable (ensemble method)"],
        "losses": ["Not Applicable"],
        "show_optimizer": False,
        "show_loss": False,
        "info": "Ensemble of trees, no gradient optimization"
    },
    "SVM": {
        "optimizers": ["Not Applicable (uses SMO algorithm)"],
        "losses": ["Not Applicable (uses hinge loss internally)"],
        "show_optimizer": False,
        "show_loss": False,
        "info": "SVM uses Sequential Minimal Optimization"
    }
}

def update_model_parameters(model_name):
    config = model_configs.get(model_name, model_configs["Neural Network"])
    
    optimizer_dropdown = gr.Dropdown(
        choices=config["optimizers"],
        label="Select Optimizer/Solver",
        value=config["optimizers"][0],
        interactive=config["show_optimizer"],
        visible=True
    )
    
    loss_dropdown = gr.Dropdown(
        choices=config["losses"],
        label="Select Loss Function",
        value=config["losses"][0],
        interactive=config["show_loss"],
        visible=config["show_loss"]
    )
    
    info_text = f"ℹ️ {config['info']}"
    
    return optimizer_dropdown, loss_dropdown, info_text

def create_interface():
    X_train, X_test, y_train, y_test, imputer, scaler, feature_names, df = load_and_prepare_data()
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Diabetes Prediction System") as app:
        gr.Markdown("# 🏥 Diabetes Prediction System")
        gr.Markdown("### 📝 Project Description:\nMachine learning classification project that predicts diabetes using multiple algorithms with different optimizers and loss functions. The system implements Logistic Regression, Decision Tree, Random Forest, SVM, and Neural Network classifiers.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Dataset Information")
                gr.Textbox(label="Total Samples", value=str(df.shape[0]), interactive=False)
                gr.Textbox(label="Total Features", value=str(df.shape[1]-1), interactive=False)
                gr.Textbox(label="Total Columns", value=str(df.shape[1]), interactive=False)
                gr.Textbox(label="Target Classes", value="0 (Healthy), 1 (Diabetic)", interactive=False)
                
                gr.Markdown("### 📋 First 6 Rows of Dataset")
                gr.Dataframe(df.head(6), label="Dataset Sample")
            
            with gr.Column(scale=2):
                gr.Markdown("### ⚙️ Model Configuration")
                
                model_dropdown = gr.Dropdown(
                    choices=["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "Neural Network"],
                    label="Select Model",
                    value="Neural Network"
                )
                
                model_info = gr.Textbox(
                    label="Model Information",
                    value="ℹ️ Neural networks use gradient-based optimizers",
                    interactive=False,
                    lines=1
                )
                
                with gr.Row():
                    optimizer_dropdown = gr.Dropdown(
                        choices=["adam", "sgd", "lbfgs"],
                        label="Select Optimizer/Solver",
                        value="adam"
                    )
                    loss_dropdown = gr.Dropdown(
                        choices=["log_loss", "hinge"],
                        label="Select Loss Function",
                        value="log_loss"
                    )
                
                train_btn = gr.Button("🔧 TRAIN / FIT MODEL", variant="primary", size="lg")
                train_status = gr.Textbox(label="Training Status", interactive=False)
                
                gr.Markdown("### 👤 Patient Medical Details")
                
                with gr.Row():
                    pregnancies = gr.Number(label="Pregnancies", value=0)
                    glucose = gr.Number(label="Glucose (mg/dL)", value=100)
                
                with gr.Row():
                    blood_pressure = gr.Number(label="Blood Pressure (mmHg)", value=70)
                    skin_thickness = gr.Number(label="Skin Thickness (mm)", value=20)
                
                with gr.Row():
                    insulin = gr.Number(label="Insulin (mu U/ml)", value=80)
                    bmi = gr.Number(label="BMI", value=25.0)
                
                with gr.Row():
                    dpf = gr.Number(label="Diabetes Pedigree Function", value=0.5)
                    age = gr.Number(label="Age (years)", value=30)
                
                gr.Markdown("### 🔘 Quick Fill Patient Data")
                with gr.Row():
                    btn_diabetic = gr.Button("🏥 Diabetic Patient", variant="secondary")
                    btn_healthy = gr.Button("✅ Healthy Patient", variant="secondary")
                
                predict_btn = gr.Button("🔮 PREDICT", variant="primary", size="lg")
                
                output = gr.Textbox(label="📈 Prediction Results & Model Performance", lines=40)
        
        model_state = gr.State()
        
        model_dropdown.change(
            fn=update_model_parameters,
            inputs=[model_dropdown],
            outputs=[optimizer_dropdown, loss_dropdown, model_info]
        )
        
        def train_action(model_name, optimizer, loss_fn):
            try:
                model = train_model(model_name, optimizer, loss_fn, X_train, y_train)
                y_pred_train = model.predict(X_train)
                train_acc = accuracy_score(y_train, y_pred_train)
                
                config = model_configs[model_name]
                if config["show_optimizer"] and config["show_loss"]:
                    status = f"✅ Model Trained Successfully!\n\nModel: {model_name}\nOptimizer: {optimizer}\nLoss Function: {loss_fn}\nTraining Accuracy: {train_acc:.4f}"
                elif config["show_optimizer"]:
                    status = f"✅ Model Trained Successfully!\n\nModel: {model_name}\nSolver: {optimizer}\nLoss: {config['losses'][0]}\nTraining Accuracy: {train_acc:.4f}"
                else:
                    status = f"✅ Model Trained Successfully!\n\nModel: {model_name}\n{config['info']}\nTraining Accuracy: {train_acc:.4f}"
                
                return model, status
            except Exception as e:
                return None, f"❌ Training Failed: {str(e)}"
        
        def predict_action(model, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
            if model is None:
                return "❌ Please train the model first! Click the TRAIN button before predicting."
            
            input_values = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
            metrics = predict_and_evaluate(model, X_test, y_test, imputer, scaler, feature_names, input_values)
            
            report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    PREDICTION RESULTS                        ║
╚══════════════════════════════════════════════════════════════╝

👤 PATIENT INPUT DATA:
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
• Result: {metrics['prediction']}
• Confidence: {metrics['confidence']:.2f}%
• Probability (Not Diabetic): {metrics['probabilities'][0]*100:.2f}%
• Probability (Diabetic): {metrics['probabilities'][1]*100:.2f}%

📊 MODEL PERFORMANCE METRICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Accuracy:  {metrics['accuracy']:.4f}
• Precision: {metrics['precision']:.4f}
• Recall:    {metrics['recall']:.4f}
• F1-Score:  {metrics['f1']:.4f}

📄 CONFUSION MATRIX:
{metrics['confusion_matrix']}

{format_classification_report(metrics['classification_report'])}
            """
            return report
        
        btn_diabetic.click(
            fn=lambda: get_diabetic_patient_data(),
            outputs=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        )
        
        btn_healthy.click(
            fn=lambda: get_healthy_patient_data(),
            outputs=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        )
        
        train_btn.click(
            fn=train_action,
            inputs=[model_dropdown, optimizer_dropdown, loss_dropdown],
            outputs=[model_state, train_status]
        )
        
        predict_btn.click(
            fn=predict_action,
            inputs=[model_state, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age],
            outputs=output
        )
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch()