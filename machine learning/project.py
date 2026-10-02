import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

class DiabetesPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Diabetes Prediction System")
        self.root.geometry("850x750")
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.imputer = None
        self.load_and_preprocess()
        self.build_interface()

    def load_and_preprocess(self):
        df = pd.read_csv("diabetes.csv")
        zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        df[zero_columns] = df[zero_columns].replace(0, np.nan)
        X_raw = df.drop("Outcome", axis=1)
        y = df["Outcome"]
        self.imputer = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X_raw), columns=X_raw.columns)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def build_interface(self):
        container = tk.Frame(self.root, padx=30, pady=20)
        container.pack(fill=tk.BOTH, expand=True)
        tk.Label(container, text="Model Selection:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        self.model_var = tk.StringVar()
        ttk.Combobox(container, textvariable=self.model_var, values=["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "Neural Network"], state="readonly", width=40).pack(pady=5)
        tk.Label(container, text="Optimizer Selection:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(10, 0))
        self.optimizer_var = tk.StringVar(value="adam")
        ttk.Combobox(container, textvariable=self.optimizer_var, values=["adam", "sgd", "lbfgs"], state="readonly", width=40).pack(pady=5)
        tk.Label(container, text="Loss Function Selection:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(10, 0))
        self.loss_var = tk.StringVar(value="log_loss")
        ttk.Combobox(container, textvariable=self.loss_var, values=["log_loss", "hinge", "squared_error"], state="readonly", width=40).pack(pady=5)
        self.train_button = tk.Button(container, text="Train / Fit Model", font=("Arial", 11), bg="#4CAF50", fg="white", command=self.execute_training)
        self.train_button.pack(fill=tk.X, pady=15)
        self.predict_button = tk.Button(container, text="Predict on Test Data", font=("Arial", 11), bg="#2196F3", fg="white", command=self.execute_prediction)
        self.predict_button.pack(fill=tk.X, pady=5)
        self.output_area = tk.Text(container, height=20, font=("Consolas", 10), state="disabled")
        self.output_area.pack(fill=tk.BOTH, expand=True, pady=15)

    def append_text(self, message):
        self.output_area.config(state="normal")
        self.output_area.insert(tk.END, message + "\n")
        self.output_area.see(tk.END)
        self.output_area.config(state="disabled")

    def execute_training(self):
        selected_model = self.model_var.get()
        selected_optimizer = self.optimizer_var.get()
        selected_loss = self.loss_var.get()
        try:
            if selected_model == "Logistic Regression":
                self.model = LogisticRegression(max_iter=1000, random_state=42)
            elif selected_model == "Decision Tree":
                self.model = DecisionTreeClassifier(random_state=42)
            elif selected_model == "Random Forest":
                self.model = RandomForestClassifier(random_state=42)
            elif selected_model == "SVM":
                self.model = SVC(random_state=42)
            elif selected_model == "Neural Network":
                self.model = MLPClassifier(hidden_layer_sizes=(64, 32), solver=selected_optimizer, loss=selected_loss, max_iter=500, random_state=42)
            else:
                raise ValueError("Invalid model selection")
            self.model.fit(self.X_train, self.y_train)
            joblib.dump(self.model, "diabetes_model.pkl")
            joblib.dump(self.scaler, "diabetes_scaler.pkl")
            joblib.dump(self.imputer, "diabetes_imputer.pkl")
            self.append_text(f"Training started for {selected_model}")
            self.append_text(f"Optimizer: {selected_optimizer} | Loss: {selected_loss}")
            self.append_text("Model trained and saved successfully.")
            self.append_text("Ready for prediction.")
        except Exception as error:
            self.append_text(f"Training failed: {str(error)}")

    def execute_prediction(self):
        if self.model is None:
            self.append_text("Error: Please train the model before predicting.")
            return
        try:
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            report = classification_report(self.y_test, y_pred)
            cm = confusion_matrix(self.y_test, y_pred)
            self.append_text("="*40)
            self.append_text("PREDICTION AND EVALUATION RESULTS")
            self.append_text("="*40)
            self.append_text(f"Accuracy:  {accuracy:.4f}")
            self.append_text(f"Precision: {precision:.4f}")
            self.append_text(f"Recall:    {recall:.4f}")
            self.append_text(f"F1-Score:  {f1:.4f}")
            self.append_text("-"*40)
            self.append_text("CLASSIFICATION REPORT:")
            self.append_text(report)
            self.append_text("-"*40)
            self.append_text("CONFUSION MATRIX:")
            self.append_text(str(cm))
            self.append_text("-"*40)
            self.append_text(f"Sample Predictions: {y_pred[:5].tolist()}")
            self.append_text(f"Sample Actuals:     {self.y_test[:5].tolist()}")
            self.append_text("="*40)
        except Exception as error:
            self.append_text(f"Prediction failed: {str(error)}")

if __name__ == "__main__":
    root_window = tk.Tk()
    application = DiabetesPredictionApp(root_window)
    root_window.mainloop()