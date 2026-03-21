import gradio as gr
import joblib
import pandas as pd
import numpy as np
import gdown
import os

# --------------------------------------------------------------------------------------------
# Check if model exists locally. And if not, download model from Google Drive using the file ID.
# --------------------------------------------------------------------------------------------
model_file_id = "1-m6VbYUJCO82hSaJQlKA7rm-YUjnXJQf"
model_path = "loan_model.joblib"

if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={model_file_id}"
    gdown.download(url, model_path, quiet=False)
    print("Download complete.")
else:
    print("Model already exists.")

# ------------------------------
# Load artifacts (adjust paths if you uploaded scaler/features separately)
# ------------------------------
model = joblib.load("loan_model.joblib")

# For scaler and feature_columns, we have two options:
# Option A: Also upload them to Drive and download similarly (recommended for large files)
# Option B: Upload them directly to Hugging Face (if they are small)


# Scaler.joblib and feature_columns.joblib in the same folder (uploaded manually):
scaler = joblib.load("scaler.joblib")  
feature_columns = joblib.load("feature_columns.joblib")  

# ------------------------------
# Prediction function
# ------------------------------
def predict_loan(applicant_type,principal, interest_rate, tenor, salary, dti, credit_score,
                salary_detected, employment_type, loan_purpose, state, threshold):
    
    if applicant_type == "New":
        credit_score_used = 500
    else:
        credit_score_used = credit_score
        
        # Create input dictionary
    input_data = {
        'principal_ngn': principal,
        'interest_rate_annual': interest_rate,
        'tenor_months': tenor,
        'salary_ngn': salary,
        'debt_to_income_ratio': dti,
        'credit_score': credit_score_used,
        'salary_detected': 1 if salary_detected == "Yes" else 0,
        'employment_type': employment_type,
        'loan_purpose': loan_purpose,
        'state': state
    }
    input_df = pd.DataFrame([input_data])

    # One-hot encode exactly as in training
    input_encoded = pd.get_dummies(input_df, columns=['employment_type', 'loan_purpose', 'state'], drop_first=False)
    
    # Align columns with training data
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    
    # Scale features
    input_scaled = scaler.transform(input_encoded)
    
    # Predict probability of default (class 1)
    prob_default = model.predict_proba(input_scaled)[0, 1]
    
     
    # Decision based on threshold
    if prob_default < threshold:
        return f"✅ APPROVED (Default Probability: {prob_default:.2%})"
    else:
        return f"❌ REJECTED (Default Probability: {prob_default:.2%})"

# ------------------------------
# Gradio Interface
# ------------------------------
inputs = [
    gr.Radio(["New", "Returning"], label="Applicant Type", value="New"),
    gr.Number(label="Principal (NGN)", value=()),
    gr.Number(label="Annual Interest Rate (%)", value=()),
    gr.Number(label="Tenor (months)", value=()),
    gr.Number(label="Monthly Salary (NGN)", value=()),
    gr.Number(label="Debt-to-Income Ratio", value=()),
    gr.Number(label="Credit Score(for returning applicants)", value=()),
    gr.Radio(["Yes", "No"], label="Salary Detected?"),
    gr.Dropdown(["private_sector", "self_employed", "unemployed"], label="Employment Type"),
    gr.Dropdown(["debt_consolidation", "education", "home_improvement", "medical", "personal"], label="Loan Purpose"),
    gr.Dropdown(["Anambra", "Delta", "Imo", "Kaduna", "Kano", "Lagos", "Ogun", "Other", "Oyo", "Rivers"], label="State"),
    gr.Slider(0.0, 1.0, value=0.5, label="Approval Threshold")
]

outputs = gr.Textbox(label="Result")

gr.Interface(
    fn=predict_loan,
    inputs=inputs,
    outputs=outputs,
    title="🏦 Loan Approval Predictor (Nigeria)",
    description="Enter applicant details to predict default risk."
).launch(share=True)