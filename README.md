# Internal Employee Loan Approval Model

## 📌 Project Overview
This repository contains a machine learning pipeline designed to automate and assist in the approval process for internal employee loans. By leveraging historical employee data, salary metrics, and performance indicators, this model aims to provide fair, consistent, and accurate loan decisions.

Special attention is given to handling imbalanced datasets and preventing unintentional biases regarding employee roles or departments.


## 🗂️ Project Structure
This project follows the Cookiecutter Data Science standard for reproducibility:

```text
├── data/
│   ├── raw/               # Immutable original employee loan data
│   └── processed/         # Cleaned, scaled, and encoded data
├── models/                # Serialized trained models (e.g., rf_loan_approval_v1.joblib)
├── notebooks/             # Jupyter notebooks for Exploratory Data Analysis (EDA)
├── scripts/               # Utility scripts for data preparation, validation, etc.
│   ├── prepare_dataset.py
│   ├── validate_data.py
│   ├── evaluate_model.py
│   └── deploy_model.py
├── src/                   # Source code for use in this project
│   ├── __init__.py
│   ├── config.py          # Centralized configuration & environment variables
│   ├── utils.py           # Shared utility functions
│   ├── preprocessing.py   # Data cleaning, feature engineering, and scaling
│   ├── train.py           # Model training and evaluation logic
│   ├── evaluate.py        # Evaluation metrics computation
│   ├── models.py          # Custom model classes
│   ├── features.py        # Feature engineering functions
│   ├── data_validation.py # Pydantic schemas for validation
│   ├── fastapi_app.py     # FastAPI inference server
│   └── app.py             # Gradio demo app
├── tests/                 # Unit tests
├── .env.example           # Example environment configuration
├── .gitignore             # Ignored files and virtual environments
├── pyproject.toml         # Project packaging and dependencies
├── requirements.txt       # Frozen environment dependencies
└── README.md              # Project documentation
```

## 🛠️ Tech Stack
* **Language:** Python 3
* **Package Manager:** `uv` (for blazing-fast dependency resolution)
* **Data Manipulation:** NumPy, Pandas
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-Learn
* **Visualization:** Matplotlib, Seaborn

## 🚀 Getting Started

### 1. Clone the Repository
Open your WSL/Ubuntu terminal and run:
```bash
git clone https://github.com/BruceIsBat/Loan_Approval_-Status.git
cd Loan_Approval_-Status
```

### 2. Set Up the Virtual Environment
This project uses a localized virtual environment named `employee_env` to isolate dependencies.
```bash
# Create the environment in Linux/WSL
python3 -m venv employee_env

# Activate the environment
source employee_env/bin/activate
```

### 3. Install Dependencies
We use `uv` for ultra-fast package installation. If you don't have `uv` installed, you can use standard `pip`.
```bash
# Install dependencies from the requirements file
uv pip install -r requirements.txt
```

## 🧠 Usage

**1. Data Preparation:**
Place the latest raw employee dataset into `data/raw/loan_data_raw.csv`.

**2. Preprocessing:**
Run the preprocessing script to handle missing values, engineer financial features (like debt-to-income ratio), and scale the data.
```bash
python src/preprocessing.py --input data/raw/loan_data_raw.csv --output data/processed/loan_data_cleaned.csv
```

**3. Model Training:**
Train the model using the processed data.
```bash
python src/train.py --data data/processed/loan_data_cleaned.csv --model-output models/rf_loan_approval_v1.joblib
```

**4. Dataset Preparation (Alternative):**
Use the utility script for consistent dataset naming.
```bash
python scripts/prepare_dataset.py --raw-input data/raw/loan_data_raw.csv --processed-output data/processed/loan_data_cleaned.csv
```

**5. Validate Data:**
Check data quality before processing.
```bash
python scripts/validate_data.py --data data/raw/loan_data_raw.csv
```

**6. Evaluate Model:**
After training, evaluate performance.
```bash
python scripts/evaluate_model.py --model models/rf_loan_approval_v1.joblib --test-data data/processed/loan_data_test.csv
```

**7. Deploy Model:**
Prepare model for production serving.
```bash
python scripts/deploy_model.py --model models/rf_loan_approval_v1.joblib --output models/deployed_model.joblib
```

## 🚀 Production API

### Start FastAPI Server
```bash
# Install FastAPI and Uvicorn
pip install fastapi uvicorn

# Run inference server
python -m uvicorn src.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Model Info:**
```bash
curl http://localhost:8000/model/info
```

**Single Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "loan_amount": 100000,
      "applicant_income": 50000
    }
  }'
```

**Batch Predictions:**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"features": {"loan_amount": 100000, "applicant_income": 50000}},
    {"features": {"loan_amount": 200000, "applicant_income": 75000}}
  ]'
```

### Interactive API Documentation
Visit `http://localhost:8000/docs` for Swagger UI
Visit `http://localhost:8000/redoc` for ReDoc UI

## ⚙️ Configuration

### Environment Variables
Copy `.env.example` to `.env` and modify settings:

```bash
cp .env.example .env
```

Supported variables:
- `ENVIRONMENT`: development or production
- `RANDOM_STATE`: Random seed
- `TEST_SIZE`: Train/test split ratio
- `N_ESTIMATORS`: Random Forest tree count
- `API_HOST`: API server host (default: 0.0.0.0)
- `API_PORT`: API server port (default: 8000)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
* **Esho Emmanuella** - *Team Lead*
* ** [@Awotedu Ibrahim](https://github.com/BruceIsBat)
* ** [@Ozono Ofeoghena Blessing](https://github.com/Username)
* ** [@Ajiborisade Ekundayo](https://github.com/ajiborishade) 
* ** [@Temitope Iwashokun](https://github.com/TemiDavid)
* ** [@Akinlawon Ladimeji](https://github.com/LadiLawon) #copy and add your name
## 🤝 Contributing
1. Create a new feature branch (`git checkout -b feature/your-feature-name`)
2. Commit your changes (`git commit -m "Add some feature"`)
3. Push to the branch (`git push origin feature/your-feature-name`)

4. Open a Pull Request for review. Direct pushes to `main` are disabled to avoid losing work.





