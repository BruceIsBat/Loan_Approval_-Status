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
├── models/                # Serialized trained models (e.g., XGBoost .pkl files)
├── notebooks/             # Jupyter notebooks for Exploratory Data Analysis (EDA)
├── src/                   # Source code for use in this project
│   ├── __init__.py
│   ├── preprocessing.py   # Data cleaning, feature engineering, and scaling
│   └── train.py           # Model training and evaluation logic
├── .gitignore             # Ignored files and virtual environments
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
Place the latest raw employee dataset into `data/raw/`. 

**2. Preprocessing:**
Run the preprocessing script to handle missing values, engineer financial features (like debt-to-income ratio), and scale the data.
```bash
python src/preprocessing.py
```

**3. Model Training:**
*(Whoever is here should insert the Instructions for running `src/train.py` to train the XGBoost model).*

## 👥 Authors
* **Esho Emmanuella** - *Team Lead*
* **Ibrahim Keji Awotedu** - *Initial Setup & ML Pipeline*
* **Temitope Iwashokun** - *Team member*

## 🤝 Contributing
1. Create a new feature branch (`git checkout -b feature/your-feature-name`)
2. Commit your changes (`git commit -m "Add some feature"`)
3. Push to the branch (`git push origin feature/your-feature-name`)

4. Open a Pull Request for review. Direct pushes to `main` are disabled to avoid losing work.
