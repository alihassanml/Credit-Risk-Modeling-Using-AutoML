# Credit Risk Modeling Using AutoML

This project demonstrates the use of **AutoML** and **Explainable AI (XAI)** to predict **credit default risk** for a large-scale dataset. The goal is to build an interpretable machine learning model using **AutoGluon**, **XGBoost**, and **LightGBM** with **SHAP** for model interpretability.

### 📊 Dataset

The dataset used in this project is the **UCI Credit Card Default Dataset** available at [UCI Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).

* **Number of records**: 30,000
* **Number of features**: 23 features related to credit, demographics, and payment history.
* **Target**: Whether the client will default on their payment next month (`1` = Default, `0` = No Default).

### 🔧 Technologies

* **LightGBM**: Gradient Boosting decision trees used for model training.
* **XGBoost**: Another powerful gradient boosting model.
* **AutoGluon**: AutoML library for easy model training and hyperparameter tuning.
* **SHAP**: SHapley Additive exPlanations for model interpretability.
* **Apache Spark**: Distributed processing for handling large-scale datasets.

### 🛠️ Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/alihassanml/Credit-Risk-Modeling-Using-AutoML.git
   cd Credit-Risk-Modeling-Using-AutoML
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

### 🚀 Running the Project

1. **Prepare Data**:
   The dataset is already available. If you want to download it manually, visit [UCI Credit Card Default Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).

2. **Preprocessing and Feature Engineering**:
   Run the following script to clean and process the data:

   ```bash
   python preprocess.py
   ```

3. **Train Models**:
   The model training pipeline is automated using **AutoGluon** and **LightGBM**. To train the models, use:

   ```bash
   python train_model.py
   ```

4. **Model Explainability**:
   After training, SHAP will be used to generate interpretability plots:

   ```bash
   python explain_model.py
   ```

5. **Evaluate the Model**:
   You can evaluate the trained model by running:

   ```bash
   python evaluate_model.py
   ```

### 📁 Directory Structure

```
.
├── data/                      # Store raw and processed data here
├── src/                       # Source code for preprocessing, training, etc.
│   ├── preprocess.py          # Data preprocessing and feature engineering
│   ├── train_model.py         # AutoML model training (AutoGluon, LightGBM, XGBoost)
│   ├── evaluate_model.py      # Evaluate the trained model
│   ├── explain_model.py       # Model explainability using SHAP
│   └── feature_engineering.py # Custom feature engineering (AutoGluon setup)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── outputs/                   # Model outputs and visualizations
```

### 🔑 Key Features

* **AutoML Model Selection**: Uses **AutoGluon** to automatically select the best model and hyperparameters.
* **Interpretable Results**: Leverages **SHAP** for explainability, visualizing high-risk patterns for default prediction.
* **Scalable**: Utilizes **Apache Spark** for large dataset processing and distributed computing.

### 🧑‍🏫 Example Workflow

1. **Preprocess Data**
   Clean data and generate derived features (e.g., `AVG_PAY_DELAY`, `MAX_PAY_DELAY`, `PAYMENT_RATIO`).

2. **Train Model**
   Use **AutoGluon** to automatically select the best models (e.g., **LightGBM**, **XGBoost**), fine-tune hyperparameters, and perform cross-validation.

3. **Evaluate Model**
   Evaluate the models using metrics such as **Accuracy**, **AUC**, **Precision**, and **Recall**.

4. **Model Explainability**
   Use **SHAP** to interpret model predictions, highlighting which features are most indicative of credit default risk.

### 🧑‍💻 Example Code Snippets

* **Feature Engineering (`feature_engineering.py`)**:

  * Includes custom feature generation like `AVG_PAY_DELAY`, `BILL_TOTAL`, and `PAYMENT_RATIO`.

* **Model Training (`train_model.py`)**:

  ```python
  from autogluon.tabular import TabularPredictor

  # Load preprocessed data
  df = pd.read_csv("processed_data.csv")

  # Train AutoML model
  predictor = TabularPredictor(label="default").fit(df)
  ```

* **Model Explainability with SHAP (`explain_model.py`)**:

  ```python
  import shap
  model = predictor.load("model.pkl")

  # Create SHAP explainer
  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(df)

  # Plot SHAP summary
  shap.summary_plot(shap_values, df)
  ```

### 📊 Evaluation Metrics

* **Accuracy**: Measures overall prediction accuracy.
* **AUC (Area Under the ROC Curve)**: Evaluates classification performance at various thresholds.
* **Precision/Recall/F1 Score**: Measures model performance, especially for imbalanced datasets.

### ⚠️ Limitations

* The dataset may not fully represent all types of credit risk situations.
* The models are highly sensitive to hyperparameter tuning and require careful optimization for optimal results.

### 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
