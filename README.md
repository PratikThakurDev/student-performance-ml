# Student Performance Predictor

A machine learning project that predicts a student's exam score based on academic, lifestyle, and environmental factors.

The project includes exploratory data analysis, preprocessing, model comparison, model training, and an interactive prediction application built with Streamlit.

## Dataset

This project uses the **Student Performance Factors** dataset from Kaggle.

**Dataset source:** Lainguyn123 — Student Performance Factors

The dataset contains **6,607 student records**, **19 input features**, and one target variable: `Exam_Score`.

Example features include:

- Hours studied
- Attendance
- Previous scores
- Sleep hours
- Tutoring sessions
- Motivation level
- Access to resources
- Teacher quality
- Parental involvement
- Internet access

The dataset contains both numerical and categorical features, along with some missing categorical values.

The dataset file is not included in this repository. After downloading `StudentPerformanceFactors.csv`, place it at:

```text
data/StudentPerformanceFactors.csv
```

## Machine Learning Pipeline

The preprocessing and model are combined into a Scikit-learn pipeline.

### Numerical Features

Numerical features are standardized using `StandardScaler`.

### Categorical Features

Categorical features are processed using:

1. `SimpleImputer(strategy="most_frequent")` for missing values
2. `OneHotEncoder(handle_unknown="ignore", drop="first")`

### Model

The final prediction model is **Linear Regression**.

## Model Comparison

Four regression algorithms were compared using 5-fold cross-validation.

| Model | CV RMSE |
|---|---:|
| Linear Regression | 2.099 |
| Gradient Boosting | 2.251 |
| Random Forest | 2.527 |
| Decision Tree | 3.881 |

Linear Regression was selected for the final pipeline based on the cross-validation results.

## Final Model Performance

The final Linear Regression pipeline achieved:

| Metric | Score |
|---|---:|
| MAE | 0.452 |
| RMSE | 1.804 |
| R² | 0.770 |

## Project Structure

```text
student-performance-ml/
├── analysis/
│   ├── eda.py
│   └── model_comparison.py
├── data/
│   └── StudentPerformanceFactors.csv
├── models/
│   └── student_score_model.pkl
├── src/
│   ├── train.py
│   └── predict.py
├── app.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository and install the required dependencies:

```bash
git clone 
cd student-performance-ml

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local URL displayed by Streamlit in your browser.

## Train the Model

To retrain the model:

```bash
python3 src/train.py
```

The trained pipeline will be saved to:

```text
models/student_score_model.pkl
```

## Run the Analysis

Exploratory data analysis:

```bash
python3 analysis/eda.py
```

Model comparison:

```bash
python3 analysis/model_comparison.py
```