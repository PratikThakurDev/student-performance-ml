# Student Performance Predictor

A machine learning project that predicts a student's exam score based on academic, lifestyle, and environmental factors.

The project includes exploratory data analysis, preprocessing, model comparison, model training, and prediction on new student data.

## Dataset

The dataset contains 6,607 student records with 19 input features and one target variable, `Exam_Score`.

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

The dataset also contains both numerical and categorical features, along with some missing categorical values.

## Machine Learning Pipeline

The preprocessing and model are combined into a Scikit-learn pipeline.

### Numerical Features

Numerical features are standardized using `StandardScaler`.

### Categorical Features

Categorical features are processed using:

1. `SimpleImputer(strategy="most_frequent")` for missing values
2. `OneHotEncoder(handle_unknown="ignore", drop="first")`

### Model

The final prediction model is Linear Regression.


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
├── requirements.txt
└── README.md