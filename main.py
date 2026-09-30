import pandas as pd

df = pd.read_csv("data/StudentPerformanceFactors.csv")

# print(df.head())
# print(df.shape)
# print(df.columns)
# df.info()
# print(df.describe())

# print("\nMissing values:")
# print(df.isna().sum())

# print("\nDuplicate rows:")
# print(df.duplicated().sum())

# print(df["Teacher_Quality"].value_counts(dropna=False))
# print()
# print(df["Parental_Education_Level"].value_counts(dropna=False))
# print()
# print(df["Distance_from_Home"].value_counts(dropna=False))

# df["Teacher_Quality"] = df["Teacher_Quality"].fillna(   "only for learning pandas"
#     df["Teacher_Quality"].mode()[0]
# )

# df["Parental_Education_Level"] = df["Parental_Education_Level"].fillna(
#     df["Parental_Education_Level"].mode()[0]
# )

# df["Distance_from_Home"] = df["Distance_from_Home"].fillna(
#     df["Distance_from_Home"].mode()[0]
# )

# print(df.isna().sum()) 

# import matplotlib
# matplotlib.use("QtAgg")
# import matplotlib.pyplot as plt

# plt.scatter(df["Hours_Studied"], df["Exam_Score"])

# plt.xlabel("Hours Studied")
# plt.ylabel("Exam Score")
# plt.title("Hours Studied vs Exam Score")

# plt.show()

# numeric_df = df.select_dtypes(include="number")

# print(numeric_df.corr()["Exam_Score"].sort_values(ascending=False))

# motivation_scores = df.groupby("Motivation_Level")["Exam_Score"].mean()

# import matplotlib.pyplot as plt

# plt.bar(motivation_scores.index, motivation_scores.values)

# plt.xlabel("Motivation Level")
# plt.ylabel("Average Exam Score")
# plt.title("Average Exam Score by Motivation Level")

# plt.show()

X = df.drop(columns=["Exam_Score"])
y = df["Exam_Score"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# print(X_train.shape)
# print(X_test.shape)

numerical_columns = X.select_dtypes(include="number").columns
categorical_columns = X.select_dtypes(exclude="number").columns

# print("Numerical:")
# print(numerical_columns)

# print("\nCategorical:")
# print(categorical_columns)


from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

numerical_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore",drop="first"))
])

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            categorical_pipeline,
            categorical_columns
        ),
    ],
    remainder="passthrough"
)

preprocessor_scaled = ColumnTransformer([
    ("cat", categorical_pipeline, categorical_columns),
    ("num", numerical_pipeline, numerical_columns)
])


from sklearn.linear_model import LinearRegression

model = Pipeline([
    ("preprocessor", preprocessor_scaled),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# print(y_pred[:10])
# print(y_test.iloc[:10].values)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

# print("MAE:", mae)  
# print("RMSE:", rmse) 
# print("R2:", r2) 

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

results["Error"] = abs(
    results["Actual"] - results["Predicted"]
)

# print(
#     results.sort_values(
#         "Error",
#         ascending=False
#     ).head(10)
# )

# print(df.loc[217])
# print(
#     df[df["Exam_Score"] >= 80]
#     .sort_values("Exam_Score", ascending=False)
# )
# print("Scores >= 80:", (df["Exam_Score"] >= 80).sum())

# print(df["Exam_Score"].describe())

# print("\nScore counts:")
# print(df["Exam_Score"].value_counts().sort_index())

# import matplotlib.pyplot as plt

# plt.hist(df["Exam_Score"], bins=30)

# plt.xlabel("Exam Score")
# plt.ylabel("Number of Students")
# plt.title("Distribution of Exam Scores")

# plt.show()

from sklearn.tree import DecisionTreeRegressor

tree_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor(random_state=42))
])

tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)

tree_mae = mean_absolute_error(y_test, tree_pred)

tree_mse = mean_squared_error(y_test, tree_pred)
tree_rmse = np.sqrt(tree_mse)

tree_r2 = r2_score(y_test, tree_pred)

# print("MAE:", tree_mae)  
# print("RMSE:", tree_rmse) 
# print("R2:", tree_r2) 

from sklearn.ensemble import RandomForestRegressor

forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

forest_model.fit(X_train, y_train)

forest_pred = forest_model.predict(X_test)

forest_model.fit(X_train, y_train)

forest_pred = forest_model.predict(X_test)

forest_mae = mean_absolute_error(y_test, forest_pred)

forest_mse = mean_squared_error(y_test, forest_pred)
forest_rmse = np.sqrt(forest_mse)

forest_r2 = r2_score(y_test, forest_pred)

# print("MAE:", forest_mae)  
# print("RMSE:", forest_rmse) 
# print("R2:", forest_r2) 

from sklearn.model_selection import GridSearchCV

param_grid = {
    "regressor__n_estimators": [50, 100, 200],
    "regressor__max_depth": [5, 10, None]
}

grid_search = GridSearchCV(
    forest_model,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# print("Best parameters:", grid_search.best_params_)
# print("Best CV score:", grid_search.best_score_)

best_forest = grid_search.best_estimator_

tuned_forest_pred = best_forest.predict(X_test)

tuned_forest_mae = mean_absolute_error(y_test, tuned_forest_pred)

tuned_forest_mse = mean_squared_error(y_test, tuned_forest_pred)
tuned_forest_rmse = np.sqrt(tuned_forest_mse)

tuned_forest_r2 = r2_score(y_test, tuned_forest_pred)

# print("MAE:", tuned_forest_mae)
# print("RMSE:", tuned_forest_rmse)
# print("R2:", tuned_forest_r2)

from sklearn.ensemble import GradientBoostingRegressor

boost_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    ))
])

boost_model.fit(X_train, y_train)

boost_pred = boost_model.predict(X_test)

boost_mae = mean_absolute_error(y_test, boost_pred)

boost_mse = mean_squared_error(y_test, boost_pred)
boost_rmse = np.sqrt(boost_mse)

boost_r2 = r2_score(y_test, boost_pred)

# print("MAE:", boost_mae)
# print("RMSE:", boost_rmse)
# print("R2:", boost_r2)

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}

from sklearn.model_selection import cross_val_score

for name, regressor in models.items():

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    scores = cross_val_score(
        model_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    rmse_scores = np.sqrt(-scores)

    # print(name)
    # print("Fold RMSE:", rmse_scores)
    # print("Mean RMSE:", rmse_scores.mean())
    # print()

from sklearn.linear_model import Ridge

ridge_model = Pipeline([
    ("preprocessor", preprocessor_scaled),
    ("regressor", Ridge(alpha=1.0))
])

ridge_scores = cross_val_score(
    ridge_model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

ridge_rmse = np.sqrt(-ridge_scores)

# print("Fold RMSE:", ridge_rmse)
# print("Mean RMSE:", ridge_rmse.mean())


from sklearn.linear_model import Lasso

lasso_model = Pipeline([
    ("preprocessor", preprocessor_scaled),
    ("regressor", Lasso(
        alpha=0.1,
        max_iter=10000
    ))
])

lasso_scores = cross_val_score(
    lasso_model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

lasso_rmse = np.sqrt(-lasso_scores)

# print("Fold RMSE:", lasso_rmse)
# print("Mean RMSE:", lasso_rmse.mean())

ridge_params = {
    "regressor__alpha": [0.001, 0.01, 0.1, 1, 10, 100]
}

ridge_grid = GridSearchCV(
    ridge_model,
    ridge_params,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

ridge_grid.fit(X_train, y_train)

# print("Best alpha:", ridge_grid.best_params_)
# print("Best CV MSE:", -ridge_grid.best_score_)
# print("Best CV RMSE:", np.sqrt(-ridge_grid.best_score_))

lasso_params = {
    "regressor__alpha": [0.001, 0.01, 0.1, 1, 10]
}

lasso_grid = GridSearchCV(
    lasso_model,
    lasso_params,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

lasso_grid.fit(X_train, y_train)

# print("Best alpha:", lasso_grid.best_params_)
# print("Best CV MSE:", -lasso_grid.best_score_)
# print("Best CV RMSE:", np.sqrt(-lasso_grid.best_score_))


linear_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

linear_cv_rmse = np.sqrt(-linear_scores.mean())

print("Linear CV RMSE:", linear_cv_rmse)

# print(
#     "Ridge CV RMSE:",
#     np.sqrt(-ridge_grid.best_score_)
# )

# print(
#     "Lasso CV RMSE:",
#     np.sqrt(-lasso_grid.best_score_)
# )

feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

coefficients = (
    model.named_steps["regressor"]
    .coef_
)

print(len(feature_names))
print(len(coefficients))

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

coef_df = coef_df.sort_values(
    "Coefficient",
    ascending=False
)

# print(coef_df)

import joblib

joblib.dump(model, "student_score_model.pkl")

print("Model saved successfully")