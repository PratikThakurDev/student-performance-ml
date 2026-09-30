from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "StudentPerformanceFactors.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isna().sum())

numeric_df = df.select_dtypes(include="number")

correlations = (
    numeric_df
    .corr()["Exam_Score"]
    .sort_values(ascending=False)
)

print("\nCorrelation with Exam Score:")
print(correlations)

plt.hist(df["Exam_Score"], bins=20, edgecolor="black")

plt.xlabel("Exam Score")
plt.ylabel("Number of Students")
plt.title("Distribution of Exam Scores")

plt.show()


plt.scatter(df["Attendance"], df["Exam_Score"], alpha=0.5)

plt.xlabel("Attendance (%)")
plt.ylabel("Exam Score")
plt.title("Attendance vs Exam Score")

plt.show()


plt.scatter(df["Hours_Studied"], df["Exam_Score"], alpha=0.5)

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Hours Studied vs Exam Score")

plt.show()


motivation_scores = (
    df.groupby("Motivation_Level")["Exam_Score"]
    .mean()
)

plt.bar(motivation_scores.index, motivation_scores.values)

plt.xlabel("Motivation Level")
plt.ylabel("Average Exam Score")
plt.title("Average Exam Score by Motivation Level")

plt.show()