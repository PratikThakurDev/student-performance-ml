import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "student_score_model.pkl"

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Predictor")

st.write(
    "Enter the student's academic and personal information below "
    "to estimate their exam score using a trained machine learning model."
)

st.divider()

st.write(
    "Enter student information below to predict the expected exam score."
)

st.subheader("Student Information")

col1, col2 = st.columns(2)

with col1:
    hours_studied = st.number_input(
        "Hours Studied", min_value=0, max_value=50, value=20
    )

    sleep_hours = st.number_input(
        "Sleep Hours", min_value=0, max_value=24, value=7
    )

    tutoring_sessions = st.number_input(
        "Tutoring Sessions", min_value=0, max_value=20, value=1
    )

with col2:
    attendance = st.number_input(
        "Attendance (%)", min_value=0, max_value=100, value=80
    )

    previous_scores = st.number_input(
        "Previous Score", min_value=0, max_value=100, value=70
    )

    physical_activity = st.number_input(
        "Physical Activity", min_value=0, max_value=10, value=3
    )

with col1:
    parental_involvement = st.selectbox(
        "Parental Involvement",
        ["Low", "Medium", "High"]
    )

    access_to_resources = st.selectbox(
        "Access to Resources",
        ["Low", "Medium", "High"]
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        ["No", "Yes"]
    )

    motivation = st.selectbox(
        "Motivation Level",
        ["Low", "Medium", "High"]
    )

    internet_access = st.selectbox(
        "Internet Access",
        ["No", "Yes"]
    )

    family_income = st.selectbox(
        "Family Income",
        ["Low", "Medium", "High"]
    )

    teacher_quality = st.selectbox(
        "Teacher Quality",
        ["Low", "Medium", "High"]
    )


with col2:
    school_type = st.selectbox(
        "School Type",
        ["Public", "Private"]
    )

    peer_influence = st.selectbox(
        "Peer Influence",
        ["Negative", "Neutral", "Positive"]
    )

    learning_disabilities = st.selectbox(
        "Learning Disabilities",
        ["No", "Yes"]
    )

    parental_education = st.selectbox(
        "Parental Education Level",
        ["High School", "College", "Postgraduate"]
    )

    distance_from_home = st.selectbox(
        "Distance From Home",
        ["Near", "Moderate", "Far"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

if st.button("Predict Score"):

    new_student = pd.DataFrame([{
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": access_to_resources,
        "Extracurricular_Activities": extracurricular,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Motivation_Level": motivation,
        "Internet_Access": internet_access,
        "Tutoring_Sessions": tutoring_sessions,
        "Family_Income": family_income,
        "Teacher_Quality": teacher_quality,
        "School_Type": school_type,
        "Peer_Influence": peer_influence,
        "Physical_Activity": physical_activity,
        "Learning_Disabilities": learning_disabilities,
        "Parental_Education_Level": parental_education,
        "Distance_from_Home": distance_from_home,
        "Gender": gender
    }])

    prediction = model.predict(new_student)[0]

    st.divider()

    st.subheader("Prediction")

    st.metric(
        label="Predicted Exam Score",
        value=f"{prediction:.2f}"
    )

    st.caption(
        "Prediction generated using a Linear Regression model trained "
        "on the Student Performance Factors dataset."
    )