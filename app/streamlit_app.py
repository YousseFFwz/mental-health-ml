import streamlit as st
import sys
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.prediction.predict import predict_student


# Page configuration
st.set_page_config(
    page_title="Student Mental Health ML",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("Student Mental Health Classification")
st.write(
    "An educational machine learning application "
    "for exploring student survey data."
)

st.info(
    "This application provides experimental model classifications. "
    "It is not a mental health diagnosis or assessment."
)

st.divider()

st.subheader("Student Data")

# Input form
with st.form("student_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=15, max_value=50, value=21)
        gender = st.selectbox("Gender", ["Male", "Female"])
        year = st.selectbox("Year of Study", [1, 2, 3, 4, 5])
        sleep = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
        study = st.number_input("Study Hours Per Week", 0.0, 100.0, 25.0)
        gpa = st.number_input("GPA", 0.0, 4.0, 3.2)
        social = st.number_input("Social Media Hours", 0.0, 24.0, 3.0)

    with col2:
        screen = st.number_input("Screen Time Hours", 0.0, 24.0, 6.0)
        activity = st.selectbox(
            "Physical Activity",
            ["Low", "Moderate", "High"]
        )
        financial = st.selectbox("Financial Stress", [1, 2, 3, 4, 5])
        family = st.selectbox("Family Support", [1, 2, 3, 4, 5])
        attendance = st.number_input("Attendance Rate", 0.0, 100.0, 85.0)
        job = st.selectbox("Part Time Job", ["Yes", "No"])
        diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
        history = st.selectbox(
            "Previous Mental Health History",
            ["Yes", "No"]
        )

    submitted = st.form_submit_button("Generate Classification")

    if submitted:
        student = {
            "Age": age,
            "Gender": gender,
            "Year_of_Study": year,
            "Sleep_Hours": sleep,
            "Study_Hours_Per_Week": study,
            "GPA": gpa,
            "Social_Media_Hours": social,
            "Screen_Time_Hours": screen,
            "Physical_Activity": activity,
            "Financial_Stress": financial,
            "Family_Support": family,
            "Attendance_Rate": attendance,
            "Part_Time_Job": job,
            "Diet_Quality": diet,
            "Previous_Mental_Health_History": history
        }

        try:
            prediction = predict_student(student)

            st.success(f"Predicted category: {prediction}")

        except Exception as error:
            st.error(f"Prediction error: {error}")