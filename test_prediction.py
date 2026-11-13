from src.prediction.predict import predict_student

student = {
    "Age": 21,
    "Gender": "Male",
    "Year_of_Study": 3,
    "Sleep_Hours": 7,
    "Study_Hours_Per_Week": 25,
    "GPA": 3.2,
    "Social_Media_Hours": 3,
    "Screen_Time_Hours": 6,
    "Physical_Activity": "Moderate",
    "Financial_Stress": 2,
    "Family_Support": 4,
    "Attendance_Rate": 85,
    "Part_Time_Job": "No",
    "Diet_Quality": "Good",
    "Previous_Mental_Health_History": "No"
}

prediction = predict_student(student)

print("Predicted category:", prediction)