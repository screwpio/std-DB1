
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from collections import defaultdict, Counter

from recommender import recommend_courses
from ml_model import predict_new_student

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df_info = pd.read_csv(r"C:\Users\mhnd7\Downloads\student-db-web\backend\data\Students List(Student List).csv")
df_info["Student ID"] = pd.to_numeric(df_info["Student ID"], errors="coerce")
df_info.dropna(subset=["Student ID"], inplace=True)
df_info["Student ID"] = df_info["Student ID"].astype(int)

df_preds = pd.read_csv(r"C:\Users\mhnd7\Downloads\student-db-web\backend\data\student_predictions.csv")

@app.get("/")
def root():
    return {"message": "Student Predictor API running."}

@app.get("/existing/{student_id}")
def get_existing_student(student_id: int):
    student = df_info[df_info["Student ID"] == student_id]
    preds = df_preds[df_preds["Student ID"] == student_id]

    if student.empty:
        raise HTTPException(status_code=404, detail="Student not found")

    s = student.iloc[0]
    taken = student[["Course", "Course Title"]].drop_duplicates().to_dict(orient="records")
    predicted = preds[["Predicted Course", "Course Title", "Confidence"]].to_dict(orient="records")

    return {
        "student": {
            "id": student_id,
            "gender": s["Gender"],
            "major": s["Major Applied for"],
            "age": s["Age When Applied"]
        },
        "courses_taken": taken,
        "predicted_courses": predicted
    }

class NewStudentRequest(BaseModel):
    age: float
    gender: str
    major: str
    top_n: int = 3

@app.post("/predict_new")
def predict_new(data: NewStudentRequest):
    try:
        preds = predict_new_student(data.age, data.gender, data.major, data.top_n)
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/course_summary")
def get_course_summary(course: str = Query("All")):
    course_counts = df_preds["Predicted Course"].value_counts().reset_index()
    course_counts.columns = ["Course", "Predicted Count"]

    if course != "All":
        course_counts = course_counts[course_counts["Course"] == course]

    result = course_counts.to_dict(orient="records")
    return {"summary": result}
