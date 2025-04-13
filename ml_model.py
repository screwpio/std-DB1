
import joblib
import pandas as pd
import numpy as np

model = joblib.load("data/course_model.pkl")
mlb = joblib.load("data/mlb.pkl")
input_columns = joblib.load("data/input_columns.pkl")
course_title_map = joblib.load("data/course_title_map.pkl")

def predict_new_student(age, gender, major, top_n=3):
    new_input = pd.DataFrame([{
        "Age When Applied": age,
        "Gender": gender,
        "Major Applied for": major
    }])
    new_input = pd.get_dummies(new_input).reindex(columns=input_columns, fill_value=0)
    y_pred = model.predict_proba(new_input)
    scores = np.array([probs[:, 1] if probs.shape[1] == 2 else np.zeros(probs.shape[0]) for probs in y_pred]).flatten()
    top_indices = np.argsort(scores)[-top_n:][::-1]
    predictions = [{"course": mlb.classes_[i], "title": course_title_map.get(mlb.classes_[i], "Unknown")} for i in top_indices]
    return predictions
