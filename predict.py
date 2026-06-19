# predict.py

import joblib
import pandas as pd

model = joblib.load("model/model.pkl")

sample_input = {
    "Age": 45,
    "Gender": "Female",
    "Years at Company": 30,
    "Job Role": "Healthcare",
    "Monthly Income": 8104,
    "Work-Life Balance": "Fair",
    "Job Satisfaction": "High",
    "Performance Rating": "Average",
    "Number of Promotions": 0,
    "Overtime": "No",
    "Distance from Home": 38,
    "Education Level": "Associate Degree",
    "Marital Status": "Divorced",
    "Number of Dependents": 0,
    "Job Level": "Senior",
    "Company Size": "Large",
    "Company Tenure": 75,
    "Remote Work": "No",
    "Leadership Opportunities": "No",
    "Innovation Opportunities": "No",
    "Company Reputation": "Good",
    "Employee Recognition": "Low"
}

pred_df = pd.DataFrame([sample_input])
prediction = model.predict(pred_df)

print(prediction)

# 0 -> Stayed, 1 0 -> Left