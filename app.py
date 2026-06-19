"""Streamlit UI for the Employee Churn model.

Run with:  .venv/bin/streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "model/model.pkl"

# OPTIONS
ORDINAL = {
    "Work-Life Balance":    ["Poor", "Fair", "Good", "Excellent"],
    "Job Satisfaction":     ["Low", "Medium", "High", "Very High"],
    "Performance Rating":   ["Low", "Below Average", "Average", "High"],
    "Education Level":      ["High School", "Associate Degree", "Bachelor’s Degree", "Master’s Degree", "PhD"],
    "Job Level":            ["Entry", "Mid", "Senior"],
    "Company Size":         ["Small", "Medium", "Large"],
    "Company Reputation":   ["Poor", "Fair", "Good", "Excellent"],
    "Employee Recognition": ["Low", "Medium", "High", "Very High"],
}
NOMINAL = {
    "Gender":                   ["Male", "Female"],
    "Job Role":                 ["Education", "Media", "Healthcare", "Technology", "Finance"],
    "Overtime":                 ["No", "Yes"],
    "Marital Status":           ["Single", "Married", "Divorced"],
    "Remote Work":              ["No", "Yes"],
    "Leadership Opportunities": ["No", "Yes"],
    "Innovation Opportunities": ["No", "Yes"],
}
# Continuous numeric: (min, max, default) taken from the training data ranges.
CONTINUOUS = {
    "Age":                (18, 59, 39),
    "Years at Company":   (1, 51, 13),
    "Monthly Income":     (1000, 100000, 7354),
    "Distance from Home": (1, 99, 50),
    "Company Tenure":     (2, 128, 56),
}
# Discrete counts: (min, max, default) -> rendered as integer sliders.
COUNTS = {
    "Number of Promotions": (0, 4, 0),
    "Number of Dependents": (0, 6, 0),
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def main():
    st.set_page_config(page_title="Employee Churn Predictor", page_icon="\U0001F464")
    st.title("\U0001F464 Employee Churn Predictor")
    st.caption("Fill in the employee's details and predict whether they are likely to leave.")

    model = load_model()
    values = {}

    ## NUMERIC
    st.subheader("Numeric")
    c1, c2 = st.columns(2)
    for i, (col, (lo, hi, default)) in enumerate(CONTINUOUS.items()):
        with (c1 if i % 2 == 0 else c2):
            values[col] = st.number_input(col, min_value=lo, max_value=hi, value=default, step=1)

    ## COUNT
    st.subheader("Counts")
    c3, c4 = st.columns(2)
    for i, (col, (lo, hi, default)) in enumerate(COUNTS.items()):
        with (c3 if i % 2 == 0 else c4):
            # slider -> so that decimal analomy
            values[col] = st.slider(col, min_value=lo, max_value=hi, value=default, step=1)

    ## CATEGORICAL
    st.subheader("Categorical")
    c5, c6 = st.columns(2)
    all_cats = list(ORDINAL.items()) + list(NOMINAL.items())
    for i, (col, opts) in enumerate(all_cats):
        with (c5 if i % 2 == 0 else c6):
            values[col] = st.selectbox(col, opts)


    # PREDICT
    if st.button("Predict", type="primary"):
        X = pd.DataFrame([values])
        proba_leave = float(model.predict_proba(X)[0, 1])  # class 1 = "Left"
        if proba_leave >= 0.5:
            st.error(f"Likely to LEAVE  —  churn probability {proba_leave:.1%}")
        else:
            st.success(f"Likely to STAY  —  churn probability {proba_leave:.1%}")
        st.progress(proba_leave)


if __name__ == "__main__":
    main()
