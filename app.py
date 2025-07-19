import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Set page configuration for a professional look
st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼", layout="centered")

# Load model and encoders
try:
    model = joblib.load("salary_predictor_model.joblib")
    label_encoders = joblib.load("label_encoders.joblib")
except Exception as e:
    st.error(f"Error loading model or encoders: {str(e)}")
    st.stop()

# Header
st.title("Employee Salary Prediction")
st.markdown("Enter employee details below to predict if their annual salary is ≤50K or >50K.")
st.info("Please fill in all fields accurately and click 'Predict' to see the result.")

# Organize inputs in two columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=17, max_value=100, value=30, help="Enter age (17-100)")
    workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"], help="Select employment type")
    fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1000000, value=200000, help="Enter final weight")
    education = st.selectbox("Education", ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"], help="Select education level")
    marital_status = st.selectbox("Marital Status", ["Never-married", "Married-civ-spouse", "Divorced", "Married-spouse-absent", "Separated", "Married-AF-spouse", "Widowed"], help="Select marital status")
    occupation = st.selectbox("Occupation", ["Adm-clerical", "Exec-managerial", "Handlers-cleaners", "Prof-specialty", "Other-service", "Sales", "Craft-repair", "Transport-moving", "Farming-fishing", "Machine-op-inspct", "Tech-support", "Protective-serv", "Armed-Forces", "Priv-house-serv"], help="Select occupation")

with col2:
    relationship = st.selectbox("Relationship", ["Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"], help="Select relationship status")
    race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"], help="Select race")
    gender = st.selectbox("Gender", ["Male", "Female"], help="Select gender")
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0, help="Enter capital gain")
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=10000, value=0, help="Enter capital loss")
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40, help="Enter hours worked per week")
    native_country = st.selectbox("Native Country", ["United-States", "Mexico", "Philippines", "Germany", "Canada", "Puerto-Rico", "El-Salvador", "India", "Cuba", "England", "Jamaica", "South", "China", "Italy", "Dominican-Republic", "Vietnam", "Guatemala", "Japan", "Poland", "Columbia", "Taiwan", "Haiti", "Iran", "Portugal", "Nicaragua", "Peru", "France", "Greece", "Ecuador", "Ireland", "Hong", "Cambodia", "Trinadad&Tobago", "Laos", "Thailand", "Yugoslavia", "Outlying-US(Guam-USVI-etc)", "Hungary", "Honduras", "Scotland"], help="Select native country")

# Education mapping
education_mapping = {
    'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5,
    '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10,
    'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14,
    'Prof-school': 15, 'Doctorate': 16
}
educational_num = education_mapping.get(education, 10)

# Prepare input data
try:
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [label_encoders['workclass'].transform([workclass])[0]],
        'fnlwgt': [fnlwgt],
        'education': [label_encoders['education'].transform([education])[0]],
        'educational-num': [educational_num],
        'marital-status': [label_encoders['marital-status'].transform([marital_status])[0]],
        'occupation': [label_encoders['occupation'].transform([occupation])[0]],
        'relationship': [label_encoders['relationship'].transform([relationship])[0]],
        'race': [label_encoders['race'].transform([race])[0]],
        'gender': [label_encoders['gender'].transform([gender])[0]],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [label_encoders['native-country'].transform([native_country])[0]]
    }).astype(np.float32)
except Exception as e:
    st.error(f"Error preparing input data: {str(e)}")
    st.stop()

# Predict button
if st.button("Predict", type="primary"):
    try:
        prediction = model.predict(input_data)
        income_label = label_encoders['income'].inverse_transform([prediction[0]])[0]
        st.success(f"Predicted Salary: **{income_label}**")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
