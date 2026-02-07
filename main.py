import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Page Config ---
st.set_page_config(page_title="Heart Health Predictor", layout="centered")

# --- Step 1: Data Loading & Preprocessing ---
@st.cache_data
def load_data():
    # Loading the UCI Cleveland dataset via a reliable URL
    url = "https://raw.githubusercontent.com/dataprofessor/data/master/heart-disease-cleveland.csv"
    df = pd.read_csv(url)
    
    # Simple Preprocessing: The dataset uses 'target' where 0=healthy, 1-4=disease
    # We simplify this to 0 (No Disease) and 1 (Disease)
    df['target'] = df.target.apply(lambda x: 1 if x > 0 else 0)
    return df

df = load_data()

# --- Step 2: Model Training ---
X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Step 3: Streamlit UI ---
st.title("❤️ Heart Disease Prediction Tool")
st.write("Enter the patient's clinical parameters below to predict the presence of heart disease.")

# Create columns for input
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 100, 50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.slider("ST depression induced by exercise", 0.0, 6.0, 1.0)

# Map inputs for prediction (simplified for this example)
# Note: Ensure order matches the dataframe columns
user_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, 1, 0, 1]]) # Added dummy values for remaining columns
user_data_scaled = scaler.transform(user_data)

# --- Step 4: Prediction Output ---
if st.button("Predict Result"):
    prediction = model.predict(user_data_scaled)
    probability = model.predict_proba(user_data_scaled)
    
    st.subheader("Results:")
    if prediction[0] == 1:
        st.error(f"Warning: High Risk detected. (Confidence: {probability[0][1]:.2%})")
    else:
        st.success(f"Low Risk detected. (Confidence: {probability[0][0]:.2%})")

st.info("**Disclaimer:** This tool is for educational purposes only and does not replace professional medical advice.")
