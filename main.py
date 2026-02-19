import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/dataprofessor/data/master/heart-disease-cleveland.csv"
    df = pd.read_csv(url)

    # Convert target: 0 = no disease, 1–4 = disease
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    return df

df = load_data()

# ---------------- PREPROCESSING ----------------
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------- MODELS ----------------
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel="rbf", probability=True, random_state=42)

models = {
    "Decision Tree": dt,
    "Random Forest": rf,
    "SVM": svm
}

# ---------------- TRAINING + 5-FOLD CV ----------------
model_accuracy = {}

for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=5)
    model_accuracy[name] = scores.mean()

# choose best model automatically
best_model_name = max(model_accuracy, key=model_accuracy.get)
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# ---------------- STREAMLIT UI ----------------
st.title("❤️ Heart Disease Prediction Tool")
st.write("Provide the patient’s clinical features to predict heart disease risk.")

st.subheader("📊 Model Performance (5-Fold CV)")
for name, acc in model_accuracy.items():
    st.write(f"**{name} Accuracy:** {acc*100:.2f}%")

st.success(f"🥇 **Best Model Selected Automatically: {best_model_name}**")

# ---------------- USER INPUT ----------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 100, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise-Induced Angina", [0, 1])
    oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)

# UCI dataset remaining features
slope = st.selectbox("Slope of ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect)", [1, 2, 3])

# Arrange user input in exact order of dataset
user_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                       thalach, exang, oldpeak, slope, ca, thal]])

user_scaled = scaler.transform(user_data)

# ---------------- PREDICTION ----------------
if st.button("Predict Result"):
    pred = best_model.predict(user_scaled)[0]
    prob = best_model.predict_proba(user_scaled)[0]

    st.subheader("🔍 Prediction Outcome")
    if pred == 1:
        st.error(f"⚠ High Risk of Heart Disease (Confidence: {prob[1]*100:.2f}%)")
    else:
        st.success(f"✔ Low Risk of Heart Disease (Confidence: {prob[0]*100:.2f}%)")

st.info("**Disclaimer:** This tool is for educational purposes only, not medical diagnosis.")
