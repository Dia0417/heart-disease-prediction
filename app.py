import streamlit as st
import pandas as pd
import joblib

model = joblib.load('heart_disease_svm_model.pkl')
scaler = joblib.load('heart_disease_scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

st.set_page_config(page_title="Heart Disease Risk Prediction", page_icon="❤️", layout="centered")

# --- Custom CSS ---
st.set_page_config(page_title="Heart Disease Risk Prediction", page_icon="❤️", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #ffecec 0%, #fff5f5 50%, #ffe8e8 100%);
    }
    .title-banner {
        background: linear-gradient(90deg, #d9534f, #c0392b);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0px 4px 15px rgba(192,57,43,0.3);
    }
    .title-banner h1 {
        color: white !important;
        margin: 0;
    }
    .title-banner p {
        color: #ffe0e0;
        margin: 5px 0 0 0;
    }
    div[data-testid="stForm"] {
        background-color: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.1);
        border: 1px solid #ffd6d6;
    }
    .stButton>button {
        background: linear-gradient(90deg, #d9534f, #c0392b);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 12px 24px;
        width: 100%;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #c0392b, #a93226);
    }
    label {
        font-weight: 600 !important;
        color: #333 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-banner">
        <h1>❤️ Heart Disease Risk Prediction</h1>
        <p>Apni health details enter karein risk assess karne ke liye</p>
    </div>
""", unsafe_allow_html=True)
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=25,
            help="Aapki umar (saal mein)")
        sex = st.selectbox("Sex", ["Male", "Female"])
        trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=115,
            help="Normal range: 110-120 mmHg (healthy young adult)")
        chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=170,
            help="Normal range: 150-180 mg/dl for young healthy adult")
        thalch = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=195,
            help="Formula: 220 - age. 20 saal ke liye ~200 normal hai")
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.0, value=0.0, step=0.1,
            help="ECG test ka reading. Healthy person mein usually 0.0 hota hai")
        ca = st.selectbox("Major Vessels Blocked (ca)", [0, 1, 2, 3],
            help="Kitni major blood vessels blocked hain. Healthy person: 0")

    with col2:
        dataset = st.selectbox("Dataset Origin", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"],
            help="Ye original data collection location hai — apne liye 'Cleveland' select karo")
        cp = st.selectbox("Chest Pain Type", ["asymptomatic", "atypical angina", "non-anginal", "typical angina"],
            help="Angina = seene mein dard jo dil tak khoon kam pohanchne se hota hai. Agar koi pain nahi to 'asymptomatic' choose karo")
        fbs = st.selectbox("Fasting Blood Sugar > 120", ["False", "True"],
            help="Fasting sugar test 120 se zyada hai? Normal healthy person: False")
        restecg = st.selectbox("Resting ECG", ["lv hypertrophy", "normal", "st-t abnormality"],
            help="ECG test ka result. Healthy person: 'normal'")
        exang = st.selectbox("Exercise Induced Angina", ["False", "True"],
            help="Exercise/walk karte waqt chest pain hota hai? Healthy: False")
        slope = st.selectbox("Slope of ST Segment", ["downsloping", "flat", "upsloping"],
            help="ECG graph ka slope. Healthy young person: 'upsloping'")
        thal = st.selectbox("Thalassemia", ["fixed defect", "normal", "reversable defect"],
            help="Blood disorder test. Healthy: 'normal'")

    submitted = st.form_submit_button("🔍 Predict Risk")

if submitted:
    input_data = {col: 0 for col in feature_columns}
    input_data['age'] = age
    input_data['sex'] = 1 if sex == "Male" else 0
    input_data['trestbps'] = trestbps
    input_data['chol'] = chol
    input_data['thalch'] = thalch
    input_data['oldpeak'] = oldpeak
    input_data['ca'] = ca

    if dataset != "Cleveland":
        col_name = f"dataset_{dataset}"
        if col_name in input_data: input_data[col_name] = True
    if cp != "asymptomatic":
        col_name = f"cp_{cp}"
        if col_name in input_data: input_data[col_name] = True
    if fbs == "True":
        input_data['fbs_True'] = True
    if restecg != "lv hypertrophy":
        col_name = f"restecg_{restecg}"
        if col_name in input_data: input_data[col_name] = True
    if exang == "True":
        input_data['exang_True'] = True
    if slope != "downsloping":
        col_name = f"slope_{slope}"
        if col_name in input_data: input_data[col_name] = True
    if thal != "fixed defect":
        col_name = f"thal_{thal}"
        if col_name in input_data: input_data[col_name] = True

    input_df = pd.DataFrame([input_data])[feature_columns]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Heart Disease**\n\nProbability: {probability:.1%}")
    else:
        st.success(f"✅ **Low Risk of Heart Disease**\n\nProbability: {probability:.1%}")