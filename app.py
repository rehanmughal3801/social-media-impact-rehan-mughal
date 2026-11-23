import streamlit as st
import pandas as pd
import pickle

# Load the saved model and encoders
@st.cache_resource
def load_artifacts():
    with open('social_media_impact_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_artifacts()

st.set_page_config(page_title="Social Media Impact Predictor", page_icon="📱", layout="centered")
st.title("📱 Social Media Impact on Life Predictor")
st.write("Enter student metrics below to predict the overall impact of social media.")

# Input Form matching the exact 15 features used during training
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        student_id = st.selectbox("Student ID", options=encoders['Student_ID'].classes_)
        age = st.number_input("Age", min_value=10, max_value=60, value=19)
        gender = st.selectbox("Gender", options=encoders['Gender'].classes_)
        academic_level = st.selectbox("Academic Level", options=encoders['Academic_Level'].classes_)
        primary_platform = st.selectbox("Primary Platform", options=encoders['Primary_Platform'].classes_)
        daily_usage = st.slider("Daily Usage Hours", 0.0, 24.0, 5.0, step=0.1)
        weekend_extra = st.slider("Weekend Extra Hours", 0.0, 24.0, 1.5, step=0.1)
        
    with col2:
        device_type = st.selectbox("Device Type", options=encoders['Device_Type'].classes_)
        sleep_duration = st.slider("Sleep Duration Hours", 0.0, 16.0, 7.0, step=0.1)
        sleep_quality = st.slider("Sleep Quality Score (1-5)", 1, 5, 3)
        late_night = st.checkbox("Late Night Usage", value=True)
        social_comparison = st.selectbox("Social Comparison Frequency", options=encoders['Social_Comparison_Frequency'].classes_)
        stress_score = st.number_input("Perceived Stress Score", min_value=0.0, max_value=50.0, value=18.0, step=0.5)
        mental_health = st.number_input("Mental Health Index", min_value=0, max_value=100, value=75)
        gpa = st.number_input("Academic Performance GPA", min_value=0.0, max_value=4.0, value=3.20, step=0.01)

    submitted = st.form_submit_button("Predict Impact", use_container_width=True)

if submitted:
    # Create DataFrame with exact column order from training
    input_data = pd.DataFrame([{
        'Student_ID': student_id,
        'Age': age,
        'Gender': gender,
        'Academic_Level': academic_level,
        'Primary_Platform': primary_platform,
        'Daily_Usage_Hours': daily_usage,
        'Weekend_Extra_Hours': weekend_extra,
        'Device_Type': device_type,
        'Sleep_Duration_Hours': sleep_duration,
        'Sleep_Quality_Score': sleep_quality,
        'Late_Night_Usage': late_night,
        'Social_Comparison_Frequency': social_comparison,
        'Perceived_Stress_Score': stress_score,
        'Mental_Health_Index': mental_health,
        'Academic_Performance_GPA': gpa
    }])

    # Apply LabelEncoders to categorical columns
    for col, le in encoders.items():
        if col in input_data.columns:
            input_data[col] = le.transform(input_data[col].astype(str))

    # Make prediction
    pred_encoded = model.predict(input_data)[0]
    pred_label = encoders['Overall_Impact'].inverse_transform([pred_encoded])[0]
    
    # Display result
    st.subheader(f"Predicted Overall Impact: **{pred_label}**")
