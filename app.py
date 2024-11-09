import streamlit as st
import joblib
import numpy as np

# โหลดโมเดล Random Forest ที่บันทึกไว้
model = joblib.load('random_forest_model.pkl')

# ฟังก์ชันทำนายประเภทของเพนกวิน
def predict_penguin(species_features):
    prediction = model.predict(species_features)
    return prediction[0]

# UI บน Streamlit
st.title("Penguin Species Prediction")
st.write("กรุณาใส่ข้อมูลของเพนกวินเพื่อทำนาย species")

# รับข้อมูล input จากผู้ใช้
bill_length = st.number_input("Bill Length (mm)")
bill_depth = st.number_input("Bill Depth (mm)")
flipper_length = st.number_input("Flipper Length (mm)")
body_mass = st.number_input("Body Mass (g)")

# แปลงข้อมูล input ให้เป็น array 2D (1, 4)
input_features = np.array([[bill_length, bill_depth, flipper_length, body_mass]])

# ปุ่มทำนาย
if st.button("Predict"):
    species = predict_penguin(input_features)
    st.write(f"The predicted species is: {species}")
