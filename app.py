import streamlit as st
import joblib
import numpy as np
import os

# ระบุ path ของโมเดลที่บันทึกไว้ใน Google Drive
model_path = '/content/drive/MyDrive/!!!Workshop698-1-67/TEST_PENGUIN/random_forest_model.pkl'

# โหลดโมเดล Random Forest ที่บันทึกไว้จาก path ที่ระบุ
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.write("โมเดลถูกโหลดเรียบร้อยแล้ว")
else:
    st.write(f"ไม่พบไฟล์โมเดลที่ {model_path}")

# ฟังก์ชันทำนายประเภทของเพนกวิน
def predict_penguin(species_features):
    prediction = model.predict([species_features])
    return prediction[0]

# UI บน Streamlit
st.title("Penguin Species Prediction")
st.write("กรุณาใส่ข้อมูลของเพนกวินเพื่อทำนาย species")

# รับข้อมูล input จากผู้ใช้
bill_length = st.number_input("Bill Length (mm)")
bill_depth = st.number_input("Bill Depth (mm)")
flipper_length = st.number_input("Flipper Length (mm)")
body_mass = st.number_input("Body Mass (g)")

# แปลงข้อมูล input ให้เป็น array
input_features = np.array([bill_length, bill_depth, flipper_length, body_mass])

# ปุ่มทำนาย
if st.button("Predict"):
    if 'model' in locals():  # ตรวจสอบว่าโมเดลถูกโหลดหรือไม่
        species = predict_penguin(input_features)
        st.write(f"The predicted species is: {species}")
    else:
        st.write("ไม่สามารถทำนายได้เนื่องจากโมเดลไม่ได้ถูกโหลด")
