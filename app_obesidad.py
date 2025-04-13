import streamlit as st
import joblib
import pandas as pd
import numpy as np


model = joblib.load("modelo_obesidad.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")


st.title(" Predicción del Nivel de Obesidad")


st.header(" Ingresa los datos del paciente")

gender = st.selectbox("Género", ["Male", "Female"])
age = st.slider("Edad", 10, 100, 25)
height = st.number_input("Altura (en metros)", 1.0, 2.5, 1.70)
weight = st.number_input("Peso (en kg)", 30.0, 200.0, 70.0)
fhwo = st.selectbox("Antecedentes familiares de sobrepeso", ["yes", "no"])
favc = st.selectbox("¿Consume alimentos altos en calorías con frecuencia?", ["yes", "no"])
fcvc = st.slider("Frecuencia de consumo de vegetales (0-3)", 0.0, 3.0, 2.0)
ncp = st.slider("Número de comidas principales al día", 1, 4, 3)
caec = st.selectbox("¿Consume alimentos entre comidas?", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("¿Fuma?", ["yes", "no"])
ch2o = st.slider("Consumo de agua diario (litros)", 0.0, 3.0, 2.0)
scc = st.selectbox("¿Monitorea su consumo de calorías?", ["yes", "no"])
faf = st.slider("Nivel de actividad física semanal (0-3)", 0.0, 3.0, 1.5)
tue = st.slider("Tiempo frente a pantalla diario (0-2)", 0.0, 2.0, 1.0)
calc = st.selectbox("Frecuencia de consumo de alcohol", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Medio de transporte habitual", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])


nuevo_paciente = {
    'Gender': gender,
    'Age': age,
    'Height': height,
    'Weight': weight,
    'family_history_with_overweight': fhwo,
    'FAVC': favc,
    'FCVC': fcvc,
    'NCP': ncp,
    'CAEC': caec,
    'SMOKE': smoke,
    'CH2O': ch2o,
    'SCC': scc,
    'FAF': faf,
    'TUE': tue,
    'CALC': calc,
    'MTRANS': mtrans
}

df_nuevo = pd.DataFrame([nuevo_paciente])


for col, le in label_encoders.items():
    if col in df_nuevo.columns:
        df_nuevo[col] = le.transform(df_nuevo[col])

df_scaled = scaler.transform(df_nuevo)


if st.button("Predecir"):
    resultado = model.predict(df_scaled)
    nivel = label_encoders["NObeyesdad"].inverse_transform(resultado)
    st.success(f" Nivel de obesidad predicho: **{nivel[0]}**")
