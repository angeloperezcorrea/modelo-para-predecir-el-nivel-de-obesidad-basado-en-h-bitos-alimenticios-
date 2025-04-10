import joblib
import numpy as np
import pandas as pd


model = joblib.load("modelo_obesidad.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")


nuevo_paciente = {
    'Gender': 'Male',
    'Age': 25,
    'Height': 1.75,
    'Weight': 85,
    'family_history_with_overweight': 'yes',
    'FAVC': 'yes',
    'FCVC': 2.0,
    'NCP': 3,
    'CAEC': 'Sometimes',
    'SMOKE': 'no',
    'CH2O': 2.0,
    'SCC': 'no',
    'FAF': 1.5,
    'TUE': 1.0,
    'CALC': 'Sometimes',
    'MTRANS': 'Public_Transportation'
}


df_nuevo = pd.DataFrame([nuevo_paciente])


for col, le in label_encoders.items():
    if col in df_nuevo.columns:  
        df_nuevo[col] = le.transform(df_nuevo[col])


df_scaled = scaler.transform(df_nuevo)


prediccion = model.predict(df_scaled)
prediccion_label = label_encoders["NObeyesdad"].inverse_transform(prediccion)

print("✅ Predicción del nivel de obesidad:", prediccion_label[0])
