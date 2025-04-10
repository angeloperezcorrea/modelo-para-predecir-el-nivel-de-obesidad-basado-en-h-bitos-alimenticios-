import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib


df = pd.read_csv("ObesityDataSet.csv")


label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)


nuevo_dato = {
    'Gender': 'Male',
    'Age': 25,
    'Height': 1.75,
    'Weight': 85,
    'family_history_with_overweight': 'yes',
    'FAVC': 'yes',
    'FCVC': 2,
    'NCP': 3,
    'CAEC': 'Sometimes',
    'SMOKE': 'no',
    'CH2O': 2,
    'SCC': 'no',
    'FAF': 1,
    'TUE': 1,
    'CALC': 'Sometimes',
    'MTRANS': 'Public_Transportation'
}


df_nuevo = pd.DataFrame([nuevo_dato])


for col, le in label_encoders.items():
    if col in df_nuevo.columns:
        df_nuevo[col] = le.transform(df_nuevo[col])


df_nuevo_scaled = scaler.transform(df_nuevo)


prediccion = model.predict(df_nuevo_scaled)
nivel_obesidad = label_encoders['NObeyesdad'].inverse_transform(prediccion)

print("👉 Nivel de obesidad predicho:", nivel_obesidad[0])
