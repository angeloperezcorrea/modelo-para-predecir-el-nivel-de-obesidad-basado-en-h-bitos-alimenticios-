import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("ObesityDataSet.csv")


plt.figure(figsize=(10, 6))
sns.boxplot(x="NObeyesdad", y="FAF", data=df, palette="coolwarm")
plt.title("Actividad Física (FAF) vs Nivel de Obesidad")
plt.ylabel("Actividad Física (horas por semana)")
plt.xlabel("Nivel de Obesidad")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x="NObeyesdad", y="WATER", data=df, palette="YlGnBu")
plt.title("Consumo de Agua vs Nivel de Obesidad")
plt.ylabel("Litros de Agua por Día")
plt.xlabel("Nivel de Obesidad")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x="NObeyesdad", y="FCVC", data=df, palette="Oranges")
plt.title("Frecuencia de Consumo de Verduras (FCVC) vs Nivel de Obesidad")
plt.ylabel("Frecuencia de Verduras")
plt.xlabel("Nivel de Obesidad")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
