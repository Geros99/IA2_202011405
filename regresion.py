
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# URL del archivo CSV en GitHub
url = "https://raw.githubusercontent.com/Geros99/IA2_202011405/refs/heads/main/datos.csv"

# Leer el archivo CSV desde la URL
df = pd.read_csv(url, encoding='ISO-8859-1', delimiter=';')



# Supongamos que el CSV tiene columnas 'Año' y 'Total'
X = df[['Año']]  # Variable independiente (año)
y = df['Valor']  # Variable dependiente (total de migrantes)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X, y)

# Predecir los valores
y_pred = model.predict(X)

# Graficar los resultados
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X, y_pred, color='red', label='Línea de regresión')
plt.xlabel('Año')
plt.ylabel('Total de Migrantes')
plt.title('Regresión Lineal del Flujo Migratorio')
plt.legend()
plt.show()

# Mostrar los coeficientes de la regresión
print(f'Coeficiente: {model.coef_[0]}')
print(f'Intercepto: {model.intercept_}')