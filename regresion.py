import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Crear un DataFrame con los datos de la tabla obtenidos de esta pagina https://www.ine.gob.gt/estadisticas-de-migracion/
data = {
    'Año': [2019, 2020, 2021, 2022, 2023],
    'Total': [9499125, 3028706, 4882989, 8579843, 10754026],
    'Mujer': [4125208, 1010278, 1734883, 3532883, 4676500],
    'Hombre': [5373917, 2018428, 3148106, 5046960, 6077526]
}

df = pd.DataFrame(data)

# Valores para crear la regresion
X = df[['Año']] 
y = df['Total']

# Crear el modelo de regresión lineal
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

# coeficientes de la regresión
print(f'Coeficiente: {model.coef_[0]}')
print(f'Intercepto: {model.intercept_}')