import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
from sklearn.metrics import r2_score

# Iniciar el generador de numeros aleatorios
np.random.seed(0)

# Generar datos aleatorios
tamanos = np.random.normal(100, 25, 10000) # Genera 10000 registros con tamaño medio de 100 y desviacion estandar de 25
precios = 50000 + (tamanos * 1000) + np.random.normal(0, 10000, 10000)

# Crear dataframe
datos = pd.DataFrame({'tamano': tamanos, 'precio':precios})

# Asegurar datos positivos
datos = datos[datos['tamano'] > 0]
datos = datos[datos['precio'] > 0]

# Guardar a csv
datos.to_csv('datos.csv',index=False)

# Cargar csv
datos = pd.read_csv('datos.csv')

# Visualizar los 5 primeros registros
print(datos.head(5))

# Histogramas
datos.hist(figsize=(10,5)) # Esto crea una figura con el tamaño especificado
plt.suptitle('Histogramas') # Título general

# Gráfica de dispersión
plt.figure()
plt.scatter(datos['tamano'], datos['precio'])
plt.xlabel('Tamaño')
plt.ylabel('Precio')
plt.title('Dispersión de precio vs tamaño')

# Mostrar todas las figuras a la vez
plt.show()

# Dividir los datos
X = datos[['tamano']]
y = datos[['precio']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
modelo = lm.LinearRegression()
modelo.fit(X_train, y_train)

# Predecir precios y evaluar el modelo
y_pred = modelo.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'Coeficiente de determinación r2: {r2}')

# Visualizar la regresión
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Tamaño')
plt.ylabel('Precios')
plt.title('Regresión lineal simple')
plt.show()
