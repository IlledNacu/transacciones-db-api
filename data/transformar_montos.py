import pandas as pd
import numpy as np

# Cargar el archivo CSV
df = pd.read_csv('transacciones_modificadas.csv')

# --- Nueva Modificación de la columna 'monto' ---

# 1. Generar un factor aleatorio entre 0.5 y 2.5 para cada transacción
# Esto hará que algunos montos bajen (hasta la mitad) y otros suban (hasta 2.5 veces)
factor_aleatorio = np.random.uniform(0.7, 1.5, size=len(df))

# 2. Multiplicar el monto original por este factor
df['monto'] = df['monto'] * factor_aleatorio

# 3. Redondear los nuevos montos al millar más cercano para que no sean tan "sucios"
df['monto'] = (df['monto'] // 1000) * 1000

# 4. Asegurarse de que el monto mínimo sea de 1000 (por si algún monto original era muy bajo)
df['monto'] = df['monto'].apply(lambda x: max(x, 1000))


# --- Guardar los resultados ---

# Guardar el DataFrame modificado en un nuevo archivo CSV
df.to_csv('transacciones_modificadas.csv', index=False)

print("¡Listo! Se ha creado el archivo 'transacciones_modificadas.csv' con la nueva lógica de montos.")