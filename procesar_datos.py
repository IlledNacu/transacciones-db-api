import pandas as pd
import os

def procesar_y_guardar_datos():
    print("--- ⚙️ Iniciando procesamiento de datos originales ---")

    DIR_BASE = os.path.dirname(os.path.abspath(__file__)) 
    RUTA_ORIGINAL = os.path.join(DIR_BASE, "data_original")
    RUTA_LIMPIO = os.path.join(DIR_BASE, "data")

    # 1. Rutas y archivos (Validaciones)
    if not os.path.exists(RUTA_ORIGINAL):
        print(f"ERROR: No se encontró la carpeta de datos originales en: {RUTA_ORIGINAL}. Abortando.")
        return # Sale de la función si hay error

    # Asegurar que la carpeta 'data' exista para guardar los archivos limpios
    os.makedirs(RUTA_LIMPIO, exist_ok=True)

    # 2. Cargar y concatenar transacciones
    archivos_transacciones = [
        "enugu_transactions.csv",
        "fct_transactions.csv",
        "kano_transactions.csv",
        "lagos_transactions.csv",
        "rivers_transactions.csv"
    ]
    lista_dfs = []

    for archivo in archivos_transacciones:
        try:
            ruta_archivo = os.path.join(RUTA_ORIGINAL, archivo)
            df = pd.read_csv(ruta_archivo)
            
            # Filtrado por fecha
            df["TransactionStartDateTime"] = pd.to_datetime(df["TransactionStartDateTime"])
            # Filtramos entre 2022-01-01 y 2022-01-31
            mask = (df["TransactionStartDateTime"] >= "2022-01-01") & (df["TransactionStartDateTime"] <= "2022-01-31")
            df_filtrado = df.loc[mask]
            lista_dfs.append(df_filtrado)
            
        except FileNotFoundError:
            print(f"Advertencia: Archivo {archivo} no encontrado en {RUTA_ORIGINAL}")
            
    if lista_dfs:
        transacciones = pd.concat(lista_dfs, ignore_index=True)
        print(f"Transacciones cargadas y filtradas: {len(transacciones)} registros.")
    else:
        print("ERROR: No se cargó ninguna transacción. Abortando.")
        return # Sale de la función si hay error

    # 3. Cargar otros archivos y limpiar/renombrar columnas
    try:
        clientes = pd.read_csv(os.path.join(RUTA_ORIGINAL, "customers_lookup.csv"))
        cajeros = pd.read_csv(os.path.join(RUTA_ORIGINAL, "atm_location_lookup.csv"))
        tipo_transaccion = pd.read_csv(os.path.join(RUTA_ORIGINAL, "transaction_type_lookup.csv"))
    except FileNotFoundError as e:
        print(f"ERROR: Archivo de lookup no encontrado: {e}. Abortando.")
        return # Sale de la función si hay error

    # Clientes
    clientes = clientes[['CardholderID', 'First Name', 'Last Name', 'Gender', 'Birth Date', 'Occupation', 'AccountType']]
    clientes = clientes.rename(columns ={
        'CardholderID': 'id',
        'First Name': 'nombre',
        'Last Name': 'apellido',
        'Gender': 'genero',
        'Birth Date': 'fecha_nacimiento',
        'Occupation': 'ocupacion',
        'AccountType': 'tipo_cuenta'
    })

    # Cajeros
    cajeros = cajeros[['LocationID', 'Location Name', 'City', 'State', 'Country']]
    cajeros = cajeros.rename(columns ={
        'LocationID': 'id',
        'Location Name': 'nombre',
        'City': 'ciudad',
        'State': 'provincia',
        'Country': 'pais'
    })

    # Transacciones
    transacciones = transacciones[['TransactionID', 'TransactionStartDateTime', 'CardholderID', 'LocationID', 'TransactionTypeID', 'TransactionAmount']]
    transacciones = transacciones.rename(columns ={
        'TransactionID': 'id',
        'TransactionStartDateTime': 'fecha_hora',
        'CardholderID': 'id_cliente',
        'LocationID': 'id_cajero',
        'TransactionTypeID': 'id_tipo_transaccion',
        'TransactionAmount': 'monto'
    })

    # Tipo Transaccion (Solo renombrar, no seleccionar)
    tipo_transaccion = tipo_transaccion.rename(columns ={
        'TransactionTypeID': 'id',
        'TransactionTypeName': 'nombre'
    })

    # 4. Ajuste de tipos de datos para SQL

    # Convertir datos a tipo numérico
    transacciones['monto'] = transacciones['monto'].astype(float)
    transacciones['id_tipo_transaccion'] = pd.to_numeric(transacciones['id_tipo_transaccion'], errors='coerce')
    tipo_transaccion['id'] = pd.to_numeric(tipo_transaccion['id'], errors='coerce')

    # Convertir datos a tipo datetime
    clientes['fecha_nacimiento'] = pd.to_datetime(clientes['fecha_nacimiento'], errors='coerce')
    transacciones['fecha_hora'] = pd.to_datetime(transacciones['fecha_hora'], errors='coerce')

    # Convertir datos a tipo texto
    for df, cols in [
        (clientes, ['id', 'nombre', 'apellido', 'genero', 'ocupacion', 'tipo_cuenta']),
        (cajeros, ['id', 'nombre', 'ciudad', 'provincia', 'pais']),
        (transacciones, ['id', 'id_cliente', 'id_cajero']),
        (tipo_transaccion, ['nombre'])
    ]:
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

    # 5. Exportar archivos limpios
    clientes.to_csv(os.path.join(RUTA_LIMPIO, 'clientes.csv'), index=False)
    cajeros.to_csv(os.path.join(RUTA_LIMPIO, 'cajeros.csv'), index=False)
    transacciones.to_csv(os.path.join(RUTA_LIMPIO, 'transacciones.csv'), index=False)
    tipo_transaccion.to_csv(os.path.join(RUTA_LIMPIO, 'tipos_transacciones.csv'), index=False)

    print("✅ Proceso de limpieza finalizado. Archivos guardados en la carpeta 'data/'.")

if __name__ == "__main__":
    procesar_y_guardar_datos()