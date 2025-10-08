from fastapi import APIRouter, Depends, HTTPException
import numpy as np
from sqlalchemy.orm import Session
from endpoints.clientes import get_clientes_count
import models
import req_res_models
import database
from typing import List, Dict
# Para el modelo de IA
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy import func
from datetime import datetime
# Para gráficos de Plotly
import plotly.express as px
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/anomalias", tags=["Anomalías"])
# APIRouter agrupa todos los endpoints bajo el prefijo /anomalias

# Función para obtener las transacciones de manera paginada debido al volumen de datos
def get_transacciones_paginated(db: Session, skip: int = 0, limit: int = 10000):
    return db.query(models.Transaccion).offset(skip).limit(limit).all()

# ---- DETECCIÓN DE ANOMALÍAS CON IA  ---- 

# Endpoint para obtener a los clientes considerados sospechosos por mínimo 2 modelos de IA y los motivos
@router.get("/clientes_sospechosos", response_model=List[req_res_models.ClienteSospechosoResponse])
def detectar_clientes_sospechosos(db: Session = Depends(database.get_db)): # Recibe una sesión de la bd que se inyecta automáticamente con Depends(database.get_db)
    transacciones = db.query(models.Transaccion).all() # Obtiene todas las transacciones
    if not transacciones:
        raise HTTPException(status_code=404, detail="No hay transacciones registradas")
    data = [
        {
            "id_cliente": t.id_cliente,
            "monto": float(t.monto),
            "fecha_hora": t.fecha_hora
        }
        for t in transacciones
    ] # y las convierte en una lista de diccionarios con los campos clave
    df = pd.DataFrame(data) # Convierte la lista en un dataframe

    # Feature engineering
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"]) # Asegura el formato correcto de la variable
    df_features = df.groupby("id_cliente").agg( # Agrupa por cliente y calcula estadísticas
        conteo_transacciones=("monto", "count"),
        monto_promedio=("monto", "mean"),
        monto_std=("monto", "std"),
        monto_maximo=("monto", "max"),
        monto_minimo=("monto", "min"),
        tiempo_entre_transacciones=("fecha_hora", lambda x: x.diff().dt.total_seconds().mean())
    ).reset_index()
    df_features.fillna(0, inplace=True) # Si algún cálculo da NaN, lo reemplaza por 0

    # Seleccionamos los features (variables de entrada (X) que usarán los modelos de IA)
    features = [
        "conteo_transacciones",
        "monto_promedio",
        "monto_std",
        "monto_maximo",
        "monto_minimo",
        "tiempo_entre_transacciones"
    ]
    X = df_features[features]

    # Modelos de detección
    
    iso_forest = IsolationForest(contamination=0.05, random_state=42) # Isolation Forest (modelo no supervisado de anomalías): Marca cada cliente como 1 (normal) o -1 (sospechoso)
    df_features["outlier_iso_forest"] = iso_forest.fit_predict(X)

    lof = LocalOutlierFactor(n_neighbors=20, contamination="auto") # LOF detecta anomalías en función de la densidad de vecinos cercanos
    df_features["outlier_lof"] = lof.fit_predict(X)

    # K-Means Clustering
    scaler = StandardScaler() # Escala las variables (K-Means es sensible a la escala)
    X_scaled = scaler.fit_transform(X)
    n_clusters = min(max(2, len(df_features) // 10), 10) # Calcula clusters de clientes (determinamos número óptimo de clusters: mínimo 2, máximo 10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Asigna un cluster a cada cliente
    df_features["cluster"] = kmeans.fit_predict(X_scaled)
    df_features["distancia_centroide"] = np.min( # Calcula la distancia de cada cliente a su centroide de cluster
        np.linalg.norm(X_scaled[:, np.newaxis] - kmeans.cluster_centers_, axis=2), 
        axis=1
    )
    threshold = np.percentile(df_features["distancia_centroide"], 95) # Marcamos como outliers los que están en el percentil 95 de distancia
    df_features["outlier_kmeans"] = df_features["distancia_centroide"].apply(
        lambda x: -1 if x > threshold else 1 # Son sospechosos (-1) los clientes que están en el 5% más alejado del centroide
    )

    # Sumamos cuántos modelos marcaron a cada cliente como sospechoso
    df_features["votos_sospechoso"] = (
        (df_features["outlier_iso_forest"] == -1).astype(int) +
        (df_features["outlier_lof"] == -1).astype(int) +
        (df_features["outlier_kmeans"] == -1).astype(int)
    )

    # Identificamos sospechosos: detectado por 1 modelo mínimo
    # sospechosos = df_features[
    #     (df_features["outlier_iso_forest"] == -1) | (df_features["outlier_lof"] == -1) | (df_features["outlier_kmeans"] == -1)
    # ]

    # Identificamos sospechosos: mínimo 2 votos (detectado por 2 modelos)
    sospechosos = df_features[df_features["votos_sospechoso"] >= 2]

    # Recorremos cada cliente sospechoso, buscamos su información en la tabla y preparamos una lista de motivos de sospecha
    resultados = []
    for _, row in sospechosos.iterrows():
        cliente = db.query(models.Cliente).filter(models.Cliente.id == row["id_cliente"]).first()
        motivos = []

        # Motivo de sospecha 1: que algún modelo lo haya detectado
        if row["outlier_iso_forest"] == -1:
            motivos.append("Detectado por Isolation Forest (patrón anómalo)")
        if row["outlier_lof"] == -1:
            motivos.append("Detectado por LOF (outlier local)")
        if row["outlier_kmeans"] == -1:
            motivos.append("Detectado por K-Means (alejado de clusters normales)")
            
        # Motivos de sospecha 2: criterios del banco
        if row["monto_maximo"] > row["monto_promedio"] * 5:
            motivos.append("Monto muy alto comparado con su promedio")
        if row["conteo_transacciones"] > df_features["conteo_transacciones"].mean() * 3:
            motivos.append("Frecuencia inusual de transacciones")
        if row["tiempo_entre_transacciones"] < df_features["tiempo_entre_transacciones"].mean() / 3:
            motivos.append("Transacciones demasiado seguidas")
        if row["distancia_centroide"] > threshold:
            motivos.append("Comportamiento alejado de patrones normales (clustering)")
        if not motivos:  # Si ninguno de los motivos anteriores aplica, deja un fallback genérico:
            motivos.append("Comportamiento atípico indefinido")

        # Creamos el DTO de respuesta
        resultados.append(req_res_models.ClienteSospechosoResponse(
            id_cliente=row["id_cliente"],
            nombre=cliente.nombre if cliente else "Desconocido",
            apellido=cliente.apellido if cliente else "Desconocido",
            sospechoso_por=motivos
        ))

    return resultados

# Detección de transacciones individuales sospechosas sin agrupar por cliente
@router.get("/transacciones_sospechosas", response_model=List[req_res_models.TransaccionSospechosaResponse])
def detectar_transacciones_sospechosas(skip: int = 0, limit: int = 10000, db: Session = Depends(database.get_db)): # Recibe los parámetros skip y limit para la paginación de los datos
    transacciones = db.query(models.Transaccion).offset(skip).limit(limit).all() # Obtiene todas las transacciones de la BD que entren entre los parámetros skip y limit
    if not transacciones:
        raise HTTPException(status_code=404, detail="No hay transacciones registradas")

    data = [
        {
            "id": t.id,  # <- id de la transacción (string según tu DTO)
            "id_cliente": t.id_cliente,
            "id_cajero": t.id_cajero,
            "id_tipo_transaccion": t.id_tipo_transaccion,
            "monto": float(t.monto),
            "fecha_hora": t.fecha_hora
        }
        for t in transacciones
    ]
    df = pd.DataFrame(data) # Convierte las transacciones en una lista de diccionarios con los datos relevantes y luego en un df

    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"]) # Formatea la fecha y extrae de ella los datos que necesitamos
    df["hora_del_dia"] = df["fecha_hora"].dt.hour
    df["dia_semana"] = df["fecha_hora"].dt.dayofweek
    df["es_fin_de_semana"] = df["dia_semana"].isin([5, 6]).astype(int)
    df["es_horario_nocturno"] = df["hora_del_dia"].between(0, 6).astype(int)

    monto_promedio_global = df["monto"].mean() # Calcula el promedio
    monto_std_global = df["monto"].std() # y la desviación estándar de todos los montos

    # Calculamos estadísticas por cliente: monto promedio, desviación estándar y número de transacciones
    df["monto_promedio_cliente"] = df.groupby("id_cliente")["monto"].transform("mean")
    df["monto_std_cliente"] = df.groupby("id_cliente")["monto"].transform("std")
    df["transacciones_cliente"] = df.groupby("id_cliente")["id"].transform("count")

    df = df.sort_values(["id_cliente", "fecha_hora"]) # Ordena las transacciones por cliente y fecha
    df["tiempo_desde_ultima"] = df.groupby("id_cliente")["fecha_hora"].diff().dt.total_seconds() # Calcula el tiempo (en segundos) desde la última transacción del cliente
    df["tiempo_desde_ultima"] = df["tiempo_desde_ultima"].fillna(0) # Rellena NaN con 0

    # Definimos los features a usar para detección de anomalías
    features = ["monto", "hora_del_dia", "es_fin_de_semana", "es_horario_nocturno", "tiempo_desde_ultima", "transacciones_cliente"]
    X = df[features].fillna(0)

    # Entrenamos el modelo Isolation Forest para detectar anomalías globales
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df["outlier_iso_forest"] = iso_forest.fit_predict(X) # outlier_iso_forest: -1 si es anómalo, 1 si es normal
    df["score_iso_forest"] = iso_forest.score_samples(X) # score_iso_forest: puntaje de anormalidad

    # Entrenamos el modelo Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1) # Detecta anomalías locales comparando vecinos cercanos
    df["outlier_lof"] = lof.fit_predict(X)
    df["score_lof"] = lof.negative_outlier_factor_ # Guarda etiquetas y score

    # Normalizamos los scores de ambos modelos entre 0 y 1 y combinamos ambos scores en un score_anomalia (ponderado 50% y 50%)
    df["score_anomalia"] = (
        (df["score_iso_forest"] - df["score_iso_forest"].min()) / (df["score_iso_forest"].max() - df["score_iso_forest"].min()) * 0.5 +
        (df["score_lof"] - df["score_lof"].min()) / (df["score_lof"].max() - df["score_lof"].min()) * 0.5
    )

    # Filtramos las transacciones que algún modelo marcó como sospechosas
    sospechosas = df[(df["outlier_iso_forest"] == -1) | (df["outlier_lof"] == -1)]

    # Creamos la lista de resultados y buscamos en la BD los datos del cliente asociado a cada transacción sospechosa
    resultados = []
    for _, row in sospechosas.iterrows():
        cliente = db.query(models.Cliente).filter(models.Cliente.id == row["id_cliente"]).first()

        # Registramos motivos de sospecha según los modelos de detección
        motivos = []
        if row["outlier_iso_forest"] == -1:
            motivos.append("Detectado por Isolation Forest (patrón anómalo global)")
        if row["outlier_lof"] == -1:
            motivos.append("Detectado por LOF (outlier local)")
        if row["monto"] > monto_promedio_global + 3 * monto_std_global:
            motivos.append(f"Monto excesivamente alto (${row['monto']:.2f})")
        if row["monto_std_cliente"] > 0:
            z_score = (row["monto"] - row["monto_promedio_cliente"]) / row["monto_std_cliente"]
            if abs(z_score) > 3:
                motivos.append(f"Monto inusual para este cliente (Z-score: {z_score:.2f})")
        if row["es_horario_nocturno"] == 1 and row["monto"] > monto_promedio_global:
            motivos.append("Transacción de alto monto en horario nocturno")
        if row["tiempo_desde_ultima"] > 0 and row["tiempo_desde_ultima"] < 60:
            motivos.append(f"Transacción muy cercana a la anterior ({row['tiempo_desde_ultima']:.0f} segs)")
        if row["es_fin_de_semana"] == 1 and row["monto"] > monto_promedio_global * 2:
            motivos.append("Transacción de alto monto en fin de semana")
        if row["monto"] % 1000 == 0 and row["monto"] >= 1000:
            motivos.append(f"Monto redondo sospechoso (${row['monto']:.2f})")

        if not motivos:
            motivos.append("Patrón atípico detectado") # Si ningún criterio coincidió, pone un motivo genérico

        # Construimos el DTO de respuesta con la transacción y sus motivos de sospecha
        resultados.append(req_res_models.TransaccionSospechosaResponse(
            id=row["id"],
            id_cliente=row["id_cliente"],
            id_cajero=row["id_cajero"],
            id_tipo_transaccion=row["id_tipo_transaccion"],
            monto=row["monto"],
            fecha_hora=row["fecha_hora"],
            nombre=cliente.nombre if cliente else "Desconocido",
            apellido=cliente.apellido if cliente else "Desconocido",
            sospechosa_por=motivos,
            score_anomalia=float(row["score_anomalia"])
        ))

    resultados.sort(key=lambda x: x.score_anomalia, reverse=False) # Ordena las transacciones sospechosas por score de anomalía (más raro primero)
    return resultados

# Agregar a este último misma lógica que a clientes sospechosos: que sea detectado por al menos 2 algoritmos; y agregar entonces también el de clustering?

# ---- GRÁFICOS ---- 

@router.get("/graficos/clientes_sospechosos", response_class=HTMLResponse) # La respuesta será de tipo HTML, ya que se devuelve un gráfico interactivo en formato web
def grafico_clientes_sospechosos(db: Session = Depends(database.get_db)):
    transacciones = db.query(models.Transaccion).all() # Consulta todas las transacciones registradas en la base de datos
    if not transacciones:
        return "<h3>No hay transacciones</h3>"
    
    data = [
        {"id_cliente": t.id_cliente, "monto": float(t.monto), "fecha_hora": t.fecha_hora}
        for t in transacciones
    ]
    df = pd.DataFrame(data) # Convierte las transacciones en una lista de diccionarios con tres campos clave y luego en un df

    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"]) # Formatea la variable

    df_features = df.groupby("id_cliente").agg( # Agrupa las transacciones por cliente y calcula por cada uno 2 features
        monto_promedio=("monto", "mean"),
        tiempo_entre_transacciones=("fecha_hora", lambda x: x.diff().dt.total_seconds().mean())
    ).reset_index() # Reinicia el índice para obtener un DataFrame plano
    df_features.fillna(0, inplace=True) # Rellena los valores nulos con 0

    # Inicializamos el modelo de Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42) # contamination=0.05 significa que aproximadamente el 5% de los clientes se marcarán como sospechosos

    # Entrenamos el modelo usando las features monto_promedio y tiempo_entre_transacciones
    df_features["sospechoso"] = iso_forest.fit_predict(df_features[["monto_promedio", "tiempo_entre_transacciones"]]) # Crea una nueva columna sospechoso: 1 → cliente normal, -1 → cliente sospechoso

    # Gráfico interactivo que representa a cada cliente como un punto
    fig = px.scatter(
        df_features,
        x="monto_promedio",
        y="tiempo_entre_transacciones",
        color=df_features["sospechoso"].map({1: "Normal", -1: "Sospechoso"}),
        hover_data=["id_cliente"]
    )

    return fig.to_html(full_html=True) # Devuelve el gráfico como un HTML completo

@router.get("/graficos/transacciones_sospechosas", response_class=HTMLResponse)
def grafico_transacciones_sospechosas(skip: int = 0, limit: int = 10000, db: Session = Depends(database.get_db)):
    transacciones = db.query(models.Transaccion).offset(skip).limit(limit).all() # Se consultan las transacciones en la bd aplicando paginación (skip y limit)
    if not transacciones:
        return "<h3>No hay transacciones</h3>"

    data = [
        {
            "id": t.id,
            "id_cliente": t.id_cliente,
            "id_cajero": t.id_cajero,
            "id_tipo_transaccion": t.id_tipo_transaccion,
            "monto": float(t.monto),
            "fecha_hora": t.fecha_hora
        }
        for t in transacciones
    ]
    df = pd.DataFrame(data) # Se transforma la lista de transacciones en una lista de diccionarios con atributos clave y luego en df

    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"]) # Formatea la variable
    df["hora"] = df["fecha_hora"].dt.hour # y extrae la hora

    # Seleccionamos las características relevantes: el monto y la hora en que ocurrió la transacción
    features = df[["monto", "hora"]].copy()

    # Inicializamos el modelo Isolation Forest, con un 10% de contaminación esperada
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df["sospechoso"] = iso_forest.fit_predict(features)
    df["categoria"] = df["sospechoso"].map({1: "Normal", -1: "Sospechoso"}) # Se entrena el modelo con las características seleccionadas y se agrega una columna "sospechoso" con valores: 1 → transacción normal, -1 → transacción sospechosa

    # Gráfico de dispersión interactivo
    fig = px.scatter(
        df,
        x="fecha_hora",
        y="monto",
        color="categoria",
        hover_data=["id", "id_cliente", "id_tipo_transaccion", "hora"],
        title="Transacciones Individuales - Detección de Anomalías",
        color_discrete_map={"Normal": "blue", "Sospechoso": "red"}
    )

    # Configuraciones de diseño del gráfico
    fig.update_layout(
        xaxis_title="Fecha y Hora",
        yaxis_title="Monto ($)",
        hovermode="closest"
    )

    return fig.to_html(full_html=True)

# ---- DASHBOARD ----

@router.get("/estadisticas", response_model=Dict[str, float])
def get_stats(db: Session = Depends(database.get_db)):
     # Ejecutamos una consulta SQL COUNT(*) sobre la tabla Transaccion y guardamos el total de filas; luego con Clientes
    total_transacciones = db.query(models.Transaccion).count()
    total_clientes = get_clientes_count(db)
    if total_transacciones == 0 or total_clientes == 0:
        raise HTTPException(status_code=404, detail="No hay datos suficientes para calcular estadísticas")

    # Obtenemos el timestamp más antiguo y el más reciente
    min_fecha = db.query(func.min(models.Transaccion.fecha_hora)).scalar() # Ejecuta una consulta SQL SELECT MIN(fecha_hora) FROM transaccion y devuelve el valor (scalar())
    max_fecha = db.query(func.max(models.Transaccion.fecha_hora)).scalar() # Ejecuta una consulta SQL SELECT MAX(fecha_hora) FROM transaccion y devuelve el valor
    if not min_fecha or not max_fecha or min_fecha == max_fecha:
        raise HTTPException(status_code=400, detail="No hay suficiente rango temporal para calcular promedio")
    
    # Calculamos la diferencia en minutos y luego el promedio de transacciones por minuto en ese periodo
    minutos = (max_fecha - min_fecha).total_seconds() / 60
    promedio_transacciones_por_minuto = total_transacciones / minutos

    # Reutilizamos la detección de clientes sospechosos
    clientes_sospechosos = detectar_clientes_sospechosos(db)

    # REVISAR QUÉ Y CÓMO ESTÁ CONTABILIZANDO AL CALCULAR ESTAS TRANSACCIONES SOSPECHOSAS --> Se redujeron
    # Total de transacciones sospechosas (las de esos clientes)
    ids_sospechosos = [c.id_cliente for c in clientes_sospechosos]
    total_transacciones_sospechosas = db.query(models.Transaccion).filter(
        models.Transaccion.id_cliente.in_(ids_sospechosos)
    ).count()

    porcentaje_clientes_sospechosos = (len(clientes_sospechosos) / total_clientes) * 100

    return {
        "promedio_transacciones_por_minuto": round(promedio_transacciones_por_minuto, 2),
        "total_clientes": total_clientes,
        "total_transacciones_sospechosas": total_transacciones_sospechosas,
        "porcentaje_clientes_sospechosos": round(porcentaje_clientes_sospechosos, 2)
    }