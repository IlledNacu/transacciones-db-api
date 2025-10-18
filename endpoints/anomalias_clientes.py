from fastapi import APIRouter, Depends, HTTPException
import numpy as np
from sqlalchemy.orm import Session
import models
import req_res_models
import database
from typing import List
# Para el modelo de IA
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Para gráficos de Plotly
import plotly.express as px
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/anomalias", tags=["Anomalías"])
# APIRouter agrupa todos los endpoints bajo el prefijo /anomalias

# ---- DETECCIÓN DE ANOMALÍAS EN CLIENTES  ----

# Funcion para evitar valores NaN en diff de tiempo (cuando un cliente tiene una sola transacción)
def safe_time_diff(x):
    if len(x) < 2:
        return 0
    return x.diff().dt.total_seconds().mean()

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
    ] # y las convierte en una lista de diccionarios con los campos 
    df = pd.DataFrame(data) # Convierte la lista en un dataframe

    # Feature engineering
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"]) # Asegura el formato correcto de la variable
    df = df.sort_values(by=["id_cliente", "fecha_hora"])
    
    df_features = df.groupby("id_cliente").agg( # Agrupa por cliente y calcula estadísticas
        conteo_transacciones=("monto", "count"),
        monto_promedio=("monto", "mean"),
        monto_std=("monto", "std"),
        monto_maximo=("monto", "max"),
        monto_minimo=("monto", "min"),
        tiempo_entre_transacciones=("fecha_hora", safe_time_diff)
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
    
    if n_clusters > len(df_features):
        n_clusters = len(df_features)

    if n_clusters < 2:
        df_features["outlier_kmeans"] = 1 # Todos normales si no hay suficientes datos
        df_features["distancia_centroide"] = 0
        threshold = 0
    else:
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
        if row["tiempo_entre_transacciones"] < df_features["tiempo_entre_transacciones"].mean() / 3 and row["tiempo_entre_transacciones"] > 0:
            motivos.append("Transacciones demasiado seguidas")
        if row["distancia_centroide"] > threshold and threshold > 0:
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

# ---- GRÁFICOS ---- 

@router.get("/graficos/clientes_sospechosos", response_class=HTMLResponse) # La respuesta será de tipo HTML, ya que se devuelve un gráfico interactivo en formato web
def grafico_clientes_sospechosos(skip: int = 0, limit: int = 5000, db: Session = Depends(database.get_db)):
    transacciones = db.query(models.Transaccion).offset(skip).limit(limit).all() # Consulta todas las transacciones registradas en la base de datos
    if not transacciones:
        return "<h3>No hay transacciones</h3>"
    
    data = [
        {"id_cliente": t.id_cliente, "monto": float(t.monto), "fecha_hora": t.fecha_hora}
        for t in transacciones
    ]
    df = pd.DataFrame(data) # Convierte las transacciones en una lista de diccionarios con tres campos clave y luego en un df

    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"]) # Formatea la variable
    df = df.sort_values(by=["id_cliente", "fecha_hora"])

    df_features = df.groupby("id_cliente").agg( # Agrupa las transacciones por cliente y calcula por cada uno 2 features
        monto_promedio=("monto", "mean"),
        tiempo_entre_transacciones=("fecha_hora", safe_time_diff)
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