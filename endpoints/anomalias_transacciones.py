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

# ---- FUNCIONES PARA LA DETECCIÓN DE ANOMALÍAS EN TRANSACCIONES  ----

# Detectamos anomalías con IA
def detectar_anomalias_transacciones(transacciones: List[models.Transaccion]):
    if not transacciones:
        raise HTTPException(status_code=404, detail="No hay transacciones registradas")

    # Preprocesamiento: Convierte las transacciones en una lista de diccionarios con los datos relevantes y luego en un df
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
    df = pd.DataFrame(data)

    # Formatea la fecha y extrae de ella los datos que necesitamos
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    df["hora_del_dia"] = df["fecha_hora"].dt.hour
    df["dia_semana"] = df["fecha_hora"].dt.dayofweek
    df["es_fin_de_semana"] = df["dia_semana"].isin([5, 6]).astype(int)
    df["es_horario_nocturno"] = df["hora_del_dia"].between(0, 6).astype(int)

    # Estadísticas globales y por cliente
    monto_promedio_global = df["monto"].mean()
    monto_std_global = df["monto"].std()
    df["monto_promedio_cliente"] = df.groupby("id_cliente")["monto"].transform("mean")
    df["monto_std_cliente"] = df.groupby("id_cliente")["monto"].transform("std")
    df["transacciones_cliente"] = df.groupby("id_cliente")["id"].transform("count")

    # Ordena por cliente y fecha
    df = df.sort_values(["id_cliente", "fecha_hora"])

    # Calcula el tiempo (en segundos) desde la última transacción del cliente y rellena NaN con 0
    df["tiempo_desde_ultima"] = df.groupby("id_cliente")["fecha_hora"].diff().dt.total_seconds().fillna(0)

    # Features y modelos a usar para detección de anomalías
    features = ["monto", "hora_del_dia", "es_fin_de_semana", "es_horario_nocturno", "tiempo_desde_ultima", "transacciones_cliente"]
    X = df[features].fillna(0)
    X_scaled = StandardScaler().fit_transform(X) # Escalamos para KMeans y LOF

    # Entrenamos el modelo Isolation Forest para detectar anomalías globales
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df["outlier_iso_forest"] = iso_forest.fit_predict(X_scaled) # outlier_iso_forest: -1 si es anómalo, 1 si es normal
    df["score_iso_forest"] = iso_forest.score_samples(X_scaled) # score_iso_forest: puntaje de anormalidad

    # Entrenamos el modelo Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1) # Detecta anomalías locales comparando vecinos cercanos
    df["outlier_lof"] = lof.fit_predict(X_scaled)
    df["score_lof"] = lof.negative_outlier_factor_ # Guarda etiquetas y score

    # Usamos KMeans para detectar puntos alejados de sus centroides
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    distances = np.linalg.norm(X_scaled - kmeans.cluster_centers_[df["cluster"]], axis=1)
    threshold = np.percentile(distances, 90)  # El 10% más alejado se considera anómalo
    df["outlier_kmeans"] = np.where(distances > threshold, -1, 1)
    df["score_kmeans"] = -distances  # Negativo para mantener la lógica de "score bajo = anómalo"

    # Normalización y score combinado
    def normalize(col):
        return (col - col.min()) / (col.max() - col.min()) if col.max() != col.min() else 0

    df["score_anomalia"] = (
        normalize(df["score_iso_forest"]) * 0.33 +
        normalize(df["score_lof"]) * 0.33 +
        normalize(df["score_kmeans"]) * 0.34
    )

    # Escalamos a 0-100 para que el umbral sea 50
    df["score_anomalia_100"] = df["score_anomalia"] * 100

    # Detección basada en consenso (al menos 2 de 3 modelos detectan anomalías)
    df["num_modelos_anomalos"] = (
        (df["outlier_iso_forest"] == -1).astype(int) +
        (df["outlier_lof"] == -1).astype(int) +
        (df["outlier_kmeans"] == -1).astype(int)
    )
    # Se considera sospechosa si al menos 2 modelos la detectan y el score >= 50
    sospechosas = df[(df["num_modelos_anomalos"] >= 2) & (df["score_anomalia_100"] >= 50)]

    return df, sospechosas, monto_promedio_global, monto_std_global

# Tomamos el total o una porción
def obtener_transacciones_y_anomalias(db: Session, skip: int = 0, limit: int = None):
    # Obtiene todas las transacciones (o una porción si se pasa limit), detecta anomalías y devuelve el dataframe completo y el filtrado de sospechosas
    query = db.query(models.Transaccion)

    # Si limit no se especifica, no aplicar paginación
    if limit is not None:
        query = query.offset(skip).limit(limit)

    transacciones = query.all()

    df, sospechosas, monto_promedio_global, monto_std_global = detectar_anomalias_transacciones(transacciones)

    return df, sospechosas, monto_promedio_global, monto_std_global

# ---- RUTAS PARA LA DETECCIÓN DE ANOMALÍAS EN TRANSACCIONES  ----

@router.get("/transacciones_sospechosas", response_model=List[req_res_models.TransaccionSospechosaResponse])
def detectar_transacciones_sospechosas(skip: int = 0, limit: int = 5000, db: Session = Depends(database.get_db)):
    df, sospechosas, monto_promedio_global, monto_std_global = obtener_transacciones_y_anomalias(db, skip, limit)

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
            motivos.append("Patrón atípico detectado")

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

@router.get("/cantidad_transacciones_sospechosas", response_model=int)
def contar_transacciones_sospechosas(db: Session = Depends(database.get_db)):
    _, sospechosas, _, _ = obtener_transacciones_y_anomalias(db, limit=None)
    return len(sospechosas)

# ---- GRÁFICOS PARA LA DETECCIÓN DE ANOMALÍAS EN TRANSACCIONES  ----

@router.get("/graficos/transacciones_heatmap", response_class=HTMLResponse)
def grafico_heatmap_transacciones(skip: int = 0, limit: int = 5000, db: Session = Depends(database.get_db)):
    # Obtenemos transacciones y detectamos anomalías
    _, sospechosas, _, _ = obtener_transacciones_y_anomalias(db, skip, limit)

    if sospechosas.empty:
        return "<h3>No hay transacciones sospechosas</h3>"

    # Extraemos hora del día y día de la semana para el heatmap
    sospechosas["hora"] = sospechosas["fecha_hora"].dt.hour
    sospechosas["dia_semana"] = sospechosas["fecha_hora"].dt.dayofweek  # 0=Lunes, 6=Domingo

    # Contamos cantidad de anomalías por hora y día
    heatmap_data = sospechosas.groupby(["dia_semana", "hora"]).size().reset_index(name="cantidad_sospechosas")

    # Convertimos día a nombre para el eje x
    dias = {0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"}
    heatmap_data["dia_semana_str"] = heatmap_data["dia_semana"].map(dias)

    # Gráfico de calor interactivo
    fig = px.density_heatmap(
        heatmap_data,
        x="dia_semana_str",
        y="hora",
        z="cantidad_sospechosas",
        color_continuous_scale="Reds",
        title="Concentración de Transacciones Sospechosas por Día y Hora",
        labels={"hora":"Hora del día", "dia_semana_str":"Día de la semana", "cantidad_sospechosas":"Cantidad sospechosa"}
    )

    fig.update_layout(
        yaxis=dict(dtick=1),
        xaxis=dict(categoryorder="array", categoryarray=["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"])
    )

    return fig.to_html(full_html=True)
