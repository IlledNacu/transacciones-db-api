from fastapi import APIRouter, Depends, HTTPException
import numpy as np
from sqlalchemy.orm import Session
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

# Detección de clientes sospechosos con IA
@router.get("/clientes_sospechosos", response_model=List[req_res_models.ClienteSospechosoResponse])
def detectar_clientes_sospechosos(db: Session = Depends(database.get_db)):
    # Obtenemos datos de transacciones
    transacciones = db.query(models.Transaccion).all()
    if not transacciones:
        raise HTTPException(status_code=404, detail="No hay transacciones registradas")
    data = [
        {
            "id_cliente": t.id_cliente,
            "monto": float(t.monto),
            "fecha_hora": t.fecha_hora
        }
        for t in transacciones
    ]
    df = pd.DataFrame(data)

    # Feature engineering
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    df_features = df.groupby("id_cliente").agg(
        conteo_transacciones=("monto", "count"),
        monto_promedio=("monto", "mean"),
        monto_std=("monto", "std"),
        monto_maximo=("monto", "max"),
        monto_minimo=("monto", "min"),
        tiempo_entre_transacciones=("fecha_hora", lambda x: x.diff().dt.total_seconds().mean())
    ).reset_index()
    df_features.fillna(0, inplace=True)

    # Seleccionamos features
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
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df_features["outlier_iso_forest"] = iso_forest.fit_predict(X)

    lof = LocalOutlierFactor(n_neighbors=20, contamination="auto")
    df_features["outlier_lof"] = lof.fit_predict(X)

    # K-Means Clustering para detección de anomalías
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determinamos número óptimo de clusters (mínimo 2, máximo 10 o número de clientes)
    n_clusters = min(max(2, len(df_features) // 10), 10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_features["cluster"] = kmeans.fit_predict(X_scaled)
    
    # Calculamos distancia de cada punto a su centroide
    df_features["distancia_centroide"] = np.min(
        np.linalg.norm(X_scaled[:, np.newaxis] - kmeans.cluster_centers_, axis=2), 
        axis=1
    )
    
    # Marcamos como outliers los que están en el percentil 95 de distancia
    threshold = np.percentile(df_features["distancia_centroide"], 95)
    df_features["outlier_kmeans"] = df_features["distancia_centroide"].apply(
        lambda x: -1 if x > threshold else 1
    )

    # Identificamos sospechosos (al menos 2 de 3 métodos deben marcarlo)
    df_features["votos_sospechoso"] = (
        (df_features["outlier_iso_forest"] == -1).astype(int) +
        (df_features["outlier_lof"] == -1).astype(int) +
        (df_features["outlier_kmeans"] == -1).astype(int)
    )

    # Identificamos sospechosos
    sospechosos = df_features[
        (df_features["outlier_iso_forest"] == -1) | (df_features["outlier_lof"] == -1)
    ]
    resultados = []
    for _, row in sospechosos.iterrows():
        cliente = db.query(models.Cliente).filter(models.Cliente.id == row["id_cliente"]).first()
        motivos = []

        #Agrego como motivo de sospecha que algun modelo lo haya detectado
        if row["outlier_iso_forest"] == -1:
            motivos.append("Detectado por Isolation Forest (patrón anómalo)")
        if row["outlier_lof"] == -1:
            motivos.append("Detectado por LOF (outlier local)")
        if row["outlier_kmeans"] == -1:
            motivos.append("Detectado por K-Means (alejado de clusters normales)")
            
        #Motivos de sospecha segun logica del banco
        if row["monto_maximo"] > row["monto_promedio"] * 5:
            motivos.append("Monto muy alto comparado con su promedio")
        if row["conteo_transacciones"] > df_features["conteo_transacciones"].mean() * 3:
            motivos.append("Frecuencia inusual de transacciones")
        if row["tiempo_entre_transacciones"] < df_features["tiempo_entre_transacciones"].mean() / 3:
            motivos.append("Transacciones demasiado seguidas")
        if row["distancia_centroide"] > threshold:
            motivos.append("Comportamiento alejado de patrones normales (clustering)")
        if not motivos:  # fallback si los modelos marcaron sospechoso pero no encontramos regla
            motivos.append("Comportamiento atípico indefinido")
        resultados.append(req_res_models.ClienteSospechosoResponse(
            id_cliente=row["id_cliente"],
            nombre=cliente.nombre if cliente else "Desconocido",
            apellido=cliente.apellido if cliente else "Desconocido",
            sospechoso_por=motivos
        ))

    return resultados

# DASHBOARD

@router.get("/estadisticas", response_model=Dict[str, float])
def get_stats(db: Session = Depends(database.get_db)):
    total_transacciones = db.query(models.Transaccion).count()
    total_clientes = db.query(models.Cliente).count()
    if total_transacciones == 0 or total_clientes == 0:
        raise HTTPException(status_code=404, detail="No hay datos suficientes para calcular estadísticas")

    # Obtenemos el timestamp más antiguo y el más reciente
    min_fecha = db.query(func.min(models.Transaccion.fecha_hora)).scalar()
    max_fecha = db.query(func.max(models.Transaccion.fecha_hora)).scalar()
    if not min_fecha or not max_fecha or min_fecha == max_fecha:
        raise HTTPException(status_code=400, detail="No hay suficiente rango temporal para calcular promedio")
    # Calculamos la diferencia en minutos
    minutos = (max_fecha - min_fecha).total_seconds() / 60
    promedio_transacciones_por_minuto = total_transacciones / minutos

    # Reutilizamos la detección de clientes sospechosos
    clientes_sospechosos = detectar_clientes_sospechosos(db)

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

@router.get("/graficos/transacciones_sospechosas", response_class=HTMLResponse)
def grafico_plotly(db: Session = Depends(database.get_db)):
    # Reutilizamos las features
    transacciones = db.query(models.Transaccion).all()
    if not transacciones:
        return "<h3>No hay transacciones</h3>"
    data = [
        {"id_cliente": t.id_cliente, "monto": float(t.monto), "fecha_hora": t.fecha_hora}
        for t in transacciones
    ]
    df = pd.DataFrame(data)
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    df_features = df.groupby("id_cliente").agg(
        monto_promedio=("monto", "mean"),
        tiempo_entre_transacciones=("fecha_hora", lambda x: x.diff().dt.total_seconds().mean())
    ).reset_index()
    df_features.fillna(0, inplace=True)

    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df_features["sospechoso"] = iso_forest.fit_predict(df_features[["monto_promedio", "tiempo_entre_transacciones"]])

    # Plot interactivo
    fig = px.scatter(
        df_features,
        x="monto_promedio",
        y="tiempo_entre_transacciones",
        color=df_features["sospechoso"].map({1: "Normal", -1: "Sospechoso"}),
        hover_data=["id_cliente"]
    )
    return fig.to_html(full_html=True)

def get_transacciones_paginated(db: Session, skip: int = 0, limit: int = 10000):
    return db.query(models.Transaccion).offset(skip).limit(limit).all()

# Detección de transacciones individuales sospechosas sin agrupar por cliente
@router.get("/transacciones_sospechosas", response_model=List[req_res_models.TransaccionSospechosaResponse])
def detectar_transacciones_sospechosas(skip: int = 0, limit: int = 10000, db: Session = Depends(database.get_db)):
    transacciones = db.query(models.Transaccion).offset(skip).limit(limit).all()
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
    df = pd.DataFrame(data)
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    df["hora_del_dia"] = df["fecha_hora"].dt.hour
    df["dia_semana"] = df["fecha_hora"].dt.dayofweek
    df["es_fin_de_semana"] = df["dia_semana"].isin([5, 6]).astype(int)
    df["es_horario_nocturno"] = df["hora_del_dia"].between(0, 6).astype(int)

    monto_promedio_global = df["monto"].mean()
    monto_std_global = df["monto"].std()

    df["monto_promedio_cliente"] = df.groupby("id_cliente")["monto"].transform("mean")
    df["monto_std_cliente"] = df.groupby("id_cliente")["monto"].transform("std")
    df["transacciones_cliente"] = df.groupby("id_cliente")["id"].transform("count")

    df = df.sort_values(["id_cliente", "fecha_hora"])
    df["tiempo_desde_ultima"] = df.groupby("id_cliente")["fecha_hora"].diff().dt.total_seconds()
    df["tiempo_desde_ultima"] = df["tiempo_desde_ultima"].fillna(0)

    features = ["monto", "hora_del_dia", "es_fin_de_semana", "es_horario_nocturno", "tiempo_desde_ultima", "transacciones_cliente"]
    X = df[features].fillna(0)

    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df["outlier_iso_forest"] = iso_forest.fit_predict(X)
    df["score_iso_forest"] = iso_forest.score_samples(X)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    df["outlier_lof"] = lof.fit_predict(X)
    df["score_lof"] = lof.negative_outlier_factor_

    df["score_anomalia"] = (
        (df["score_iso_forest"] - df["score_iso_forest"].min()) / (df["score_iso_forest"].max() - df["score_iso_forest"].min()) * 0.5 +
        (df["score_lof"] - df["score_lof"].min()) / (df["score_lof"].max() - df["score_lof"].min()) * 0.5
    )

    sospechosas = df[(df["outlier_iso_forest"] == -1) | (df["outlier_lof"] == -1)]

    resultados = []
    for _, row in sospechosas.iterrows():
        cliente = db.query(models.Cliente).filter(models.Cliente.id == row["id_cliente"]).first()

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

    resultados.sort(key=lambda x: x.score_anomalia, reverse=False)
    return resultados


#Gráfico interactivo de transacciones individuales mostrando las sospechosas
@router.get("/graficos/transacciones_individuales", response_class=HTMLResponse)
def grafico_transacciones_individuales(skip: int = 0, limit: int = 10000, db: Session = Depends(database.get_db)):
    transacciones = db.query(models.Transaccion).offset(skip).limit(limit).all()
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
    df = pd.DataFrame(data)
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    df["hora"] = df["fecha_hora"].dt.hour

    features = df[["monto", "hora"]].copy()
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df["sospechoso"] = iso_forest.fit_predict(features)
    df["categoria"] = df["sospechoso"].map({1: "Normal", -1: "Sospechoso"})

    fig = px.scatter(
        df,
        x="fecha_hora",
        y="monto",
        color="categoria",
        hover_data=["id", "id_cliente", "id_tipo_transaccion", "hora"],
        title="Transacciones Individuales - Detección de Anomalías",
        color_discrete_map={"Normal": "blue", "Sospechoso": "red"}
    )

    fig.update_layout(
        xaxis_title="Fecha y Hora",
        yaxis_title="Monto ($)",
        hovermode="closest"
    )

    return fig.to_html(full_html=True)
