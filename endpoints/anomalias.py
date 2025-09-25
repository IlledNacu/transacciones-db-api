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

    # Identificamos sospechosos
    sospechosos = df_features[
        (df_features["outlier_iso_forest"] == -1) | (df_features["outlier_lof"] == -1)
    ]
    resultados = []
    for _, row in sospechosos.iterrows():
        cliente = db.query(models.Cliente).filter(models.Cliente.id == row["id_cliente"]).first()
        motivos = []
        if row["monto_maximo"] > row["monto_promedio"] * 5:
            motivos.append("Monto muy alto comparado con su promedio")
        if row["conteo_transacciones"] > df_features["conteo_transacciones"].mean() * 3:
            motivos.append("Frecuencia inusual de transacciones")
        if row["tiempo_entre_transacciones"] < df_features["tiempo_entre_transacciones"].mean() / 3:
            motivos.append("Transacciones demasiado seguidas")
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
