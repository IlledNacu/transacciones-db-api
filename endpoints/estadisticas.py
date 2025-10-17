from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from endpoints.anomalias_clientes import detectar_clientes_sospechosos
from endpoints.anomalias_transacciones import contar_transacciones_sospechosas
from endpoints.clientes import get_clientes_count
from endpoints.transacciones import get_transacciones_count
import models
import database
from typing import Dict
# Para el modelo de IA
from sqlalchemy import func


router = APIRouter(prefix="/estadisticas", tags=["Anomalías"])
# APIRouter agrupa todos los endpoints bajo el prefijo /anomalias

# ---- DASHBOARD ----

@router.get("/", response_model=Dict[str, float])
def get_stats(db: Session = Depends(database.get_db)):
    total_transacciones = get_transacciones_count(db)
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

    porcentaje_clientes_sospechosos = (len(clientes_sospechosos) / total_clientes) * 100

    porcentaje_transacciones_sospechosas = (contar_transacciones_sospechosas(db)/total_transacciones) * 100

    return {
        "total_clientes": total_clientes,
        "porcentaje_clientes_sospechosos": round(porcentaje_clientes_sospechosos, 2),
        "total_transacciones": total_transacciones,
        "promedio_transacciones_por_minuto": round(promedio_transacciones_por_minuto, 2),
        # "total_transacciones_sospechosas": contar_transacciones_sospechosas(db)
        "porcentaje_transacciones_sospechosas": round(porcentaje_transacciones_sospechosas, 2)
        
    }