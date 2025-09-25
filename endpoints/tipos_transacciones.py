from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import models
import req_res_models
import database
from typing import List

router = APIRouter(prefix="/tipos_transacciones", tags=["Tipos de transacciones"])

@router.get("/{id}", response_model=req_res_models.TipoTransaccionResponse)
def get_tipo_transaccion(id: int, db: Session = Depends(database.get_db)):
    tipo_transaccion = db.query(models.TipoTransaccion).filter(models.TipoTransaccion.id == id).first()
    if not tipo_transaccion:
        raise HTTPException(status_code=404, detail="Tipo de transacción no encontrado.")
    return tipo_transaccion

@router.post("/", response_model=req_res_models.TipoTransaccionResponse)
def create_tipo_transaccion(tipo_transaccion: req_res_models.TipoTransaccionCreate, db: Session = Depends(database.get_db)):
    if db.query(models.TipoTransaccion).filter(models.TipoTransaccion.nombre == tipo_transaccion.nombre).first():
        raise HTTPException(status_code=400, detail="Tipo de transacción ya registrado.")
    nuevo_tipo = models.TipoTransaccion(**tipo_transaccion.model_dump())
    db.add(nuevo_tipo)
    db.commit()
    db.refresh(nuevo_tipo)
    return nuevo_tipo

@router.put("/{id}", response_model=req_res_models.TipoTransaccionResponse)
def update_tipo_transaccion(id: int, tipo_transaccion: req_res_models.TipoTransaccionCreate, db: Session = Depends(database.get_db)):
    db_tipo_transaccion = db.query(models.TipoTransaccion).filter(models.TipoTransaccion.id == id).first()
    if not db_tipo_transaccion:
        raise HTTPException(status_code=404, detail="Este tipo de transacción no existe.")
    for field, value in tipo_transaccion.model_dump().items():
        setattr(db_tipo_transaccion, field, value)
    db.commit()
    db.refresh(db_tipo_transaccion)
    return db_tipo_transaccion

@router.delete("/{id}")
def delete_tipo_transaccion(id:int, db:Session = Depends(database.get_db)):
    db_tipo_transaccion = db.query(models.TipoTransaccion).filter(models.TipoTransaccion.id == id).first()
    if not db_tipo_transaccion:
        raise HTTPException(status_code=404, detail="Este tipo de transacción no existe.")
    db.delete(db_tipo_transaccion)
    db.commit()
    return {"message":"Número de cuenta eliminado."}

@router.get("/", response_model=List[req_res_models.TipoTransaccionResponse])
def get_all_tipo_transaccion(db:Session = Depends(database.get_db)):
    return db.query(models.TipoTransaccion).all()

@router.get("/count")
def get_tipos_transacciones_count(db: Session = Depends(database.get_db)):
    return {"total": db.query(models.TipoTransaccion).count()}