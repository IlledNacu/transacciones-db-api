from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import models
import req_res_models
import database
from typing import List

router = APIRouter(prefix="/transacciones", tags=["Transacciones"])

@router.get("/{id}", response_model=req_res_models.TransaccionResponse)
def get_transaccion(id:str, db:Session = Depends(database.get_db)):
    transaccion = db.query(models.Transaccion).filter(models.Transaccion.id == id).first()
    if not transaccion:
        raise HTTPException(status_code=404, detail="Transacción no encontrada.")
    return transaccion

@router.post("/", response_model=req_res_models.TransaccionResponse)
def create_transaccion(transaccion:req_res_models.TransaccionCreate, db:Session = Depends(database.get_db)):
    nueva_transaccion = models.Transaccion(**transaccion.model_dump())
    db.add(nueva_transaccion)
    db.commit()
    db.refresh(nueva_transaccion)
    return nueva_transaccion

@router.put("/{id}", response_model=req_res_models.TransaccionResponse)
def update_transaccion(id:str, transaccion:req_res_models.TransaccionCreate, db:Session = Depends(database.get_db)):
    db_transaccion = db.query(models.Transaccion).filter(models.Transaccion.id == id).first()
    if not db_transaccion:
        raise HTTPException(status_code=404, detail="Esta transacción no existe.")
    for field, value in transaccion.model_dump().items():
        setattr(db_transaccion, field, value)
    db.commit()
    db.refresh(db_transaccion)
    return db_transaccion

@router.delete("/{id}")
def delete_transaccion(id:str, db:Session = Depends(database.get_db)):
    db_transaccion = db.query(models.Transaccion).filter(models.Transaccion.id == id).first()
    if not db_transaccion:
        raise HTTPException(status_code=404, detail="Esta transacción no existe.")
    db.delete(db_transaccion)
    db.commit()
    return {"message":"Transacción eliminada."}

@router.get("/", response_model=List[req_res_models.TransaccionResponse])
def get_all_transaccion(db:Session = Depends(database.get_db)):
    return db.query(models.Transaccion).all()

# Todas las consultas siguen la misma lógica que las de cajeros.py