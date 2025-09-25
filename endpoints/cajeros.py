from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import models
import req_res_models
import database
from typing import List

router = APIRouter(prefix="/cajeros", tags=["Cajeros"])

@router.get("/{id}", response_model=req_res_models.CajeroResponse)
def get_cajero(id:int, db:Session = Depends(database.get_db)):
    tipo_cajero = db.query(models.Cajero).filter(models.Cajero.id == id).first()
    if not tipo_cajero:
        raise HTTPException(status_code=404, detail="Cajero no encontrado.")
    return tipo_cajero

@router.post("/", response_model=req_res_models.CajeroResponse)
def create_cajero(cajero:req_res_models.CajeroCreate, db:Session = Depends(database.get_db)):
    nuevo_cajero = models.Cajero(**cajero.model_dump())
    db.add(nuevo_cajero)
    db.commit()
    db.refresh(nuevo_cajero)
    return nuevo_cajero

@router.put("/{id}", response_model=req_res_models.CajeroResponse)
def update_cajero(id:int, cajero:req_res_models.CajeroCreate, db:Session = Depends(database.get_db)):
    db_cajero = db.query(models.Cajero).filter(models.Cajero.id == id).first()
    if not db_cajero:
        raise HTTPException(status_code=404, detail="Este cajero no existe.")
    for field, value in cajero.model_dump().items():
        setattr(db_cajero, field, value)
    db.commit()
    db.refresh(db_cajero)
    return db_cajero

@router.delete("/{id}")
def delete_cajero(id:int, db:Session = Depends(database.get_db)):
    db_cajero = db.query(models.Cajero).filter(models.Cajero.id == id).first()
    if not db_cajero:
        raise HTTPException(status_code=404, detail="Este cajero no existe.")
    db.delete(db_cajero)
    db.commit()
    return {"message":"Cajero eliminado."}

@router.get("/", response_model=List[req_res_models.CajeroResponse])
def get_all_cajero(db:Session = Depends(database.get_db)):
    return db.query(models.Cajero).all()

@router.get("/count")
def get_cajeros_count(db: Session = Depends(database.get_db)):
    return {"total": db.query(models.Cajero).count()}