from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import models
import req_res_models
import database
from typing import List

router = APIRouter(prefix="/cajeros", tags=["Cajeros"])
# APIRouter agrupa todos los endpoints bajo el prefijo /cajeros

@router.get("/{id}", response_model=req_res_models.CajeroResponse)
def get_cajero(id:str, db:Session = Depends(database.get_db)):
    tipo_cajero = db.query(models.Cajero).filter(models.Cajero.id == id).first() # Busca en la tabla cajeros un registro con ese id
    if not tipo_cajero:
        raise HTTPException(status_code=404, detail="Cajero no encontrado.")
    return tipo_cajero # Si existe, lo devuelve como CajeroResponse (FastAPI convierte el objeto SQLAlchemy en un dict gracias a los modelos de req_res de Pydantic)

@router.post("/", response_model=req_res_models.CajeroResponse)
def create_cajero(cajero:req_res_models.CajeroCreate, db:Session = Depends(database.get_db)): # Recibe un JSON con los datos (CajeroCreate)
    nuevo_cajero = models.Cajero(**cajero.model_dump()) # Crea un objeto SQLAlchemy Cajero (convierte el modelo Pydantic en dict)
    db.add(nuevo_cajero)
    db.commit() # Guarda los cambios en la bd
    db.refresh(nuevo_cajero)
    return nuevo_cajero

@router.put("/{id}", response_model=req_res_models.CajeroResponse)
def update_cajero(id:int, cajero:req_res_models.CajeroCreate, db:Session = Depends(database.get_db)):
    db_cajero = db.query(models.Cajero).filter(models.Cajero.id == id).first()
    if not db_cajero:
        raise HTTPException(status_code=404, detail="Este cajero no existe.")
    for field, value in cajero.model_dump().items(): # Si existe el cajero con ese id, recorre los campos enviados en el CajeroCreate y los actualiza con setattr
        setattr(db_cajero, field, value)
    db.commit() # Guarda los cambios en la bd
    db.refresh(db_cajero)
    return db_cajero

@router.delete("/{id}")
def delete_cajero(id:int, db:Session = Depends(database.get_db)):
    db_cajero = db.query(models.Cajero).filter(models.Cajero.id == id).first()
    if not db_cajero:
        raise HTTPException(status_code=404, detail="Este cajero no existe.")
    db.delete(db_cajero) # Si existe el cajero con ese id lo elimina
    db.commit() # Guarda los cambios en la bd
    return {"message":"Cajero eliminado."}

@router.get("/", response_model=List[req_res_models.CajeroResponse])
def get_all_cajero(db:Session = Depends(database.get_db)):
    return db.query(models.Cajero).all() # Devuelve todos los cajeros de la tabla