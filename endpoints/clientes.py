from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import models
import req_res_models
import database
from typing import List

router = APIRouter(prefix="/clientes", tags=["Clientes"])

@router.get("/{id}", response_model=req_res_models.ClienteResponse)
def get_cliente(id:str, db:Session = Depends(database.get_db)):
    cliente = db.query(models.Cliente).filter(models.Cliente.id == id).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Número de cuenta no encontrado.")
    return cliente

@router.post("/", response_model=req_res_models.ClienteResponse)
def create_cliente(cliente:req_res_models.ClienteCreate, db:Session = Depends(database.get_db)):
    nuevo_cliente = models.Cliente(**cliente.model_dump())
    db.add(nuevo_cliente)
    db.commit()
    db.refresh(nuevo_cliente)
    return nuevo_cliente

@router.put("/{id}", response_model=req_res_models.ClienteResponse)
def update_cliente(id:str, cliente:req_res_models.ClienteCreate, db:Session = Depends(database.get_db)):
    db_cliente = db.query(models.Cliente).filter(models.Cliente.id == id).first()
    if not db_cliente:
        raise HTTPException(status_code=404, detail="Este número de cuenta no existe.")
    for field, value in cliente.model_dump().items():
        setattr(db_cliente, field, value)
    db.commit()
    db.refresh(db_cliente)
    return db_cliente

@router.delete("/{id}")
def delete_cliente(id:str, db:Session = Depends(database.get_db)):
    db_cliente = db.query(models.Cliente).filter(models.Cliente.id == id).first()
    if not db_cliente:
        raise HTTPException(status_code=404, detail="Este número de cuenta no existe.")
    db.delete(db_cliente)
    db.commit()
    return {"message":"Número de cuenta eliminado."}

@router.get("/", response_model=List[req_res_models.ClienteResponse])
def get_all_cliente(db:Session = Depends(database.get_db)):
    return db.query(models.Cliente).all()

# Todas las consultas siguen la misma lógica que las de cajeros.py