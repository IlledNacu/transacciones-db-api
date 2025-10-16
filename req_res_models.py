from pydantic import BaseModel
from datetime import date, datetime
from decimal import Decimal
from typing import List

# ---- MODELOS PYDANTIC (Dataclass DTO para hacer las HTTP request y recibir las HTTP response) ---- 

class TipoTransaccionCreate(BaseModel):
    nombre: str

class TipoTransaccionResponse(BaseModel):
    id: int
    nombre: str
    class Config:
        from_attributes = True

class CajeroCreate(BaseModel):
    id: str
    nombre: str
    ciudad: str
    provincia: str
    pais: str

class CajeroResponse(CajeroCreate):
    class Config:
        from_attributes = True

class ClienteCreate(BaseModel):
    id: str
    nombre: str
    apellido: str
    genero: str
    fecha_nacimiento: date
    ocupacion: str
    tipo_cuenta: str

class ClienteResponse(ClienteCreate):
    class Config:
        from_attributes = True

class TransaccionCreate(BaseModel):
    id: str
    fecha_hora: datetime
    id_cliente: str
    id_cajero: str
    id_tipo_transaccion: int
    monto: Decimal

class TransaccionResponse(TransaccionCreate):
    class Config:
        from_attributes = True

class TransaccionSospechosaResponse(TransaccionCreate):
    class Config:
        from_attributes = True
    nombre: str
    apellido: str
    sospechosa_por: List[str]
    score_anomalia: float

class ClienteSospechosoResponse(BaseModel):
    id_cliente: str
    nombre: str
    apellido: str
    sospechoso_por: List[str]

# Hay que revisar si están funcionando bien las herencias actuales y si puede mejorarse