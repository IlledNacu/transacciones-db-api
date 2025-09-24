# En terminal: pip install fastapi sqlalchemy uvicorn scikit-learn pandas

# IMPORTS
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Numeric, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel
from datetime import date, datetime
from decimal import Decimal
from typing import Optional, List, Dict
# Para el modelo de IA
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# MI API
app = FastAPI(title="Transacciones bancarias")

# Configuración base de datos
engine = create_engine("sqlite:///transacciones.db", connect_args={"check_same_thread":False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modelo base de datos
class TipoTransaccion(Base):
    __tablename__ = "tipos_transacciones"
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    nombre = Column(String(100), nullable=False, unique=True)

    transacciones = relationship("Transaccion", back_populates="tipo_transaccion")

class Cajero(Base):
    __tablename__ = "cajeros"
    id = Column(String(100), primary_key=True, index=True)
    nombre = Column(String(100), nullable=False)
    ciudad = Column(String(100), nullable=False)
    provincia = Column(String(100), nullable=False)
    pais = Column(String(100), nullable=False)

    transacciones = relationship("Transaccion", back_populates="cajero")

class Cliente(Base):
    __tablename__="clientes"
    id = Column(String(100), primary_key=True, index=True)
    nombre = Column(String(100), nullable=False)
    apellido = Column(String(100), nullable=False)
    genero = Column(String(50), nullable=False)
    fecha_nacimiento = Column(Date, nullable=False)
    ocupacion = Column(String(100), nullable=False)
    tipo_cuenta = Column(String(100), nullable=False)

    transacciones = relationship("Transaccion", back_populates="cliente")

class Transaccion(Base):
    __tablename__="transacciones"
    id = Column(String(100), primary_key=True, index=True)
    fecha_hora = Column(DateTime, nullable=False)
    id_cliente = Column(String(100), ForeignKey("clientes.id"), nullable=False)
    id_cajero = Column(String(100), ForeignKey("cajeros.id"), nullable=False)
    id_tipo_transaccion = Column(Integer, ForeignKey("tipos_transacciones.id"), nullable=False)
    monto = Column(Numeric(12,2), nullable=False)

    cliente = relationship("Cliente", back_populates="transacciones")
    cajero = relationship("Cajero", back_populates="transacciones")
    tipo_transaccion = relationship("TipoTransaccion", back_populates="transacciones")

class ClienteSospechosoResponse(BaseModel):
    id_cliente: str
    nombre: str
    apellido: str
    sospechoso_por: List[str]

Base.metadata.create_all(engine)

# Modelo Pydantic (Dataclass)
class TipoTransaccionCreate(BaseModel):
    nombre:str
class TipoTransaccionResponse(BaseModel):
    id:int
    nombre:str
    class Config:
        from_attributes = True

class CajeroCreate(BaseModel):
    id:str
    nombre:str
    ciudad:str
    provincia:str
    pais:str
class CajeroResponse(BaseModel):
    id:str
    nombre:str
    ciudad:str
    provincia:str
    pais:str
    class Config:
        from_attributes = True

class ClienteCreate(BaseModel):
    id:str
    nombre:str
    apellido:str
    genero:str
    fecha_nacimiento:date 
    ocupacion:str
    tipo_cuenta:str
class ClienteResponse(BaseModel):
    id:str
    nombre:str
    apellido:str 
    genero:str
    fecha_nacimiento:date
    ocupacion:str
    tipo_cuenta:str
    class Config:
        from_attributes = True

class TransaccionCreate(BaseModel):
    id:str
    fecha_hora:datetime 
    id_cliente:int
    id_cajero:int
    id_tipo_transaccion:int
    monto:Decimal
class TransaccionResponse(BaseModel):
    id:str
    fecha_hora:datetime 
    id_cliente:str
    id_cajero:str
    id_tipo_transaccion:int
    monto:Decimal
    class Config:
        from_attributes = True

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

get_db()

# ENDPOINTS
@app.get("/")
def root():
    return {"message":"Agregue '/docs' al final del enlace para ingresar a la API de transacciones bancarias."}

# CRUD TIPOS DE TRANSACCIONES
@app.get("/tipos_transacciones/{id}", response_model=TipoTransaccionResponse)
def get_tipo_transaccion(id:int, db:Session = Depends(get_db)):
    tipo_transaccion = db.query(TipoTransaccion).filter(TipoTransaccion.id == id).first()
    if not tipo_transaccion:
        raise HTTPException(status_code=404, detail="Tipo de transacción no encontrado.")
    return tipo_transaccion
@app.post("/tipos_transacciones/", response_model=TipoTransaccionResponse)
def create_tipo_transaccion(tipo_transaccion:TipoTransaccionCreate, db:Session = Depends(get_db)):
    if db.query(TipoTransaccion).filter(TipoTransaccion.nombre == tipo_transaccion.nombre).first():
        raise HTTPException(status_code=404, detail="Tipo de transacción ya registrado.")
    else:
        nuevo_tipo_transaccion = TipoTransaccion(**tipo_transaccion.model_dump())
        db.add(nuevo_tipo_transaccion)
        db.commit()
        db.refresh(nuevo_tipo_transaccion)
        return nuevo_tipo_transaccion
@app.put("/tipos_transacciones/{id}", response_model=TipoTransaccionResponse)
def update_tipo_transaccion(id:int, tipo_transaccion:TipoTransaccionCreate, db:Session = Depends(get_db)):
    db_tipo_transaccion = db.query(TipoTransaccion).filter(TipoTransaccion.id == id).first()
    if not db_tipo_transaccion:
        raise HTTPException(status_code=404, detail="Este tipo de transacción no existe.")
    for field, value in tipo_transaccion.model_dump().items():
        setattr(db_tipo_transaccion, field, value)
    db.commit()
    db.refresh(db_tipo_transaccion)
    return db_tipo_transaccion
@app.delete("/tipos_transacciones/{id}")
def delete_tipo_transaccion(id:int, db:Session = Depends(get_db)):
    db_tipo_transaccion = db.query(TipoTransaccion).filter(TipoTransaccion.id == id).first()
    if not db_tipo_transaccion:
        raise HTTPException(status_code=404, detail="Este tipo de transacción no existe.")
    db.delete(db_tipo_transaccion)
    db.commit()
    return {"message":"Número de cuenta eliminado."}
@app.get("/tipos_transacciones/", response_model=List[TipoTransaccionResponse])
def get_all_tipo_transaccion(db:Session = Depends(get_db)):
    return db.query(TipoTransaccion).all()

# CRUD CAJEROS
@app.get("/cajeros/{id}", response_model=CajeroResponse)
def get_cajero(id:int, db:Session = Depends(get_db)):
    tipo_cajero = db.query(Cajero).filter(Cajero.id == id).first()
    if not tipo_cajero:
        raise HTTPException(status_code=404, detail="Cajero no encontrado.")
    return tipo_cajero
@app.post("/cajeros/", response_model=CajeroResponse)
def create_cajero(cajero:CajeroCreate, db:Session = Depends(get_db)):
    nuevo_cajero = Cajero(**cajero.model_dump())
    db.add(nuevo_cajero)
    db.commit()
    db.refresh(nuevo_cajero)
    return nuevo_cajero
@app.put("/cajeros/{id}", response_model=CajeroResponse)
def update_cajero(id:int, cajero:CajeroCreate, db:Session = Depends(get_db)):
    db_cajero = db.query(Cajero).filter(Cajero.id == id).first()
    if not db_cajero:
        raise HTTPException(status_code=404, detail="Este cajero no existe.")
    for field, value in cajero.model_dump().items():
        setattr(db_cajero, field, value)
    db.commit()
    db.refresh(db_cajero)
    return db_cajero
@app.delete("/cajeros/{id}")
def delete_cajero(id:int, db:Session = Depends(get_db)):
    db_cajero = db.query(Cajero).filter(Cajero.id == id).first()
    if not db_cajero:
        raise HTTPException(status_code=404, detail="Este cajero no existe.")
    db.delete(db_cajero)
    db.commit()
    return {"message":"Cajero eliminado."}
@app.get("/cajeros/", response_model=List[CajeroResponse])
def get_all_cajero(db:Session = Depends(get_db)):
    return db.query(Cajero).all()

# CRUD CLIENTES
@app.get("/clientes/{id}", response_model=ClienteResponse)
def get_cliente(id:str, db:Session = Depends(get_db)):
    cliente = db.query(Cliente).filter(Cliente.id == id).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Número de cuenta no encontrado.")
    return cliente
@app.post("/clientes/", response_model=ClienteResponse)
def create_cliente(cliente:ClienteCreate, db:Session = Depends(get_db)):
    nuevo_cliente = Cliente(**cliente.model_dump())
    db.add(nuevo_cliente)
    db.commit()
    db.refresh(nuevo_cliente)
    return nuevo_cliente
@app.put("/clientes/{id}", response_model=ClienteResponse)
def update_cliente(id:str, cliente:ClienteCreate, db:Session = Depends(get_db)):
    db_cliente = db.query(Cliente).filter(Cliente.id == id).first()
    if not db_cliente:
        raise HTTPException(status_code=404, detail="Este número de cuenta no existe.")
    for field, value in cliente.model_dump().items():
        setattr(db_cliente, field, value)
    db.commit()
    db.refresh(db_cliente)
    return db_cliente
@app.delete("/clientes/{id}")
def delete_cliente(id:str, db:Session = Depends(get_db)):
    db_cliente = db.query(Cliente).filter(Cliente.id == id).first()
    if not db_cliente:
        raise HTTPException(status_code=404, detail="Este número de cuenta no existe.")
    db.delete(db_cliente)
    db.commit()
    return {"message":"Número de cuenta eliminado."}
@app.get("/clientes/", response_model=List[ClienteResponse])
def get_all_cliente(db:Session = Depends(get_db)):
    return db.query(Cliente).all()

# CRUD TRANSACCIONES
@app.get("/transacciones/{id}", response_model=TransaccionResponse)
def get_transaccion(id:str, db:Session = Depends(get_db)):
    transaccion = db.query(Transaccion).filter(Transaccion.id == id).first()
    if not transaccion:
        raise HTTPException(status_code=404, detail="Transacción no encontrada.")
    return transaccion
@app.post("/transacciones/", response_model=TransaccionResponse)
def create_transaccion(transaccion:TransaccionCreate, db:Session = Depends(get_db)):
    nueva_transaccion = Transaccion(**transaccion.model_dump())
    db.add(nueva_transaccion)
    db.commit()
    db.refresh(nueva_transaccion)
    return nueva_transaccion
@app.put("/transacciones/{id}", response_model=TransaccionResponse)
def update_transaccion(id:str, transaccion:TransaccionCreate, db:Session = Depends(get_db)):
    db_transaccion = db.query(Transaccion).filter(Transaccion.id == id).first()
    if not db_transaccion:
        raise HTTPException(status_code=404, detail="Esta transacción no existe.")
    for field, value in transaccion.model_dump().items():
        setattr(db_transaccion, field, value)
    db.commit()
    db.refresh(db_transaccion)
    return db_transaccion
@app.delete("/transacciones/{id}")
def delete_transaccion(id:str, db:Session = Depends(get_db)):
    db_transaccion = db.query(Transaccion).filter(Transaccion.id == id).first()
    if not db_transaccion:
        raise HTTPException(status_code=404, detail="Esta transacción no existe.")
    db.delete(db_transaccion)
    db.commit()
    return {"message":"Transacción eliminada."}
@app.get("/transacciones/", response_model=List[TransaccionResponse])
def get_all_transaccion(db:Session = Depends(get_db)):
    return db.query(Transaccion).all()

# Detección de clientes sospechosos con IA
@app.get("/clientes_sospechosos", response_model=List[ClienteSospechosoResponse])
def detectar_clientes_sospechosos(db: Session = Depends(get_db)):
    # Obtenemos datos de transacciones
    transacciones = db.query(Transaccion).all()
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
        cliente = db.query(Cliente).filter(Cliente.id == row["id_cliente"]).first()
        motivos = []
        if row["monto_maximo"] > row["monto_promedio"] * 5:
            motivos.append("Monto muy alto comparado con su promedio")
        if row["conteo_transacciones"] > df_features["conteo_transacciones"].mean() * 3:
            motivos.append("Frecuencia inusual de transacciones")
        if row["tiempo_entre_transacciones"] < df_features["tiempo_entre_transacciones"].mean() / 3:
            motivos.append("Transacciones demasiado seguidas")
        if not motivos:  # fallback si los modelos marcaron sospechoso pero no encontramos regla
            motivos.append("Comportamiento atípico indefinido")
        resultados.append(ClienteSospechosoResponse(
            id_cliente=row["id_cliente"],
            nombre=cliente.nombre if cliente else "Desconocido",
            apellido=cliente.apellido if cliente else "Desconocido",
            sospechoso_por=motivos
        ))

    return resultados

@app.get("/stats", response_model=Dict[str, float])
def get_stats(db: Session = Depends(get_db)):
    total_transacciones = db.query(Transaccion).count()
    total_clientes = db.query(Cliente).count()
    if total_transacciones == 0 or total_clientes == 0:
        raise HTTPException(status_code=404, detail="No hay datos suficientes para calcular estadísticas")

    # Reutilizamos la detección de clientes sospechosos
    clientes_sospechosos = detectar_clientes_sospechosos(db)

    # Total de transacciones sospechosas (las de esos clientes)
    ids_sospechosos = [c.id_cliente for c in clientes_sospechosos]
    total_transacciones_sospechosas = db.query(Transaccion).filter(
        Transaccion.id_cliente.in_(ids_sospechosos)
    ).count()

    porcentaje_clientes_sospechosos = (len(clientes_sospechosos) / total_clientes) * 100

    return {
        "total_transacciones": total_transacciones,
        "total_clientes": total_clientes,
        "total_transacciones_sospechosas": total_transacciones_sospechosas,
        "porcentaje_clientes_sospechosos": round(porcentaje_clientes_sospechosos, 2)
    }

# Levantamos la API  en la terminal con: uvicorn transaccionesapi:app --reload