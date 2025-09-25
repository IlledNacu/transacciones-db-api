from fastapi import FastAPI
from database import engine, Base
from endpoints import anomalias, tipos_transacciones, cajeros, clientes, transacciones

# Creación de las tablas
Base.metadata.create_all(bind=engine)

# Creación de la API
app = FastAPI(title="Transacciones bancarias")

# Endpoint inicial
@app.get("/")
def root():
    return {"message": "Agregue '/docs' al final del enlace para ingresar a la API de transacciones bancarias."}

# Inclusión de rutas (endpoints)
# CRUD
app.include_router(cajeros.router)
app.include_router(clientes.router)
app.include_router(transacciones.router)
app.include_router(tipos_transacciones.router)
# Modelo de negocio
app.include_router(anomalias.router)
