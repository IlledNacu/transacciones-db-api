from fastapi import FastAPI # Importamos el framework que nos permite construir y levantar una API
from database import engine, Base # Importamos de database.py el motor de conexión y el tipo de base elegidos
from endpoints import anomalias, tipos_transacciones, cajeros, clientes, transacciones # Importamos de endpoints/ las rutas creadas para consumir nuestra API

# ---- CREACIÓN DE LAS TABLAS ---- 

Base.metadata.create_all(bind=engine)
# Base.metadata, por ser de tipo declarativo, interpreta que los modelos definidos en models.py corresponden a las tablas

# ---- CREACIÓN DE LA API ---- 

app = FastAPI(title="Transacciones bancarias")

# ---- ENDPOINTS ---- 

# Endpoint inicial
@app.get("/")
def root():
    return {"message": "Agregue '/docs' al final del enlace para ingresar a la API de transacciones bancarias."}
# CRUD
app.include_router(cajeros.router)
app.include_router(clientes.router)
app.include_router(transacciones.router)
app.include_router(tipos_transacciones.router)
# Modelo de negocio
app.include_router(anomalias.router)
