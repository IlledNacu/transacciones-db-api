from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base
from endpoints import anomalias, tipos_transacciones, cajeros, clientes, transacciones

# Creación de las tablas
Base.metadata.create_all(bind=engine)

# Creación de la API
app = FastAPI(title="Transacciones bancarias")

# Permitir CORS para que se comunique con el frontend
origins = [
    "http://127.0.0.1:5500",  # ruteo al front
    "http://localhost:5500"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
