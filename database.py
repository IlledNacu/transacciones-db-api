from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ---- CONFIGURACIÓN DE LA BD CON SQLAlchemy (biblioteca de Python) ---- 

DATABASE_URL = "sqlite:///transacciones.db"
# URL de conexión a la base de datos. En este caso es un archivo SQLite que se crea en el directorio del proyecto si no existe.
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
# El engine es el Motor de Conexión que mantiene la conexión entre SQLAlchemy y la bd. "connect_args" es un parámetro propio de SQLite.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Crea una sesión que permite interactuar con la bd.
Base = declarative_base()
# La clase de base. Determina cómo tiene que ser definida sintácticamente una tabla para ser interpretada dentro de Base.metadata (la estructura que usamos en models.py)

# ---- OBTENCIÓN DE LA BD ---- 

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
