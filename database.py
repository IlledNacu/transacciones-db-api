from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///transacciones.db"

# Configuración de la base de datos
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Obtención de la base de datos
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
