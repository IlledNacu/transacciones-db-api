from sqlalchemy import Column, Integer, String, Date, DateTime, Numeric, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

# ---- MODELADO DE LA BD (según la sintaxis de SQLAlchemy) ---- 

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
    __tablename__ = "clientes"
    id = Column(String(100), primary_key=True, index=True)
    nombre = Column(String(100), nullable=False)
    apellido = Column(String(100), nullable=False)
    genero = Column(String(50), nullable=False)
    fecha_nacimiento = Column(Date, nullable=False)
    ocupacion = Column(String(100), nullable=False)
    tipo_cuenta = Column(String(100), nullable=False)

    transacciones = relationship("Transaccion", back_populates="cliente")

class Transaccion(Base):
    __tablename__ = "transacciones"
    id = Column(String, primary_key=True, index=True)
    fecha_hora = Column(DateTime, nullable=False, index=True)
    monto = Column(Numeric(12, 2), nullable=False, index=True)
    id_cliente = Column(String, ForeignKey("clientes.id"), nullable=False, index=True)
    id_cajero = Column(String, ForeignKey("cajeros.id"), nullable=False)
    id_tipo_transaccion = Column(Integer, ForeignKey("tipos_transacciones.id"), nullable=False)

    cliente = relationship("Cliente", back_populates="transacciones")
    cajero = relationship("Cajero", back_populates="transacciones")
    tipo_transaccion = relationship("TipoTransaccion", back_populates="transacciones")
