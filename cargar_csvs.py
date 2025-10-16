import pandas as pd
from sqlalchemy.orm import Session
from datetime import datetime
from decimal import Decimal
from database import SessionLocal
from models import Cliente, Cajero, Transaccion, TipoTransaccion

# ---- CARGA DE DATOS A LA BD DESDE CSVs ----

db: Session = SessionLocal() # Creamos una sesión para conectar con la bd.

# Cargamos data/clientes.csv usando la biblioteca de pandas y en base a la estructura de nuestros modelos
clientes_df = pd.read_csv("data/clientes.csv") # Se crea un dataframe con todos los datos
for _, row in clientes_df.iterrows(): # y por cada línea, iterativamente
    fecha_nac = datetime.strptime(row["fecha_nacimiento"], "%Y-%m-%d").date() # transformamos el formato de la fecha
    # y guardamos cada dato en la columna correspondiente:
    cliente = Cliente(
        id=row["id"],
        nombre=row["nombre"],
        apellido=row["apellido"],
        genero=row["genero"],
        fecha_nacimiento=fecha_nac,
        ocupacion=row["ocupacion"],
        tipo_cuenta=row["tipo_cuenta"]
    )
    db.add(cliente)

# Ídem a clientes con las demás tablas (cajeros, tipos_transacciones, transacciones)
cajeros_df = pd.read_csv("data/cajeros.csv")
for _, row in cajeros_df.iterrows():
    cajero = Cajero(
        id=row["id"],
        nombre=row["nombre"],
        ciudad=row["ciudad"],
        provincia=row["provincia"],
        pais=row["pais"]
    )
    db.add(cajero)

tipos_df = pd.read_csv("data/tipos_transacciones.csv")
for _, row in tipos_df.iterrows():
    tipo = TipoTransaccion(
        nombre=row["nombre"]
    )
    db.add(tipo)

transacciones_df = pd.read_csv("data/transacciones_modificadas.csv")
for _, row in transacciones_df.iterrows():
    fecha_hora = datetime.strptime(row["fecha_hora"], "%Y-%m-%d %H:%M:%S")

    transaccion = Transaccion(
        id=row["id"],
        fecha_hora=fecha_hora,
        id_cliente=row["id_cliente"],
        id_cajero=row["id_cajero"],
        id_tipo_transaccion=row["id_tipo_transaccion"],
        monto=Decimal(str(row["monto"]))
    )
    db.add(transaccion)

db.commit() # Confirmamos los cambios (si no no se guardan)
db.close() # Cerramos la bd

print("Datos cargados correctamente")