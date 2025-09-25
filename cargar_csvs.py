import pandas as pd
from sqlalchemy.orm import Session
from datetime import datetime
from decimal import Decimal
from database import SessionLocal
from models import Cliente, Cajero, Transaccion, TipoTransaccion

# Creación de la sesión
db: Session = SessionLocal()

# Carga de datos
# ---- CLIENTES ----
clientes_df = pd.read_csv("data/clientes.csv")
for _, row in clientes_df.iterrows():
    fecha_nac = datetime.strptime(row["fecha_nacimiento"], "%Y-%m-%d").date()

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

# ---- CAJEROS ----
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

# ---- TIPOS DE TRANSACCIONES ----
tipos_df = pd.read_csv("data/tipos_transacciones.csv")
for _, row in tipos_df.iterrows():
    tipo = TipoTransaccion(
        nombre=row["nombre"]
    )
    db.add(tipo)

# ---- TRANSACCIONES ----
transacciones_df = pd.read_csv("data/transacciones.csv")
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

# Confirmación de cambios
db.commit()
db.close()

print("Datos cargados correctamente")