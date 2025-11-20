# Sistema de Detección de Fraude Bancario (Detector de Anomalías)

Este proyecto implementa un **API RESTful** utilizando **FastAPI** para la gestión de datos de transacciones bancarias y la detección de anomalías (fraude) en clientes y transacciones a través de algoritmos de Machine Learning no supervisado.
El enfoque principal del sistema, siguiendo la recomendación de optimización, se centra en la **detección de clientes sospechosos** para luego obtener las transacciones asociadas, logrando así una mayor eficiencia en el procesamiento de grandes volúmenes de datos.

## Arquitectura y Algoritmos de Detección

El sistema utiliza tres modelos de Detección de Anomalías no supervisados para establecer un **consenso** y mejorar la precisión:

* **Isolation Forest (iForest):** Detecta anomalías basándose en el aislamiento de los puntos atípicos.
* **Local Outlier Factor (LOF):** Identifica anomalías midiendo la densidad local de un punto de datos en comparación con sus vecinos.
* **K-Means (Clustering):** Marca como anómalos aquellos puntos que se encuentran más alejados de su centroide de clúster, detectando patrones de comportamiento alejados de la norma.

### Detección de Clientes Sospechosos (`anomalias_clientes.py`)

Se calculan *features* de comportamiento (ej: `monto_promedio`, `conteo_transacciones`, `tiempo_entre_transacciones`) para cada cliente. Un cliente es marcado como sospechoso si es detectado por un **mínimo de 2 de los 3 modelos de IA**, además de cumplir con criterios de negocio (ej: monto máximo desproporcionado respecto al promedio).

## Estructura del Proyecto

El proyecto se organiza en módulos claros para la gestión de la lógica de negocio y los *endpoints* del API:<br>

├── data/ # CSVs limpios<br>
├── data_original/ # CSVs originales obtenidos en [kaggle.com/dataset](https://www.kaggle.com/datasets/obinnaiheanachor/wisabi-bank-dataset)<br>
├── endpoints/<br>
│ ├── anomalias_clientes.py # Lógica de detección de Clientes Sospechosos<br>
│ ├── anomalias_transacciones.py # Lógica de detección de Transacciones Sospechosas<br>
│ ├── estadisticas.py # Lógica para métricas y dashboard (utiliza la detección de anomalías)<br>
│ └── (otros endpoints: cajeros, clientes, transacciones, etc.)<br>
├── cargar_csvs.py # Script para cargar datos iniciales (CSVs) a la base de datos.<br>
├── database.py # Configuración de la conexión a la base de datos (SQLite/SQLAlchemy).<br>
├── main.py # Punto de entrada de la aplicación FastAPI.<br>
├── models.py # Definición de los modelos de base de datos (SQLAlchemy).<br>
├── procesar_datos.py # Lógica de pre-procesamiento de datos iniciales (limpia y ajusta los CSVs originales).<br>
└── req_res_models.py # Modelos de solicitud/respuesta (Pydantic).

## Configuración y Ejecución

Pasos para poner en marcha el sistema:

### 1. Requisitos Previos

Asegúrate de tener instalado Python (preferentemente 3.9+) y `pip`.

### 2. Instalación de Dependencias

Ejecutar en Bash:<br>
`pip install -r requerimientos.txt` # Instala las dependencias: fastapi, uvicorn, sqlalchemy, pandas, scikit-learn, plotly.

### 3. Orden de Ejecución de Archivos

Ejecutar en la terminal:<br>
1. `python procesar_datos.py` # Realiza la limpieza, normalización o cualquier transformación inicial de los datos fuente.
2. `python main.py` # Inicializa la aplicación FastAPI, asegurando que las tablas del ORM se creen en la base de datos.
3. `python cargar_csvs.py` # Carga los datos ya procesados (CSVs) a las tablas de la base de datos.
4. `uvicorn main:app --reload` # Inicia el servidor API en modo desarrollo.

### 4. Acceso al Dashboard y API

Para acceder a la API Interactiva (Swagger UI): http://127.0.0.1:8000/docs <br>

Ya podemos levantar el Frontend con LiveServer y acceder a nuestro Dashboard.
