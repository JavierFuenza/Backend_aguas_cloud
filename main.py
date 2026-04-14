import logging
from contextlib import asynccontextmanager
from queue import Queue, Empty
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize Config
from core.config import setup_config
setup_config()

import core.database as db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    db.connection_pool = Queue(maxsize=db.POOL_SIZE)

    # Pre-populate connection pool
    for _ in range(db.POOL_SIZE):
        try:
            conn = db.create_db_connection()
            db.connection_pool.put(conn)
        except Exception as e:
            logging.error(f"Failed to create initial connection: {e}")

    logging.info(f"Connection pool initialized with {db.connection_pool.qsize()} connections")
    yield

    # Shutdown
    while not db.connection_pool.empty():
        try:
            conn = db.connection_pool.get_nowait()
            conn.close()
        except Empty:
            break
        except Exception as e:
            logging.error(f"Error closing connection: {e}")

tags_metadata = [
    {
        "name": "System",
        "description": "Endpoints de sistema y diagnóstico para verificar el estado del servicio y la base de datos.",
    },
    {
        "name": "Puntos de Medición",
        "description": "Consulta de puntos de medición con coordenadas UTM, información de cuenca y estadísticas de caudal. Soporta filtros por región, cuenca, subcuenca y rangos de caudal.",
    },
    {
        "name": "Cuencas Hidrográficas",
        "description": "Información sobre cuencas, subcuencas y subsubcuencas hidrográficas, incluyendo estadísticas agregadas.",
    },
    {
        "name": "Series Temporales",
        "description": "Series temporales de caudal, altura limnimétrica y nivel freático por cuenca, subcuenca o punto específico.",
    },
    {
        "name": "Atlas",
        "description": "Divisiones administrativas (regiones, provincias y comunas) disponibles en el sistema.",
    },
    {
        "name": "Cache y Rendimiento",
        "description": "Gestión de caché y optimización del rendimiento de consultas.",
    },
    {
        "name": "Informantes",
        "description": "Consulta de informantes que han reportado mediciones y sus respectivas estaciones.",
    }
]

app = FastAPI(
    title="Aguas Transparentes API",
    description="API de Recursos Hídricos de Chile. Proporciona acceso a datos de mediciones de caudal, cuencas hidrográficas y series temporales almacenados en Azure Synapse Analytics. Sistema UTM Zona 19S.",
    version="1.7.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    root_path="/api",
    openapi_tags=tags_metadata,
    contact={
        "name": "Aguas Transparentes",
        "url": "https://github.com/JavierFuenza/Backend_aguas_cloud",
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400
)

# Include internal routers
from api.routers import (
    system,
    cache_y_rendimiento,
    puntos_de_medicion,
    cuencas_hidrograficas,
    series_temporales,
    atlas,
    informantes
)

app.include_router(system.router)
app.include_router(cache_y_rendimiento.router)
app.include_router(puntos_de_medicion.router)
app.include_router(cuencas_hidrograficas.router)
app.include_router(series_temporales.router)
app.include_router(atlas.router)
app.include_router(informantes.router)
