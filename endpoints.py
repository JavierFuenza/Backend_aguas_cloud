# Función helper para construir nombres
def build_full_name(nomb_inf: str, a_pat_inf: str, a_mat_inf: str) -> str:
    """Construye el nombre completo del informante de manera eficiente"""
    parts = [part for part in [nomb_inf, a_pat_inf, a_mat_inf] if part]
    return " ".join(parts) if parts else "Desconocido"


# Context manager para manejo de sesión DB
@asynccontextmanager
async def get_db_session():
    """Context manager para manejo seguro de sesiones de base de datos"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UTMLocation(BaseModel):
    utm_norte: int
    utm_este: int


@app.on_event("startup")
def on_startup():
    """
    Función que se ejecuta al inicio de la aplicación FastAPI.
    Aquí crearemos todas las tablas definidas en nuestros modelos.
    """
    logging.info("Iniciando la aplicación y creando tablas en la base de datos...")
    create_tables()
    logging.info("Tablas creadas exitosamente o ya existentes.")


@app.get("/count", summary="Obtiene el número total de registros en la tabla ObrasMedicion",tags=["count"])
async def get_obras_count():
    db = SessionLocal()
    try:
        count = db.query(func.count(ObrasMedicion.id)).scalar()
        return {"total_records": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

'''
@app.get("/puntos", summary="Obtiene coordenadas únicas filtradas con datos importantes",tags=["puntos"])
async def get_coordenadas_unicas(
    region: Optional[int] = Query(None),
    cod_cuenca: Optional[int] = Query(None),
    cod_subcuenca: Optional[int] = Query(None),
    limit: Optional[int] = Query(120),
    filtro_null_subcuenca: Optional[bool] = Query(None, description="Si es True, filtra por subcuenca nula. Ignora 'cod_subcuenca' si es True."),
    caudal_minimo: Optional[float] = Query(None),
    caudal_maximo: Optional[float] = Query(None),
    orden_caudal: Optional[str] = Query("min", description="Orden del caudal promedio: 'min' o 'max'"),
):
    db: Session = SessionLocal()

    try:
        # Construir query base con agrupación y promedio de caudal
        # Seleccionar nom_inf, a_pat_inf, a_mat_inf
        query = db.query(
            ObrasMedicion.nom_cuenca,
            ObrasMedicion.nom_subcuenca,
            ObrasMedicion.comuna,
            ObrasMedicion.utm_norte,
            ObrasMedicion.utm_este,
            ObrasMedicion.huso,
            ObrasMedicion.cod_subcuenca,
            ObrasMedicion.cod_cuenca,
            func.avg(ObrasMedicion.caudal).label("caudal_promedio"),
            func.count(ObrasMedicion.caudal).label("n_mediciones"),
            ObrasMedicion.nomb_inf, # Incluir nombre
            ObrasMedicion.a_pat_inf, # Incluir apellido paterno
            ObrasMedicion.a_mat_inf # Incluir apellido materno
        )

        # Filtros básicos
        if region is not None:
            query = query.filter(ObrasMedicion.region == region)
        if cod_cuenca is not None:
            query = query.filter(ObrasMedicion.cod_cuenca == cod_cuenca)

        # Handle subcuenca filtering logic
        if filtro_null_subcuenca:
            query = query.filter(ObrasMedicion.cod_subcuenca.is_(None))
        elif cod_subcuenca is not None:
            query = query.filter(ObrasMedicion.cod_subcuenca == cod_subcuenca)

        # Ensure we only consider records with non-null caudal for averaging and counting.
        query = query.filter(ObrasMedicion.caudal.isnot(None))

        # Agrupar por coordenadas únicas y otros atributos para obtener el promedio y conteo por punto
        # Añadir nomb_inf, a_pat_inf, a_mat_inf al GROUP BY
        query = query.group_by(
            ObrasMedicion.nom_cuenca,
            ObrasMedicion.nom_subcuenca,
            ObrasMedicion.comuna,
            ObrasMedicion.utm_norte,
            ObrasMedicion.utm_este,
            ObrasMedicion.huso,
            ObrasMedicion.cod_subcuenca,
            ObrasMedicion.cod_cuenca,
            ObrasMedicion.nomb_inf,
            ObrasMedicion.a_pat_inf,
            ObrasMedicion.a_mat_inf
        )

        # Filtrar por caudal mínimo y máximo después del promedio (HAVING clause)
        if caudal_minimo is not None:
            query = query.having(func.avg(ObrasMedicion.caudal) >= caudal_minimo)
        if caudal_maximo is not None:
            query = query.having(func.avg(ObrasMedicion.caudal) <= caudal_maximo)

        # Ordenamiento por caudal promedio
        if orden_caudal == "max":
            query = query.order_by(func.avg(ObrasMedicion.caudal).desc())
        else:  # por defecto, "min"
            query = query.order_by(func.avg(ObrasMedicion.caudal).asc())

        # Aplicar límite
        results = query.limit(limit).all()

        # Conversión de coordenadas
        transformer = Transformer.from_crs("epsg:32719", "epsg:4326", always_xy=True)
        coordenadas = []
        for r in results:
            try:
                lon, lat = transformer.transform(r.utm_este, r.utm_norte)
            except Exception as e:
                logging.warning(f"Error transforming coordinates {r.utm_este}, {r.utm_norte}: {e}")
                lon, lat = None, None

            # Construir el nombre completo del informante
            full_informante_name_parts = [r.nomb_inf, r.a_pat_inf, r.a_mat_inf]
            full_informante_name = " ".join(filter(None, full_informante_name_parts)).strip()
            if not full_informante_name:
                full_informante_name = "Desconocido"

            coordenadas.append({
                "lat": lat,
                "lon": lon,
                "utm_norte": r.utm_norte,
                "utm_este": r.utm_este,
                "nombre_cuenca": r.nom_cuenca or "No existe registro",
                "nombre_subcuenca": r.nom_subcuenca or "No existe registro",
                "comuna": r.comuna or "No existe registro",
                "cod_cuenca": r.cod_cuenca,
                "cod_subcuenca": r.cod_subcuenca,
                "caudal_promedio": round(r.caudal_promedio, 2) if r.caudal_promedio is not None else None,
                "n_mediciones": r.n_mediciones,
                "nombre_informante": full_informante_name # Usar el nombre completo
            })

        return coordenadas

    except Exception as e:
        logging.error(f"Error in get_coordenadas_unicas: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
'''

@app.get("/puntos", summary="Obtiene coordenadas únicas filtradas con datos importantes",tags=["puntos"])
async def get_coordenadas_unicas(
    region: Optional[int] = Query(None),
    cod_cuenca: Optional[int] = Query(None),
    cod_subcuenca: Optional[int] = Query(None),
    limit: Optional[int] = Query(120),
    filtro_null_subcuenca: Optional[bool] = Query(None, description="Si es True, filtra por subcuenca nula. Ignora 'cod_subcuenca' si es True."),
    caudal_minimo: Optional[float] = Query(None),
    caudal_maximo: Optional[float] = Query(None),
    orden_caudal: Optional[str] = Query("min", description="Orden del caudal promedio: 'min' o 'max'"),
):
    # Usar dependency injection en lugar de crear sesión manualmente
    async with get_db_session() as db:
        try:
            # Query optimizada con menos campos en GROUP BY usando subquery
            subquery = db.query(
                ObrasMedicion.utm_norte,
                ObrasMedicion.utm_este,
                func.avg(ObrasMedicion.caudal).label("caudal_promedio"),
                func.count(ObrasMedicion.caudal).label("n_mediciones"),
                # Usar agregaciones para los otros campos para evitar GROUP BY extenso
                func.max(ObrasMedicion.nom_cuenca).label("nom_cuenca"),
                func.max(ObrasMedicion.nom_subcuenca).label("nom_subcuenca"),
                func.max(ObrasMedicion.comuna).label("comuna"),
                func.max(ObrasMedicion.huso).label("huso"),
                func.max(ObrasMedicion.cod_subcuenca).label("cod_subcuenca"),
                func.max(ObrasMedicion.cod_cuenca).label("cod_cuenca"),
                func.max(ObrasMedicion.nomb_inf).label("nomb_inf"),
                func.max(ObrasMedicion.a_pat_inf).label("a_pat_inf"),
                func.max(ObrasMedicion.a_mat_inf).label("a_mat_inf")
            )

            # Aplicar filtros early para reducir dataset
            subquery = subquery.filter(ObrasMedicion.caudal.isnot(None))
            
            if region is not None:
                subquery = subquery.filter(ObrasMedicion.region == region)
            if cod_cuenca is not None:
                subquery = subquery.filter(ObrasMedicion.cod_cuenca == cod_cuenca)

            # Handle subcuenca filtering logic
            if filtro_null_subcuenca:
                subquery = subquery.filter(ObrasMedicion.cod_subcuenca.is_(None))
            elif cod_subcuenca is not None:
                subquery = subquery.filter(ObrasMedicion.cod_subcuenca == cod_subcuenca)

            # GROUP BY solo por coordenadas (clave principal)
            subquery = subquery.group_by(
                ObrasMedicion.utm_norte,
                ObrasMedicion.utm_este
            )

            # Filtros de caudal con HAVING
            if caudal_minimo is not None:
                subquery = subquery.having(func.avg(ObrasMedicion.caudal) >= caudal_minimo)
            if caudal_maximo is not None:
                subquery = subquery.having(func.avg(ObrasMedicion.caudal) <= caudal_maximo)

            # Ordenamiento
            if orden_caudal == "max":
                subquery = subquery.order_by(func.avg(ObrasMedicion.caudal).desc())
            else:
                subquery = subquery.order_by(func.avg(ObrasMedicion.caudal).asc())

            # Aplicar límite en la subquery
            results = subquery.limit(limit).all()

            # Optimizar transformación de coordenadas
            if not results:
                return []

            # Crear transformer una sola vez
            transformer = Transformer.from_crs("epsg:32719", "epsg:4326", always_xy=True)
            
            # Batch transform para mejor performance
            utm_coords = [(r.utm_este, r.utm_norte) for r in results]
            try:
                # Transformar todas las coordenadas de una vez
                transformed_coords = [transformer.transform(east, north) for east, north in utm_coords]
            except Exception as e:
                logging.warning(f"Batch coordinate transformation failed: {e}")
                # Fallback a transformación individual
                transformed_coords = []
                for east, north in utm_coords:
                    try:
                        lon, lat = transformer.transform(east, north)
                        transformed_coords.append((lon, lat))
                    except Exception:
                        transformed_coords.append((None, None))

            # Construir respuesta usando list comprehension para mejor performance
            coordenadas = [
                {
                    "lat": coord[1],
                    "lon": coord[0],
                    "utm_norte": r.utm_norte,
                    "utm_este": r.utm_este,
                    "nombre_cuenca": r.nom_cuenca or "No existe registro",
                    "nombre_subcuenca": r.nom_subcuenca or "No existe registro",
                    "comuna": r.comuna or "No existe registro",
                    "cod_cuenca": r.cod_cuenca,
                    "cod_subcuenca": r.cod_subcuenca,
                    "caudal_promedio": round(r.caudal_promedio, 2) if r.caudal_promedio is not None else None,
                    "n_mediciones": r.n_mediciones,
                    "nombre_informante": build_full_name(r.nomb_inf, r.a_pat_inf, r.a_mat_inf)
                }
                for r, coord in zip(results, transformed_coords)
            ]

            return coordenadas

        except Exception as e:
            logging.error(f"Error in get_coordenadas_unicas: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/cuencas", summary="Obtiene regiones, cuencas y subcuencas únicas",tags=["cuencas"])
async def get_unique_cuencas():
    db: Session = SessionLocal()
    try:
        # Subconsulta: promedios por coordenadas únicas (utm_norte)
        # This subquery is crucial for the logic. The indexes on `cod_cuenca`, `nom_cuenca`,
        # `cod_subcuenca`, `nom_subcuenca`, `region`, and `caudal` will help here.
        subq = (
            db.query(
                ObrasMedicion.utm_norte.label("utm_norte"),
                ObrasMedicion.nom_cuenca.label("nom_cuenca"), #
                ObrasMedicion.cod_cuenca.label("cod_cuenca"), #
                ObrasMedicion.nom_subcuenca.label("nom_subcuenca"), #
                ObrasMedicion.cod_subcuenca.label("cod_subcuenca"), #
                ObrasMedicion.region.label("cod_region"), #
                func.avg(ObrasMedicion.caudal).label("promedio_caudal") #
            )
            .filter(ObrasMedicion.caudal != None) #
            .group_by(
                ObrasMedicion.utm_norte,
                ObrasMedicion.nom_cuenca,
                ObrasMedicion.cod_cuenca,
                ObrasMedicion.nom_subcuenca,
                ObrasMedicion.cod_subcuenca,
                ObrasMedicion.region
            )
        ).subquery()

        # Lista única de regiones, cuencas y subcuencas
        # The `DISTINCT` clause will benefit from the indexes on `cod_region`, `nom_cuenca`, etc.,
        # as these columns are part of the underlying subquery and are explicitly indexed.
        cuencas = db.query(
            subq.c.cod_region,
            subq.c.nom_cuenca,
            subq.c.cod_cuenca,
            subq.c.nom_subcuenca,
            subq.c.cod_subcuenca
        ).distinct().all()

        return {
            "cuencas": [
                {
                    "cod_region": r.cod_region,
                    "nom_cuenca": r.nom_cuenca,
                    "cod_cuenca": r.cod_cuenca,
                    "nom_subcuenca": r.nom_subcuenca,
                    "cod_subcuenca": r.cod_subcuenca
                } for r in cuencas
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/cuencas/stats", summary="Obtiene estadísticas de caudal por cuenca y subcuenca",tags=["cuencas"])
async def get_cuencas_stats():
    db: Session = SessionLocal()

    try:
        # Subconsulta: promedios por coordenadas únicas (utm_norte)
        # Same rationale as get_unique_cuencas: indexes help this subquery significantly.
        subq = (
            db.query(
                ObrasMedicion.utm_norte.label("utm_norte"),
                ObrasMedicion.nom_cuenca.label("nom_cuenca"),
                ObrasMedicion.cod_cuenca.label("cod_cuenca"),
                ObrasMedicion.nom_subcuenca.label("nom_subcuenca"),
                ObrasMedicion.cod_subcuenca.label("cod_subcuenca"),
                ObrasMedicion.region.label("cod_region"),
                func.avg(ObrasMedicion.caudal).label("promedio_caudal")
            )
            .filter(ObrasMedicion.caudal != None) #
            .group_by(
                ObrasMedicion.utm_norte,
                ObrasMedicion.nom_cuenca,
                ObrasMedicion.cod_cuenca,
                ObrasMedicion.nom_subcuenca,
                ObrasMedicion.cod_subcuenca,
                ObrasMedicion.region
            )
        ).subquery()

        # Estadísticas globales
        # Aggregations over the subquery also benefit from the optimized subquery.
        min_max_result = db.query(
            func.min(subq.c.promedio_caudal),
            func.max(subq.c.promedio_caudal)
        ).first()
        if min_max_result is not None:
            min_global, max_global = min_max_result
        else:
            min_global, max_global = None, None

        total_puntos_unicos = db.query(func.count(subq.c.utm_norte)).scalar()

        # Por cuenca
        # Grouping by nom_cuenca benefits from its index.
        caudal_por_cuenca = db.query(
            subq.c.nom_cuenca,
            func.min(subq.c.promedio_caudal).label("avgMin"),
            func.max(subq.c.promedio_caudal).label("avgMax"),
            func.count(subq.c.utm_norte).label("puntos")
        ).group_by(subq.c.nom_cuenca).all()

        # Por subcuenca
        # Grouping by nom_cuenca and nom_subcuenca benefits from their indexes,
        # and potentially from 'idx_cuenca_subcuenca' if the optimizer can use it.
        caudal_por_subcuenca = db.query(
            subq.c.nom_cuenca,
            subq.c.nom_subcuenca,
            func.min(subq.c.promedio_caudal).label("avgMin"),
            func.max(subq.c.promedio_caudal).label("avgMax"),
            func.count(subq.c.utm_norte).label("puntos")
        ).group_by(subq.c.nom_cuenca, subq.c.nom_subcuenca).all()

        return {
            "estadisticas": {
                "caudal_global": {
                    "avgMin": min_global,
                    "avgMax": max_global,
                    "total_puntos_unicos": total_puntos_unicos
                },
                "caudal_por_cuenca": [
                    {
                        "nom_cuenca": c.nom_cuenca,
                        "avgMin": c.avgMin,
                        "avgMax": c.avgMax,
                        "total_puntos": c.puntos
                    } for c in caudal_por_cuenca
                ],
                "caudal_por_subcuenca": [
                    {
                        "nom_cuenca": s.nom_cuenca,
                        "nom_subcuenca": s.nom_subcuenca,
                        "avgMin": s.avgMin,
                        "avgMax": s.avgMax,
                        "total_puntos": s.puntos
                    } for s in caudal_por_subcuenca
                ]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/cuencas/analisis_caudal", summary="Realiza un análisis estadístico de caudal por cuenca",tags=["cuencas"])
async def get_analisis_cuenca(
    cuenca_identificador: str = Query(..., description="Código o nombre de la cuenca")
):
    db = SessionLocal()
    try:
        query = db.query(ObrasMedicion.caudal)

        # Filters on `cod_cuenca` or `nom_cuenca` directly use their indexes.
        if cuenca_identificador.isdigit():  # Si es un número, asumir que es un código
            query = query.filter(ObrasMedicion.cod_cuenca == int(cuenca_identificador)) #
        else:  # Si no, asumir que es un nombre
            query = query.filter(ObrasMedicion.nom_cuenca == cuenca_identificador) #

        # Filtering `caudal.isnot(None)` leverages the `caudal` index.
        query = query.filter(ObrasMedicion.caudal.isnot(None)) #

        # Calcular estadísticas
        # Aggregations benefit from the filters narrowing down the dataset.
        count = query.count()
        avg_caudal = query.with_entities(func.avg(ObrasMedicion.caudal)).scalar()
        min_caudal = query.with_entities(func.min(ObrasMedicion.caudal)).scalar()
        max_caudal = query.with_entities(func.max(ObrasMedicion.caudal)).scalar()
        std_caudal = query.with_entities(func.stddev(ObrasMedicion.caudal)).scalar()

        if count == 0:
            return {"message": "No se encontraron datos de caudal para la cuenca especificada."}

        return {
            "cuenca_identificador": cuenca_identificador,
            "total_registros_con_caudal": count,
            "caudal_promedio": avg_caudal,
            "caudal_minimo": min_caudal,
            "caudal_maximo": max_caudal,
            "desviacion_estandar_caudal": std_caudal
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/cuencas/analisis_informantes", summary="Genera datos para gráficos de barras de informantes por cuenca",tags=["cuencas"])
async def get_informantes_por_cuenca(
    cuenca_identificador: str = Query(..., description="Código o nombre de la cuenca")
):
    db = SessionLocal()
    try:
        base_query = db.query(ObrasMedicion)

        # Filtering by cuenca code or name uses the respective indexes.
        if cuenca_identificador.isdigit():
            base_query = base_query.filter(ObrasMedicion.cod_cuenca == int(cuenca_identificador)) #
        else:
            base_query = base_query.filter(ObrasMedicion.nom_cuenca == cuenca_identificador) #

        # Agrupación por informante para el conteo de registros
        # The 'nomb_inf' index supports this GROUP BY.
        informantes_count = base_query.with_entities(
            ObrasMedicion.nomb_inf, #
            func.count(ObrasMedicion.id)
        ).group_by(ObrasMedicion.nomb_inf).all() #

        # Agrupación por informante para la suma de caudal total extraído
        # Filtering `caudal.isnot(None)` and grouping by `nomb_inf` both benefit from indexes.
        informantes_caudal_total = base_query.with_entities(
            ObrasMedicion.nomb_inf, #
            func.sum(ObrasMedicion.caudal) #
        ).filter(ObrasMedicion.caudal.isnot(None)).group_by(ObrasMedicion.nomb_inf).all() #

        # Formatear resultados para el gráfico de cantidad de registros
        data_registros = []
        for nom_inf, count in informantes_count:
            data_registros.append({
                "informante": nom_inf if nom_inf else "Desconocido",
                "cantidad_registros": count
            })

        # Formatear resultados para el gráfico de caudal total
        data_caudal = []
        for nom_inf, total_caudal in informantes_caudal_total:
            data_caudal.append({
                "informante": nom_inf if nom_inf else "Desconocido",
                "caudal_total_extraido": total_caudal if total_caudal else 0
            })
        
        # Opcional: Contar obras únicas por informante
        # `nombre_obra` index helps with the distinct count.
        informantes_obras_unicas = base_query.with_entities(
            ObrasMedicion.nomb_inf, #
            func.count(ObrasMedicion.nombre_obra.distinct()) #
        ).group_by(ObrasMedicion.nomb_inf).all() #

        data_obras_unicas = []
        for nom_inf, unique_works_count in informantes_obras_unicas:
            data_obras_unicas.append({
                "informante": nom_inf if nom_inf else "Desconocido",
                "cantidad_obras_unicas": unique_works_count
            })

        return {
            "cuenca_identificador": cuenca_identificador,
            "grafico_cantidad_registros_por_informante": data_registros,
            "grafico_caudal_total_por_informante": data_caudal,
            "grafico_cantidad_obras_unicas_por_informante": data_obras_unicas # Nuevo dato para el gráfico
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/cuencas/series_de_tiempo/caudal", summary="Obtiene el caudal extraído a lo largo del tiempo para una cuenca específica",tags=["cuencas"])
async def get_caudal_por_tiempo_por_cuenca(
    cuenca_identificador: str = Query(..., description="Código o nombre de la cuenca"),
    fecha_inicio: Optional[date] = Query(None, description="Fecha de inicio (YYYY-MM-DD) para filtrar mediciones"),
    fecha_fin: Optional[date] = Query(None, description="Fecha de fin (YYYY-MM-DD) para filtrar mediciones")
):
    db = SessionLocal()
    try:
        query = db.query(
            ObrasMedicion.fecha_medicion, #
            ObrasMedicion.caudal #
        )

        # Filtering by cuenca code or name uses the respective indexes.
        # The composite indexes `idx_cuenca_fecha_medicion` and `idx_nom_cuenca_fecha_medicion`
        # are designed specifically for these types of queries.
        if cuenca_identificador.isdigit():
            query = query.filter(ObrasMedicion.cod_cuenca == int(cuenca_identificador)) #
        else:
            query = query.filter(ObrasMedicion.nom_cuenca == cuenca_identificador) #

        # Filtrar solo registros con caudal y fecha de medición no nulos
        # This, in conjunction with the cuenca filter, leverages the composite indexes.
        query = query.filter(
            ObrasMedicion.caudal.isnot(None), #
            ObrasMedicion.fecha_medicion.isnot(None) #
        )

        if fecha_inicio:
            query = query.filter(ObrasMedicion.fecha_medicion >= fecha_inicio) #
        if fecha_fin:
            query = query.filter(ObrasMedicion.fecha_medicion <= fecha_fin) #

        # Ordenar por fecha para una mejor visualización temporal
        # The `fecha_medicion` part of the composite index also helps with ordering.
        query = query.order_by(ObrasMedicion.fecha_medicion) #

        results = query.all()

        caudal_por_tiempo = []
        for fecha, caudal in results:
            caudal_por_tiempo.append({
                "fecha_medicion": fecha.isoformat() if fecha else None, # Formato ISO para fechas
                "caudal": caudal
            })
        
        if not caudal_por_tiempo:
            raise HTTPException(status_code=404, detail="No se encontraron datos de caudal para el período o cuenca especificada.")

        return {
            "cuenca_identificador": cuenca_identificador,
            "caudal_por_tiempo": caudal_por_tiempo
        }

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


from fastapi import Body, HTTPException
from sqlalchemy import or_

@app.post("/puntos/estadisticas", summary="Obtiene estadísticas de caudal para uno o varios puntos UTM específicos", tags=["puntos"])
async def get_point_statistics(
    locations: List[UTMLocation] = Body(..., description="Lista de puntos UTM (Norte, Este)")
):
    db: Session = SessionLocal()
    try:
        if not locations:
            raise HTTPException(status_code=400, detail="Debe proporcionar al menos una coordenada UTM")

        # Caso: una sola coordenada → comportamiento original
        if len(locations) == 1:
            loc = locations[0]
            utm_norte = loc.utm_norte
            utm_este = loc.utm_este

            query = db.query(ObrasMedicion).filter(
                ObrasMedicion.utm_norte == utm_norte,
                ObrasMedicion.utm_este == utm_este,
                ObrasMedicion.caudal.isnot(None)
            )

            count = query.count()
            avg_caudal = query.with_entities(func.avg(ObrasMedicion.caudal)).scalar()
            min_caudal = query.with_entities(func.min(ObrasMedicion.caudal)).scalar()
            max_caudal = query.with_entities(func.max(ObrasMedicion.caudal)).scalar()
            std_caudal = query.with_entities(func.stddev(ObrasMedicion.caudal)).scalar()

            if count == 0:
                return [{
                    "utm_norte": utm_norte,
                    "utm_este": utm_este,
                    "message": "No se encontraron datos de caudal para las coordenadas UTM especificadas."
                }]
            else:
                return [{
                    "utm_norte": utm_norte,
                    "utm_este": utm_este,
                    "total_registros_con_caudal": count,
                    "caudal_promedio": round(avg_caudal, 2) if avg_caudal is not None else None,
                    "caudal_minimo": round(min_caudal, 2) if min_caudal is not None else None,
                    "caudal_maximo": round(max_caudal, 2) if max_caudal is not None else None,
                    "desviacion_estandar_caudal": round(std_caudal, 2) if std_caudal is not None else None
                }]

        # Caso: múltiples coordenadas → análisis conjunto
        else:
            # Generar condiciones OR encadenadas para las coordenadas
            condiciones = [
                (ObrasMedicion.utm_norte == loc.utm_norte) & (ObrasMedicion.utm_este == loc.utm_este)
                for loc in locations
            ]

            query = db.query(ObrasMedicion).filter(
                or_(*condiciones),
                ObrasMedicion.caudal.isnot(None)
            )

            count = query.count()
            avg_caudal = query.with_entities(func.avg(ObrasMedicion.caudal)).scalar()
            min_caudal = query.with_entities(func.min(ObrasMedicion.caudal)).scalar()
            max_caudal = query.with_entities(func.max(ObrasMedicion.caudal)).scalar()
            std_caudal = query.with_entities(func.stddev(ObrasMedicion.caudal)).scalar()

            return [{
                "puntos_consultados": len(locations),
                "total_registros_con_caudal": count,
                "caudal_promedio": round(avg_caudal, 2) if avg_caudal is not None else None,
                "caudal_minimo": round(min_caudal, 2) if min_caudal is not None else None,
                "caudal_maximo": round(max_caudal, 2) if max_caudal is not None else None,
                "desviacion_estandar_caudal": round(std_caudal, 2) if std_caudal is not None else None
            }]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()



@app.get("/puntos/series_de_tiempo/caudal", summary="Obtiene el caudal extraído a lo largo del tiempo para un punto UTM específico",tags=["puntos"])
async def get_caudal_por_tiempo_por_punto(
    utm_norte: int = Query(..., description="Coordenada UTM Norte del punto"),
    utm_este: int = Query(..., description="Coordenada UTM Este del punto"),
    fecha_inicio: Optional[date] = Query(None, description="Fecha de inicio (YYYY-MM-DD) para filtrar mediciones"),
    fecha_fin: Optional[date] = Query(None, description="Fecha de fin (YYYY-MM-DD) para filtrar mediciones")
):
    db = SessionLocal()
    try:
        query = db.query(
            ObrasMedicion.fecha_medicion,
            ObrasMedicion.caudal
        )

        # Filtrar por las coordenadas UTM exactas, aprovechando el índice 'idx_utm_coords'
        query = query.filter(
            ObrasMedicion.utm_norte == utm_norte,
            ObrasMedicion.utm_este == utm_este
        )

        # Filtrar solo registros con caudal y fecha de medición no nulos
        query = query.filter(
            ObrasMedicion.caudal.isnot(None),
            ObrasMedicion.fecha_medicion.isnot(None)
        )

        if fecha_inicio:
            query = query.filter(ObrasMedicion.fecha_medicion >= fecha_inicio)
        if fecha_fin:
            query = query.filter(ObrasMedicion.fecha_medicion <= fecha_fin)

        # Ordenar por fecha para una mejor visualización temporal
        query = query.order_by(ObrasMedicion.fecha_medicion)

        results = query.all()

        caudal_por_tiempo = []
        for fecha, caudal in results:
            caudal_por_tiempo.append({
                "fecha_medicion": fecha.isoformat() if fecha else None,
                "caudal": caudal
            })
        
        if not caudal_por_tiempo:
            raise HTTPException(status_code=404, detail="No se encontraron datos de caudal para el punto UTM o período especificado.")

        return {
            "utm_norte": utm_norte,
            "utm_este": utm_este,
            "caudal_por_tiempo": caudal_por_tiempo
        }

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()