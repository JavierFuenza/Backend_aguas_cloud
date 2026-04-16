"""
Script de verificación: ¿Las tablas de BD tienen las columnas que el código espera?
"""
import os
from dotenv import load_dotenv
load_dotenv()

from core.database import _execute_query_sync as execute_query

# Columnas que el código actual espera en cada tabla
EXPECTED = {
    "dw.Puntos_Mapa": [
        "UTM_Norte", "UTM_Este", "Huso", "Region", "Cod_Cuenca", "Cod_Subcuenca",
        "Cod_Subsubcuenca", "es_pozo_subterraneo", "codigo", "caudal_promedio",
        # Nuevas columnas del plan:
        "COD_SECTOR_SHA", "SECTOR_SHA", "APR", "ID_JUNTA",
        "PARTE_JUNTA", "REPRESENTA_JUNTA", "CANAL_TRANSMISION"
    ],
    "dw.Series_Tiempo": [
        "UTM_NORTE", "UTM_ESTE", "FECHA_MEDICION", "CAUDAL",
        "ALTURA_LIMNIMETRICA", "NIVEL_FREATICO",
        # Nueva columna del plan:
        "TOTALIZADOR",
        # Columnas de cuenca (usadas en filtros):
        "COD_CUENCA", "NOM_CUENCA", "COD_SUBCUENCA", "NOM_SUBCUENCA",
        "COD_SUBSUBCUENCA", "NOM_SUBSUBCUENCA"
    ],
    "dw.Mediciones_full": [
        "UTM_NORTE", "UTM_ESTE", "FECHA_MEDICION",
        "SECTOR_SHA", "APR", "ID_JUNTA", "PARTE_JUNTA",
        "REPRESENTA_JUNTA", "CANAL_TRANSMISION",
        "Cod_Cuenca", "Cod_Subcuenca", "Cod_Subsubcuenca"
    ]
}

def check_table(table_name, expected_cols):
    print(f"\n{'='*60}")
    print(f"  {table_name}")
    print(f"{'='*60}")
    try:
        res = execute_query(f"SELECT TOP 1 * FROM {table_name}", use_cache=False)
        if not res:
            print(f"  ⚠️  Tabla existe pero está vacía")
            return
        
        actual_cols = list(res[0].keys())
        actual_cols_upper = [c.upper() for c in actual_cols]
        
        print(f"  Columnas encontradas ({len(actual_cols)}):")
        for col in actual_cols:
            print(f"    - {col}")
        
        print(f"\n  Verificación de columnas esperadas:")
        missing = []
        for expected in expected_cols:
            if expected.upper() in actual_cols_upper:
                print(f"    ✅ {expected}")
            else:
                print(f"    ❌ {expected} — NO ENCONTRADA")
                missing.append(expected)
        
        if missing:
            print(f"\n  ⚠️  Faltan {len(missing)} columnas: {', '.join(missing)}")
        else:
            print(f"\n  ✅ Todas las columnas esperadas están presentes")
            
    except Exception as e:
        print(f"  ❌ Error al consultar: {e}")

def check_indices():
    print(f"\n{'='*60}")
    print(f"  ÍNDICES en dw.Puntos_Mapa")
    print(f"{'='*60}")
    try:
        res = execute_query("""
            SELECT i.name AS index_name, 
                   STRING_AGG(c.name, ', ') WITHIN GROUP (ORDER BY ic.key_ordinal) AS columns
            FROM sys.indexes i
            JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
            JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
            WHERE i.object_id = OBJECT_ID('dw.Puntos_Mapa')
            AND ic.is_included_column = 0
            GROUP BY i.name
        """, use_cache=False)
        if res:
            for r in res:
                print(f"    📊 {r['index_name']}: ({r['columns']})")
        else:
            print("    ⚠️  No se encontraron índices")
    except Exception as e:
        print(f"    ❌ Error consultando índices: {e}")

def check_sample_data():
    print(f"\n{'='*60}")
    print(f"  DATOS DE EJEMPLO — Columnas nuevas en Puntos_Mapa")
    print(f"{'='*60}")
    try:
        res = execute_query("""
            SELECT TOP 5 
                COD_SECTOR_SHA, SECTOR_SHA, APR, ID_JUNTA, CANAL_TRANSMISION
            FROM dw.Puntos_Mapa
            WHERE COD_SECTOR_SHA IS NOT NULL OR APR IS NOT NULL
        """, use_cache=False)
        if res:
            for r in res:
                print(f"    SHAC={r.get('COD_SECTOR_SHA')}, SHA={r.get('SECTOR_SHA')}, "
                      f"APR={r.get('APR')}, Junta={r.get('ID_JUNTA')}, Canal={r.get('CANAL_TRANSMISION')}")
        else:
            print("    ⚠️  No hay registros con datos en las columnas nuevas")
    except Exception as e:
        print(f"    ❌ Error: {e}")

if __name__ == "__main__":
    print("🔍 Verificación de esquema de BD vs. código del plan")
    print(f"   Servidor: {os.getenv('SYNAPSE_SERVER')}")
    print(f"   BD: {os.getenv('SYNAPSE_DATABASE')}")
    
    for table, cols in EXPECTED.items():
        check_table(table, cols)
    
    check_indices()
    check_sample_data()
    
    print(f"\n{'='*60}")
    print("✅ Verificación completada")
