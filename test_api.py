import requests
import time
import os
from datetime import datetime
from colorama import init, Fore, Style
import pandas as pd
import random

# --- CONFIGURACIÓN ---
BASE_URL = "http://127.0.0.1:8000/api"  # Asegúrate de que coincida con tu API
REPORTS_DIR = "reports"
init(autoreset=True)

results = []

def log_result(endpoint, method, status, time_ms, error=None):
    """Registra en memoria y muestra en consola el resultado"""
    color = Fore.GREEN if 200 <= status < 300 else Fore.RED
    status_icon = "✅" if 200 <= status < 300 else "❌"
    
    # Imprimir en consola (feedback inmediato)
    print(f"{status_icon} {Fore.CYAN}[{method}]{Style.RESET_ALL} {endpoint.ljust(60)} "
          f"{color}{status}{Style.RESET_ALL} | {time_ms:.2f} ms")
    
    if error:
        print(f"    {Fore.YELLOW}Error: {error}{Style.RESET_ALL}")

    # Guardar en la lista de resultados
    results.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Endpoint": endpoint,
        "Method": method,
        "Status": status,
        "Time_ms": round(time_ms, 2),
        "Result": "PASS" if 200 <= status < 300 else "FAIL",
        "Error": str(error) if error else ""
    })

def save_reports(df):
    """Genera archivos de reporte (CSV y TXT)"""
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{REPORTS_DIR}/test_run_{timestamp_str}"
    
    # 1. Guardar CSV (Datos crudos para análisis)
    csv_filename = f"{base_filename}.csv"
    df.to_csv(csv_filename, index=False)
    
    # 2. Guardar Resumen TXT (Para lectura rápida)
    txt_filename = f"{base_filename}_summary.txt"
    
    avg_latency = df["Time_ms"].mean()
    success_count = df[df["Result"] == "PASS"].shape[0]
    total_count = df.shape[0]
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(f"REPORTE DE EJECUCIÓN API - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Target:       {BASE_URL}\n")
        f.write(f"Total Tests:  {total_count}\n")
        f.write(f"Exitosos:     {success_count}\n")
        f.write(f"Fallidos:     {total_count - success_count}\n")
        f.write(f"Tasa Éxito:   {success_rate:.2f}%\n")
        f.write(f"Latencia Prom:{avg_latency:.2f} ms\n\n")
        f.write("TOP 5 ENDPOINTS MÁS LENTOS:\n")
        f.write("-" * 30 + "\n")
        # Ordenar por tiempo descendente y tomar los top 5
        slowest = df.sort_values(by="Time_ms", ascending=False).head(5)
        for _, row in slowest.iterrows():
            f.write(f"{row['Time_ms']:.2f} ms | {row['Method']} {row['Endpoint']}\n")
            
    return csv_filename, txt_filename

def test_endpoint(endpoint, method="GET", params=None, json_body=None):
    """Ejecuta una petición y mide el tiempo"""
    url = f"{BASE_URL}{endpoint}"
    start_time = time.time()
    try:
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            response = requests.post(url, json=json_body)
        
        elapsed_time = (time.time() - start_time) * 1000
        
        # Intentar parsear JSON para verificar validez
        try:
            data = response.json()
        except:
            data = None
            
        log_result(endpoint, method, response.status_code, elapsed_time)
        return response.status_code, data

    except Exception as e:
        elapsed_time = (time.time() - start_time) * 1000
        log_result(endpoint, method, 0, elapsed_time, str(e))
        return 0, None

def run_tests():
    print(f"{Fore.YELLOW}=== Iniciando Pruebas de Carga Funcional: Aguas Transparentes API ==={Style.RESET_ALL}")
    print(f"Target: {BASE_URL}\n")

    # 1. SISTEMA Y HEALTH
    print(f"\n{Fore.MAGENTA}--- Sistema ---{Style.RESET_ALL}")
    test_endpoint("/health")
    test_endpoint("/test-db")
    test_endpoint("/count")
    test_endpoint("/atlas")
    test_endpoint("/cache/stats")

    # 2. OBTENER DATOS SEMILLA
    print(f"\n{Fore.MAGENTA}--- Recolectando datos semilla ---{Style.RESET_ALL}")
    
    _, cuencas_data = test_endpoint("/cuencas")
    cuenca_sample = None
    subcuenca_sample = None
    subsubcuenca_sample = None
    
    if cuencas_data and "cuencas" in cuencas_data and len(cuencas_data["cuencas"]) > 0:
        candidates = [c for c in cuencas_data["cuencas"] if c['cod_subsubcuenca'] is not None]
        if candidates:
            sample = random.choice(candidates)
            cuenca_sample = sample['cod_cuenca']
            subcuenca_sample = sample['cod_subcuenca']
            subsubcuenca_sample = sample['cod_subsubcuenca']
            print(f"    -> Usando Cuenca ID: {cuenca_sample}, Sub: {subcuenca_sample}, SubSub: {subsubcuenca_sample}")
        else:
            sample = cuencas_data["cuencas"][0]
            cuenca_sample = sample['cod_cuenca']
            print(f"    -> Usando Cuenca ID: {cuenca_sample} (Sin subsubcuenca)")

    _, puntos_data = test_endpoint("/puntos", params={"limit": 5})
    punto_sample = None
    if puntos_data and isinstance(puntos_data, list) and len(puntos_data) > 0:
        punto_sample = puntos_data[0]
        print(f"    -> Usando Punto UTM: N:{punto_sample['utm_norte']} E:{punto_sample['utm_este']}")

    # 3. PRUEBAS DE CUENCAS
    print(f"\n{Fore.MAGENTA}--- Cuencas e Hidrografía ---{Style.RESET_ALL}")
    test_endpoint("/filtrosreactivos")
    
    if cuenca_sample:
        test_endpoint("/cuencas/stats", params={"cod_cuenca": cuenca_sample, "include_global": True})
        test_endpoint("/cuencas/cuenca/series_de_tiempo/caudal", params={"cuenca_identificador": cuenca_sample})
        test_endpoint("/cuencas/cuenca/series_de_tiempo/altura_linimetrica", params={"cuenca_identificador": cuenca_sample})
        test_endpoint("/cuencas/cuenca/series_de_tiempo/nivel_freatico", params={"cuenca_identificador": cuenca_sample})

    if subcuenca_sample:
        test_endpoint("/cuencas/subcuenca/series_de_tiempo/caudal", params={"cuenca_identificador": subcuenca_sample})
        test_endpoint("/cuencas/subcuenca/series_de_tiempo/altura_linimetrica", params={"cuenca_identificador": subcuenca_sample}) # Agregado
        test_endpoint("/cuencas/subcuenca/series_de_tiempo/nivel_freatico", params={"cuenca_identificador": subcuenca_sample})

    if subsubcuenca_sample:
        test_endpoint("/cuencas/subsubcuenca/series_de_tiempo/caudal", params={"cuenca_identificador": subsubcuenca_sample})
        test_endpoint("/cuencas/subsubcuenca/series_de_tiempo/altura_linimetrica", params={"cuenca_identificador": subsubcuenca_sample})
        test_endpoint("/cuencas/subsubcuenca/series_de_tiempo/nivel_freatico", params={"cuenca_identificador": subsubcuenca_sample}) # Agregado

    # 4. PRUEBAS DE PUNTOS
    print(f"\n{Fore.MAGENTA}--- Puntos de Medición ---{Style.RESET_ALL}")
    test_endpoint("/puntos/count", params={"region": 15}) 
    
    if punto_sample:
        norte = punto_sample['utm_norte']
        este = punto_sample['utm_este']
        test_endpoint("/puntos/info", params={"utm_norte": norte, "utm_este": este})
        test_endpoint("/puntos/series_de_tiempo/caudal", params={"utm_norte": norte, "utm_este": este})
        test_endpoint("/puntos/series_de_tiempo/altura_linimetrica", params={"utm_norte": norte, "utm_este": este})
        test_endpoint("/puntos/series_de_tiempo/nivel_freatico", params={"utm_norte": norte, "utm_este": este})
        
        body = [{"utm_norte": norte, "utm_este": este}]
        test_endpoint("/puntos/estadisticas", method="POST", json_body=body)

    # 5. RENDIMIENTO Y CACHÉ
    print(f"\n{Fore.MAGENTA}--- Rendimiento ---{Style.RESET_ALL}")
    test_endpoint("/cache/clear", method="POST") # Agregado
    test_endpoint("/performance/warm-up")

    # --- GENERACIÓN DE REPORTE ---
    print(f"\n{Fore.YELLOW}=== Generando Reportes ==={Style.RESET_ALL}")
    if results:
        df = pd.DataFrame(results)
        
        # Mostrar resumen en consola
        avg_latency = df["Time_ms"].mean()
        success_rate = (df[df["Result"] == "PASS"].shape[0] / df.shape[0]) * 100
        
        print(f"Success Rate:   {success_rate:.1f}%")
        print(f"Avg Latency:    {avg_latency:.2f} ms")
        
        # Guardar archivos
        csv_path, txt_path = save_reports(df)
        print(f"\n{Fore.GREEN}Reportes guardados exitosamente:{Style.RESET_ALL}")
        print(f"📂 CSV: {csv_path}")
        print(f"📄 TXT: {txt_path}")
    else:
        print(f"{Fore.RED}No se ejecutaron pruebas.{Style.RESET_ALL}")

if __name__ == "__main__":
    try:
        run_tests()
    except requests.exceptions.ConnectionError:
        print(f"{Fore.RED}ERROR CRÍTICO: No se pudo conectar a {BASE_URL}. Asegúrate de que la API esté corriendo.{Style.RESET_ALL}")