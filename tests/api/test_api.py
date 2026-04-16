import requests
import time
import os
import sys
import argparse
import subprocess
from datetime import datetime
from colorama import init, Fore, Style
import pandas as pd
import random

# --- CONFIGURACIÓN ---
# NOTA: Cambia BASE_URL a "http://localhost:8000" si estás probando localmente antes de subir a Azure!
BASE_URL = "http://localhost:8000"  # <- URL LOCAL (Cambiar por Azure en producción)
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
init(autoreset=True)

results = []
current_run_id = 1

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
        "Run_ID": current_run_id,
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
        f.write(f"Total Runs:   {df['Run_ID'].nunique() if 'Run_ID' in df else 1}\n")
        f.write(f"Total Tests:  {total_count}\n")
        f.write(f"Exitosos:     {success_count}\n")
        f.write(f"Fallidos:     {total_count - success_count}\n")
        f.write(f"Tasa Éxito:   {success_rate:.2f}%\n")
        f.write(f"Latencia Prom:{avg_latency:.2f} ms\n\n")
        f.write("RESUMEN POR ENDPOINT:\n")
        f.write("-" * 30 + "\n")
        endpoint_stats = df.groupby(["Method", "Endpoint"]).agg(
            Avg_Time=('Time_ms', 'mean'),
            Count=('Result', 'count'),
            Pass_Count=('Result', lambda x: (x == 'PASS').sum())
        ).reset_index()
        for _, row in endpoint_stats.iterrows():
            f.write(f"{row['Method']} {row['Endpoint'][:40].ljust(40)} | Prom: {row['Avg_Time']:.2f}ms | Éxito: {row['Pass_Count']}/{row['Count']}\n")
        f.write("\nTOP 5 ENDPOINTS MÁS LENTOS:\n")
        f.write("-" * 30 + "\n")
        # Ordenar por tiempo descendente y tomar los top 5
        slowest = df.sort_values(by="Time_ms", ascending=False).head(5)
        for _, row in slowest.iterrows():
            f.write(f"{row['Time_ms']:.2f} ms | {row['Method']} {row['Endpoint']} (Run {row.get('Run_ID', 1)})\n")
            
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
    print(f"{Fore.YELLOW}=== RUN {current_run_id} | Iniciando Pruebas: Aguas Transparentes API ==={Style.RESET_ALL}")

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

    # PRUEBAS SHAC
    print(f"\n{Fore.MAGENTA}--- SHAC ---{Style.RESET_ALL}")
    test_endpoint("/cuencas/shac/series_de_tiempo/caudal", params={"shac_identificador": "499"})

    # 4. PRUEBAS DE PUNTOS
    print(f"\n{Fore.MAGENTA}--- Puntos de Medición ---{Style.RESET_ALL}")
    test_endpoint("/puntos/count", params={"region": 15}) 
    test_endpoint("/puntos/count", params={"apr": True, "id_junta": 1.0, "shac": 121}) # Filtros nuevos
    test_endpoint("/puntos/count", params={"id_tipo_extraccion": 2}) # Extracción superficial/subterránea
    test_endpoint("/puntos/count", params={"search": "obracod123"}) # Búsqueda por obra
    
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

def generate_final_report():
    print(f"\n{Fore.YELLOW}=== Generando Reportes Globales ==={Style.RESET_ALL}")
    if results:
        df = pd.DataFrame(results)
        
        # Mostrar resumen en consola
        avg_latency = df["Time_ms"].mean()
        success_rate = (df[df["Result"] == "PASS"].shape[0] / df.shape[0]) * 100
        
        print(f"Total Runs:     {df['Run_ID'].nunique()}")
        print(f"Total Tests:    {df.shape[0]}")
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
    parser = argparse.ArgumentParser(description="Pruebas de Carga Funcional para Aguas Transparentes API")
    parser.add_argument("--runs", type=int, default=None, help="Número de veces a ejecutar las pruebas")
    parser.add_argument("--target", type=str, default=None, help="Target API URL")
    args = parser.parse_args()

    # Preguntar por entorno
    is_local = True
    if args.target:
        BASE_URL = args.target
        is_local = "localhost" in BASE_URL or "127.0.0.1" in BASE_URL
    else:
        env_choice = input("¿Deseas probar el entorno Local (L) o Producción (P)? [L/P]: ").strip().upper()
        if env_choice == 'P':
            is_local = False
            prod_input = input("Ingresa la URL de producción (Ej: https://tu-api.azurewebsites.net): ").strip()
            BASE_URL = prod_input if prod_input else "https://aguastransparentes.azurewebsites.net"
        else:
            is_local = True
            BASE_URL = "http://127.0.0.1:8000"

    runs = args.runs
    if runs is None:
        user_input = input("¿Cuántas veces deseas repetir las pruebas? (Enter para 1): ")
        try:
            runs = int(user_input) if user_input.strip() else 1
        except ValueError:
            print("Entrada no válida, ejecutando 1 vez por defecto.")
            runs = 1

    server_process = None
    if is_local:
        print(f"\n{Fore.CYAN}Iniciando servidor local automáticamente (uvicorn main:app)...{Style.RESET_ALL}")
        # Iniciar uvicorn con uv descartando el output para no ensuciar la consola
        server_process = subprocess.Popen(
            ["uv", "run", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        print("Esperando a que el servidor local esté listo (máx 30s)...")
        server_ready = False
        for _ in range(30):
            # Verificar si el proceso terminó inesperadamente (ej: puerto ya en uso)
            if server_process.poll() is not None:
                print(f"{Fore.RED}Error: Uvicorn se cerró inesperadamente. ¿Quizás el puerto 8000 ya está en uso?{Style.RESET_ALL}")
                sys.exit(1)
                
            try:
                res = requests.get(f"{BASE_URL}/health")
                if res.status_code == 200:
                    server_ready = True
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
                
        if not server_ready:
            print(f"{Fore.RED}Error: El servidor local no se inició a tiempo.{Style.RESET_ALL}")
            if server_process:
                server_process.terminate()
            sys.exit(1)
        print(f"{Fore.GREEN}Servidor local iniciado correctamente.{Style.RESET_ALL}\n")

    print(f"{Fore.YELLOW}=== Iniciando Pruebas de Carga Funcional: Aguas Transparentes API ==={Style.RESET_ALL}")
    print(f"Target: {BASE_URL}")
    print(f"Repeticiones: {runs}\n")

    try:
        for i in range(1, runs + 1):
            current_run_id = i
            run_tests()
            if i < runs:
                time.sleep(1) # Pequeña pausa entre corridas
        generate_final_report()
    except requests.exceptions.ConnectionError:
        print(f"{Fore.RED}ERROR CRÍTICO: No se pudo conectar a {BASE_URL}. Asegúrate de que la API esté corriendo.{Style.RESET_ALL}")
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Pruebas interrumpidas por el usuario. Generando reporte parcial...{Style.RESET_ALL}")
        generate_final_report()
    finally:
        if server_process:
            print(f"\n{Fore.CYAN}Apagando servidor local...{Style.RESET_ALL}")
            server_process.terminate()
            server_process.wait()
            print(f"{Fore.GREEN}Servidor apagado.{Style.RESET_ALL}")