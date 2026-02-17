import questionary
from huggingface_hub import HfApi, scan_cache_dir
import subprocess
import sys
import os
import psutil
import torch
import threading
import time

# --- Configuración Global ---
api = HfApi()

CATEGORIES = {
    "🔍 Búsqueda Manual": "search",
    "💻 Programación (Coding)": "coder",
    "🧠 Razonamiento (Reasoning/R1)": "reasoning",
    "💬 Chat General (Instruct)": "instruct",
    "👁️ Visión / Multimodal": "multimodal",
    "🎭 Roleplay / Historia": "roleplay",
    "🧪 Modelos Pequeños (<3B)": "smol",
    "🏠 Ver mis modelos descargados": "local_only"
}

# Variable global para la caché (se llena en segundo plano)
LOCAL_CACHE_REPOS = set()
CACHE_READY = False

# --- Funciones de Sistema y Fondo ---

def get_hardware_info():
    """Obtiene info básica rápido para la cabecera"""
    ram = psutil.virtual_memory().total / (1024**3)
    vram = 0
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return ram, vram

def background_cache_scanner():
    """Escanea la caché sin bloquear el menú"""
    global LOCAL_CACHE_REPOS, CACHE_READY
    try:
        cache_info = scan_cache_dir()
        repos = set()
        for repo in cache_info.repos:
            # Solo nos interesan repos con GGUFs
            has_gguf = any(f.file_name.lower().endswith(".gguf") for r in repo.revisions for f in r.files)
            if has_gguf:
                repos.add(repo.repo_id)
        LOCAL_CACHE_REPOS = repos
        CACHE_READY = True
    except Exception:
        pass

# --- Lógica de Búsqueda ---

def get_models(query, limit=10):
    """Busca en la API de HF"""
    print(f"\n🌐 Buscando '{query}' (Top {limit})...")
    try:
        models = api.list_models(
            filter="gguf",
            search=query,
            sort="trending_score",
            direction=-1,
            limit=limit
        )
        return [m.modelId for m in models]
    except Exception as e:
        print(f"Error de conexión: {e}")
        return []

# --- Menú Principal ---

def main():
    # 1. Iniciar escáner en hilo separado (Asíncrono real)
    scanner_thread = threading.Thread(target=background_cache_scanner, daemon=True)
    scanner_thread.start()

    # 2. Leer hardware una sola vez al inicio
    ram_sys, vram_sys = get_hardware_info()
    
    current_limit = 10
    last_query = None
    last_tag = None

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Cabecera Informativa
        print("==========================================")
        print(f"   🚀 HF MODEL CHECKER  |  RAM: {ram_sys:.1f}GB  VRAM: {vram_sys:.1f}GB")
        if CACHE_READY:
            print(f"   💾 Caché Local: {len(LOCAL_CACHE_REPOS)} modelos detectados")
        else:
            print("   💾 Caché Local: Escaneando en segundo plano...")
        print("==========================================\n")

        # Selección de Categoría
        cat_name = questionary.select(
            "¿Qué quieres explorar hoy?",
            choices=list(CATEGORIES.keys()) + ["❌ Salir"]
        ).ask()

        if cat_name == "❌ Salir" or cat_name is None:
            break

        tag = CATEGORIES[cat_name]
        
        # Gestión de búsqueda nueva vs paginación
        if tag != last_tag:
            current_limit = 10 # Reset paginación si cambiamos de categoría
            last_tag = tag

        model_ids = []

        # Lógica por tipo de categoría
        if tag == "local_only":
            # Esperar a que la caché esté lista si el usuario pide ver SOLO lo local
            if not CACHE_READY:
                print("⏳ Esperando al escáner de disco...")
                while not CACHE_READY: time.sleep(0.5)
            model_ids = list(LOCAL_CACHE_REPOS)
        
        elif tag == "search":
            query = questionary.text("Escribe el nombre (ej: Mistral):").ask()
            if not query: continue
            last_query = query
            model_ids = get_models(query, limit=current_limit)
        
        else:
            # Categorías predefinidas
            last_query = tag # Para usarlo en la paginación
            model_ids = get_models(tag, limit=current_limit)

        if not model_ids:
            print("❌ No se encontraron modelos.")
            questionary.press_any_key_to_continue().ask()
            continue

        # --- Sub-Menú de Resultados con Paginación ---
        while True:
            choices = []
            for m_id in model_ids:
                # Icono dinámico: Si el hilo de fondo ya lo encontró, pone la casita
                prefix = "🏠" if (CACHE_READY and m_id in LOCAL_CACHE_REPOS) else "🌐"
                urlbase = "https://huggingface.co/"
                choices.append(questionary.Choice(title=f"{prefix} {urlbase}{m_id}", value=m_id))
            
            # Botones de control
            if tag != "local_only": # No paginamos lo local
                choices.append(questionary.Choice(title="⬇️  Cargar 10 más...", value="LOAD_MORE"))
            
            choices.append(questionary.Choice(title="⬅️  Volver al menú", value="BACK"))

            selected = questionary.select(
                f"Resultados ({len(model_ids)}):",
                choices=choices
            ).ask()

            if selected == "BACK":
                break
            
            elif selected == "LOAD_MORE":
                current_limit += 10
                # Recargamos usando la query guardada
                if tag == "search":
                    model_ids = get_models(last_query, limit=current_limit)
                else:
                    model_ids = get_models(last_tag, limit=current_limit)
                continue # Volvemos a pintar la lista con los nuevos items
            
            elif selected:
                # Ejecutar el Checker
                
                # 1. Obtener la ruta absoluta donde vive ESTE script (hf_navigator.py)
                script_dir = os.path.dirname(os.path.abspath(__file__))
                
                # 2. Construir la ruta completa al checker
                checker_path = os.path.join(script_dir, "hf_model_checker.py")

                # 3. Ejecutar usando la ruta absoluta
                # Verificamos si existe por si acaso
                if os.path.exists(checker_path):
                    # Pasamos la VRAM como argumento para no recalcularla en el hijo si no queremos
                    subprocess.run([sys.executable, checker_path, "--model", selected])
                else:
                    print(f"❌ Error: No encuentro {checker_path}")                


                print("\n" + "-"*50)
                questionary.press_any_key_to_continue().ask()
                # Al volver, redibujamos la lista (break del submenu visual, no del loop principal)
                # O podemos hacer continue para quedarnos en la lista. Haremos continue visual.
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"--- Viendo resultados para: {cat_name} ---")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Saliendo...")