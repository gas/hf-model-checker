import argparse
import psutil
import torch
import re
from huggingface_hub import HfApi, scan_cache_dir
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# --- Funciones de Sistema ---

def get_system_memory():
    ram = psutil.virtual_memory().total / (1024**3)
    vram = 0
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return ram, vram

def get_local_files_for_repo(repo_id):
    local_files = set()
    try:
        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == repo_id:
                for r in repo.revisions:
                    for f in r.files:
                        local_files.add(f.file_name)
    except: pass
    return local_files

# --- Lógica de Agrupación (NUEVA) ---

def consolidate_files(files):
    """
    Agrupa archivos partidos (shards) tipo 'model-00001-of-00005.gguf'
    y filtra basura como 'imatrix'.
    """
    groups = {}
    
    for f in files:
        fname = f.rfilename
        
        # 1. Ignorar archivos que no son modelos reales
        if "imatrix" in fname.lower() or "mmproj" in fname.lower():
            continue

        # 2. Detectar patrón de Sharding (ej: -00001-of-00005)
        # Regex busca: (cualquier_cosa) seguido de -Digitos-of-Digitos.gguf
        shard_match = re.search(r"(.*)-\d{5}-of-\d{5}\.gguf", fname)
        
        if shard_match:
            # Si es una parte, usamos el nombre base como clave
            base_name = shard_match.group(1) + ".gguf (Split)"
            display_name = base_name.split('/')[-1] # Limpiar rutas largas
        else:
            # Si es un archivo único
            base_name = fname
            display_name = fname.split('/')[-1]

        if base_name not in groups:
            groups[base_name] = {
                'display_name': display_name,
                'real_files': [],
                'total_size': 0
            }
        
        groups[base_name]['real_files'].append(fname)
        groups[base_name]['total_size'] += f.size

    return groups.values()

# --- Análisis Principal ---

def analyze_model(repo_id):
    api = HfApi()
    try:
        model_info = api.model_info(repo_id, files_metadata=True)
    except Exception as e:
        console.print(f"[bold red]Error al contactar Hugging Face:[/bold red] {e}")
        return

    ram_sys, vram_sys = get_system_memory()
    local_files = get_local_files_for_repo(repo_id)

    # Filtrar solo GGUFs
    raw_gguf_files = [f for f in model_info.siblings if f.rfilename.endswith(".gguf")]

    if not raw_gguf_files:
        console.print("[yellow]Este repositorio no contiene archivos GGUF.[/yellow]")
        return

    # Usar la nueva función para limpiar y sumar
    consolidated_models = consolidate_files(raw_gguf_files)

    # Tabla
    table = Table(title=f"Análisis VRAM: {repo_id}")
    table.add_column("Modelo / Quant", style="cyan")
    table.add_column("Tamaño Total", justify="right")
    table.add_column("VRAM Req (+Ctx)", justify="right", style="magenta")
    table.add_column("Local", justify="center")
    table.add_column("Estado Real")

    recommendation = None
    best_score = -1
    
    # Convertir a lista y ordenar por tamaño
    sorted_models = sorted(consolidated_models, key=lambda x: x['total_size'])

    for m in sorted_models:
        size_gb = m['total_size'] / (1024**3)
        req_vram = size_gb * 1.15  # 15% margen contexto
        
        # Verificar si TODAS las partes están en local
        all_parts_local = all(f in local_files for f in m['real_files'])
        local_icon = "✅" if all_parts_local else ""

        # Estado
        if vram_sys >= req_vram:
            status = "[green]GPU (Rápido)[/green]"
            score = size_gb + 100
        elif (vram_sys + ram_sys) >= req_vram:
            offload_pct = (vram_sys / req_vram) * 100
            status = f"[yellow]Híbrido ({int(offload_pct)}% GPU)[/yellow]"
            score = size_gb
        else:
            status = "[red]No Cabe (Lento)[/red]"
            score = -1

        if score > best_score:
            best_score = score
            recommendation = m['display_name']

        table.add_row(
            m['display_name'],
            f"{size_gb:.2f} GB",
            f"{req_vram:.2f} GB",
            local_icon,
            status
        )

    console.print(table)
    
    if recommendation:
        console.print(Panel(
            f"Mejor opción real para tu hardware: [bold blue]{recommendation}[/bold blue]",
            title="🎯 Recomendación Inteligente", border_style="green"
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    repo = args.model.replace("https://huggingface.co/", "")
    analyze_model(repo)