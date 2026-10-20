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

# --- Lógica de Agrupación Universal ---

def consolidate_files(files):
    """
    Agrupa archivos partidos (shards) de cualquier tipo (.gguf, .safetensors, .bin)
    y filtra archivos de metadatos o irrelevantes.
    """
    groups = {}
    
    for f in files:
        fname = f.rfilename
        
        # 1. Filtros de limpieza
        if any(x in fname.lower() for x in ["imatrix", "mmproj", "vocab", "tokenizer", "config.json"]):
            continue

        # 2. Detectar patrón de Sharding Genérico
        # Soporta: model-00001-of-00005.safetensors, part-1-of-2.gguf, etc.
        # Regex: (NombreBase)(Separador)(Digitos)(Separador)(Digitos)(Extension)
        shard_match = re.search(r"(.*)([\-\_\.])\d{2,5}[-\_]of[-\_]\d{2,5}(\.(gguf|safetensors|bin))", fname)
        
        if shard_match:
            base_name = shard_match.group(1) + ".SHARED" # Marca interna
            display_name = base_name.split('/')[-1].replace(".SHARED", "") + " (Split)"
            file_type = shard_match.group(3) # .gguf, .safetensors
        else:
            base_name = fname
            display_name = fname.split('/')[-1]
            file_type = "." + fname.split('.')[-1]

        if base_name not in groups:
            groups[base_name] = {
                'display_name': display_name,
                'real_files': [],
                'total_size': 0,
                'type': file_type
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

    all_files = model_info.siblings
    
    # Estrategia de Búsqueda:
    # 1. Prioridad GGUF (Modelos listos para correr)
    target_files = [f for f in all_files if f.rfilename.endswith(".gguf")]
    mode = "GGUF (Inferencia)"

    # 2. Si no hay GGUF, buscar Safetensors/Bin (Modelos base/entrenamiento)
    if not target_files:
        target_files = [f for f in all_files if f.rfilename.endswith(".safetensors") or f.rfilename.endswith(".bin")]
        mode = "Base (Safetensors/Bin)"
    
    if not target_files:
        console.print("[yellow]Este repositorio no contiene modelos detectables (GGUF/Safetensors/Bin).[/yellow]")
        return

    # Consolidar
    consolidated_models = consolidate_files(target_files)

    # Tabla
    table = Table(title=f"Análisis VRAM [{mode}]: {repo_id}")
    table.add_column("Archivo / Grupo", style="cyan")
    table.add_column("Tamaño Total", justify="right")
    table.add_column("VRAM Req (+Ctx)", justify="right", style="magenta")
    table.add_column("Local", justify="center")
    table.add_column("Estado")

    recommendation = None
    best_score = -1
    
    # Ordenar: 
    # - Si es GGUF, preferimos el más grande que quepa (mayor calidad).
    # - Si es Safetensors, generalmente solo hay uno o pocos (fp16/fp32), ordenamos por tamaño igual.
    sorted_models = sorted(consolidated_models, key=lambda x: x['total_size'])

    for m in sorted_models:
        size_gb = m['total_size'] / (1024**3)
        
        # Cálculo de VRAM:
        # GGUF: Tamaño real + 15% contexto
        # Safetensors: Tamaño real (FP16/32) + 20% contexto/activaciones (suelen gastar más RAM al cargar)
        overhead = 1.15 if ".gguf" in m['type'] else 1.20
        req_vram = size_gb * overhead
        
        all_parts_local = all(f in local_files for f in m['real_files'])
        local_icon = "✅" if all_parts_local else ""

        # Estado
        if vram_sys >= req_vram:
            status = "[green]GPU (Nativo)[/green]"
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
            f"Mejor opción viable: [bold blue]{recommendation}[/bold blue]",
            title="🎯 Recomendación", border_style="green"
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    repo = args.model.replace("https://huggingface.co/", "")
    analyze_model(repo)