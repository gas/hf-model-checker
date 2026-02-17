# HF Model Checker (Modificado)

Herramienta de línea de comandos para el análisis de requisitos de hardware y validación de modelos de Hugging Face, con soporte específico para formato GGUF.

## Funcionalidades

* **Escaneo de caché local:** Identifica modelos ya descargados en `~/.cache/huggingface`.
* **Soporte para Sharding:** Calcula el tamaño total de modelos divididos en varios archivos (ej: `00001-of-00005.gguf`).
* **Cálculo de VRAM:** * Estima el consumo sumando el peso del modelo y un margen del 15% para el contexto (KV Cache).
* Excluye archivos auxiliares (`imatrix`, `mmproj`) del cálculo de peso del modelo.
* Determina el estado de ejecución: Full GPU, Híbrido (CPU+GPU) o insuficiente.


* **Navegador Interactivo:** Interfaz para búsqueda de modelos por nombre o categorías.
* **Asincronía:** El escaneo de archivos locales no bloquea la interfaz de usuario.

## Requisitos

* Python 3.8+
* Dependencias: `questionary`, `rich`, `huggingface_hub`, `psutil` y `torch`.
* Torch es necesario para la detección automática de VRAM en hardware NVIDIA/AMD.

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/gas/hf-model-checker.git
cd hf-model-checker

```


2. Instalar dependencias:
```bash
pip install -r requirements.txt

```



## Uso

### Interfaz interactiva

Permite buscar modelos y explorar el historial o categorías:

```bash
python hf_navigator.py

```

### Análisis directo

Para analizar un modelo específico mediante URL o ID de Hugging Face:

```bash
python hf_model_checker.py --model unsloth/Llama-3-8B-Instruct-GGUF

```

## Salida de datos

El script devuelve una tabla con los siguientes datos por cada cuantización:

* **Tamaño:** Peso real en disco.
* **VRAM Req:** Memoria de video estimada incluyendo el margen de contexto.
* **Local:** Indicador de si el archivo existe en el sistema.
* **Estado:** Capacidad de carga en el hardware detectado.

## Licencia

Basado en el proyecto original de Adversing. Distribuido bajo licencia MIT.