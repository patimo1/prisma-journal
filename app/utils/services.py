"""Service initialization and health checks.

Provides a central place to probe external dependencies (Ollama, Whisper,
ChromaDB, sentence-transformers, Stable Diffusion) and report which are
available.  The app continues in degraded mode when services are missing —
features that depend on an unavailable service are simply disabled at runtime.
"""

import logging
import sys
import os
import time
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import requests
from config import Config
from .i18n import translate as _t

# Optional import for system diagnostics
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Service status registry
# ---------------------------------------------------------------------------


class ServiceStatus:
    """Tracks availability of each optional service."""

    def __init__(self):
        self.ollama: bool = False
        self.ollama_message: str = ""
        self.lmstudio: bool = False
        self.lmstudio_message: str = ""
        self.whisper: bool = False
        self.whisper_message: str = ""
        self.chromadb: bool = False
        self.chromadb_message: str = ""
        self.embeddings: bool = False
        self.embeddings_message: str = ""
        self.stable_diffusion: bool = False
        self.sd_message: str = ""

    def summary(self) -> list[dict]:
        """Return a list of dicts suitable for display in the settings page."""
        active_marker = " [ACTIVE]" if Config.LLM_PROVIDER == "ollama" else ""
        lms_active_marker = " [ACTIVE]" if Config.LLM_PROVIDER == "lmstudio" else ""
        
        return [
            {
                "name": f"Ollama (LLM){active_marker}",
                "available": self.ollama,
                "message": self.ollama_message,
            },
            {
                "name": f"LM Studio (LLM){lms_active_marker}",
                "available": self.lmstudio,
                "message": self.lmstudio_message,
            },
            {
                "name": "Whisper (Voice)",
                "available": self.whisper,
                "message": self.whisper_message,
            },
            {
                "name": "ChromaDB (Vector Store)",
                "available": self.chromadb,
                "message": self.chromadb_message,
            },
            {
                "name": "Sentence-Transformers (Embeddings)",
                "available": self.embeddings,
                "message": self.embeddings_message,
            },
            {
                "name": "Stable Diffusion (Images)",
                "available": self.stable_diffusion,
                "message": self.sd_message,
            },
        ]


# Module-level singleton — populated by init_services()
status = ServiceStatus()


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_ollama() -> tuple[bool, str]:
    """Ping the Ollama API.  Returns (ok, message)."""
    url = f"{Config.OLLAMA_BASE_URL}/api/tags"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if not models:
            return True, (
                "Ollama is running but has no models pulled. "
                f"Run: ollama pull {Config.OLLAMA_MODEL}"
            )
        if not any(Config.OLLAMA_MODEL in m for m in models):
            return True, (
                f"Ollama is running but model '{Config.OLLAMA_MODEL}' not found. "
                f"Available: {', '.join(models[:5])}. "
                f"Run: ollama pull {Config.OLLAMA_MODEL}"
            )
        return True, f"Connected — model '{Config.OLLAMA_MODEL}' available"
    except requests.ConnectionError:
        return False, (
            f"Cannot connect to Ollama at {Config.OLLAMA_BASE_URL}. "
            "Install from https://ollama.ai and run: ollama serve"
        )
    except requests.Timeout:
        return False, f"Ollama at {Config.OLLAMA_BASE_URL} timed out."
    except Exception as e:
        return False, f"Ollama check failed: {e}"


def check_lmstudio() -> tuple[bool, str]:
    """Check LM Studio API via OpenAI-compatible endpoint. Returns (ok, message)."""
    url = f"{Config.LMSTUDIO_BASE_URL}/models"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = [m.get("id", "") for m in data.get("data", [])]
        if not models:
            return False, (
                f"LM Studio at {Config.LMSTUDIO_BASE_URL} is running but has no models loaded. "
                "Load a model in LM Studio."
            )
        # Show available models
        model_list = ", ".join(models[:3])
        return True, (
            f"LM Studio is running. Available models: {model_list}. "
            f"Set LMSTUDIO_MODEL to match."
        )
    except requests.ConnectionError:
        return False, (
            f"Cannot connect to LM Studio at {Config.LMSTUDIO_BASE_URL}. "
            "Ensure LM Studio is running with the local server enabled."
        )
    except requests.Timeout:
        return False, f"LM Studio at {Config.LMSTUDIO_BASE_URL} timed out."
    except Exception as e:
        return False, f"LM Studio check failed: {e}"


def check_whisper() -> tuple[bool, str]:
    """Verify that the whisper package is importable."""
    try:
        import whisper  # noqa: F401
        return True, f"Installed — will use '{Config.WHISPER_MODEL}' model (loaded on first use)"
    except ImportError:
        return False, (
            "openai-whisper is not installed. "
            "Run: pip install openai-whisper  (also requires ffmpeg)"
        )


def load_whisper_model():
    """Actually load the Whisper model into memory.  Heavy — call only when
    the user first requests transcription, not at startup."""
    import whisper
    return whisper.load_model(Config.WHISPER_MODEL)


def check_chromadb() -> tuple[bool, str]:
    """Verify ChromaDB can initialise a persistent client."""
    try:
        import chromadb  # noqa: F401
        os.makedirs(Config.CHROMA_PATH, exist_ok=True)
        client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
        client.get_or_create_collection(
            name=Config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        return True, f"Connected — collection '{Config.CHROMA_COLLECTION}' at {Config.CHROMA_PATH}"
    except ImportError:
        return False, "chromadb is not installed. Run: pip install chromadb"
    except Exception as e:
        return False, f"ChromaDB initialisation failed: {e}"


def check_embeddings() -> tuple[bool, str]:
    """Verify sentence-transformers is available (model loaded on demand)."""
    try:
        import sentence_transformers  # noqa: F401
        return True, f"Installed — model '{Config.EMBEDDING_MODEL}' (downloaded on first use)"
    except ImportError:
        return False, (
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )


def init_sentence_transformer():
    """Load the sentence-transformers embedding model.  Returns the model
    object or None on failure."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(Config.EMBEDDING_MODEL)
    except Exception as e:
        log.warning("Failed to load sentence-transformer model: %s", e)
        return None


def check_stable_diffusion() -> tuple[bool, str]:
    """Ping the Stable Diffusion WebUI API if enabled."""
    if not Config.SD_ENABLED:
        return False, "Disabled in configuration (SD_ENABLED=false)"
    url = f"{Config.SD_API_URL}/sdapi/v1/sd-models"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        models = resp.json()
        count = len(models) if isinstance(models, list) else 0
        return True, f"Connected — {count} model(s) available at {Config.SD_API_URL}"
    except requests.ConnectionError:
        return False, (
            f"Cannot connect to Stable Diffusion at {Config.SD_API_URL}. "
            "Start the WebUI with --api flag."
        )
    except requests.Timeout:
        return False, f"Stable Diffusion at {Config.SD_API_URL} timed out."
    except Exception as e:
        return False, f"Stable Diffusion check failed: {e}"
    """Check if FLUX model is available via Ollama."""
    if not Config.FLUX_ENABLED:
        return False, "Disabled in configuration (FLUX_ENABLED=false)"
# ---------------------------------------------------------------------------
# Aggregate initialisation
# ---------------------------------------------------------------------------


def init_services() -> ServiceStatus:
    """Run all service checks and populate the global status object.
    Called once at app startup.  Individual features check ``status.*``
    before attempting to use a service."""
    global status

    # Validate config values first
    warnings = Config.validate()
    for field, msg in warnings:
        log.warning("Config warning [%s]: %s", field, msg)

    status.ollama, status.ollama_message = check_ollama()
    status.lmstudio, status.lmstudio_message = check_lmstudio()
    status.whisper, status.whisper_message = check_whisper()
    status.chromadb, status.chromadb_message = check_chromadb()
    status.embeddings, status.embeddings_message = check_embeddings()
    status.stable_diffusion, status.sd_message = check_stable_diffusion()

    # Log a startup summary
    log.info("--- Service status ---")
    log.info("  Active LLM Provider: %s", Config.LLM_PROVIDER.upper())
    for svc in status.summary():
        level = "OK" if svc["available"] else "UNAVAILABLE"
        log.info("  %-45s [%s] %s", svc["name"], level, svc["message"])
    log.info("----------------------")

    return status


def refresh_service_status() -> ServiceStatus:
    """Re-run all checks (e.g. when user clicks 'refresh' on settings page)."""
    return init_services()


# ---------------------------------------------------------------------------
# Detailed diagnostics for System Status page
# ---------------------------------------------------------------------------


def get_detailed_status(lang="de") -> dict:
    """Get comprehensive diagnostic information for all services.

    Args:
        lang: Language code for localized messages (default: "de")
    
    Returns a dict with detailed status for each service including:
    - Connection status
    - Version/model info
    - Configuration details
    - Performance metrics
    - Setup instructions
    """
    import platform

    system_info: dict[str, Any] = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
    }

    if HAS_PSUTIL and psutil is not None:
        system_info["memory_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
        system_info["memory_available_gb"] = round(psutil.virtual_memory().available / (1024**3), 1)
        system_info["memory_percent_used"] = psutil.virtual_memory().percent
        system_info["cpu_count"] = psutil.cpu_count()

    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "system": system_info,
        "services": {},
    }

    # Ollama diagnostics
    ollama_diag = _diagnose_ollama(lang)
    diagnostics["services"]["ollama"] = ollama_diag

    # LM Studio diagnostics
    lmstudio_diag = _diagnose_lmstudio(lang)
    diagnostics["services"]["lmstudio"] = lmstudio_diag

    # Whisper diagnostics
    whisper_diag = _diagnose_whisper(lang)
    diagnostics["services"]["whisper"] = whisper_diag

    # ChromaDB diagnostics
    chromadb_diag = _diagnose_chromadb(lang)
    diagnostics["services"]["chromadb"] = chromadb_diag

    # Embeddings diagnostics
    embeddings_diag = _diagnose_embeddings(lang)
    diagnostics["services"]["embeddings"] = embeddings_diag

    # Stable Diffusion diagnostics
    sd_diag = _diagnose_stable_diffusion(lang)
    diagnostics["services"]["stable_diffusion"] = sd_diag

    # Database diagnostics
    db_diag = _diagnose_database(lang)
    diagnostics["services"]["database"] = db_diag

    return diagnostics


def _diagnose_ollama(lang="de") -> dict:
    """Get detailed Ollama diagnostics."""
    diag = {
        "name": "Ollama (LLM)",
        "icon": "brain",
        "available": False,
        "message": "",
        "details": {},
        "setup_instructions": None,
        "config": {
            "endpoint": Config.OLLAMA_BASE_URL,
            "model": Config.OLLAMA_MODEL,
            "timeout": Config.OLLAMA_TIMEOUT,
        },
    }

    url = f"{Config.OLLAMA_BASE_URL}/api/tags"
    try:
        start = time.time()
        resp = requests.get(url, timeout=5)
        latency_ms = round((time.time() - start) * 1000, 1)
        resp.raise_for_status()

        models = resp.json().get("models", [])
        model_names = [m["name"] for m in models]

        diag["available"] = True
        diag["details"]["latency_ms"] = latency_ms
        diag["details"]["models_available"] = len(model_names)
        diag["details"]["model_list"] = model_names[:10]  # Limit to 10

        if not model_names:
            diag["message"] = "Connected, but no models installed"
            diag["setup_instructions"] = f"Run: ollama pull {Config.OLLAMA_MODEL}"
        elif not any(Config.OLLAMA_MODEL in m for m in model_names):
            diag["message"] = f"Connected, but model '{Config.OLLAMA_MODEL}' not found"
            diag["setup_instructions"] = f"Run: ollama pull {Config.OLLAMA_MODEL}"
            diag["details"]["configured_model_available"] = False
        else:
            diag["message"] = f"Connected - {len(model_names)} model(s) available"
            diag["details"]["configured_model_available"] = True

            # Try to get model info
            for m in models:
                if Config.OLLAMA_MODEL in m["name"]:
                    diag["details"]["active_model"] = {
                        "name": m["name"],
                        "size": m.get("size"),
                        "modified_at": m.get("modified_at"),
                    }
                    break

    except requests.ConnectionError:
        diag["message"] = _t("service.ollama.cannot_connect", lang)
        diag["setup_instructions"] = (
            f"1. {_t('setup.ollama.step1', lang)}\n"
            f"2. {_t('setup.ollama.step2', lang)}\n"
            f"3. {_t('setup.ollama.step3', lang, model=Config.OLLAMA_MODEL)}"
        )
    except requests.Timeout:
        diag["message"] = _t("service.ollama.connection_timeout", lang)
        diag["setup_instructions"] = _t('setup.ollama.overloaded', lang)
    except Exception as e:
        diag["message"] = f"Error: {str(e)}"

    return diag


def _diagnose_lmstudio(lang="de") -> dict:
    """Get detailed LM Studio diagnostics."""
    diag = {
        "name": "LM Studio (LLM)",
        "icon": "cpu",
        "available": False,
        "message": "",
        "details": {},
        "setup_instructions": None,
        "config": {
            "endpoint": Config.LMSTUDIO_BASE_URL,
            "model": Config.LMSTUDIO_MODEL,
            "provider_active": Config.LLM_PROVIDER == "lmstudio",
        },
    }

    url = f"{Config.LMSTUDIO_BASE_URL}/models"
    try:
        start = time.time()
        resp = requests.get(url, timeout=5)
        latency_ms = round((time.time() - start) * 1000, 1)
        resp.raise_for_status()

        payload = resp.json()
        models = payload.get("data", []) if isinstance(payload, dict) else []
        model_ids = [m.get("id", "") for m in models if isinstance(m, dict)]

        diag["details"]["latency_ms"] = latency_ms
        diag["details"]["models_available"] = len(model_ids)
        diag["details"]["model_list"] = model_ids[:10]

        if not model_ids:
            diag["message"] = _t("service.lmstudio.no_models_loaded", lang)
            diag["setup_instructions"] = _t("setup.lmstudio.load_model_first", lang)
            return diag

        configured_ok = any(Config.LMSTUDIO_MODEL == model_id for model_id in model_ids)
        diag["details"]["configured_model_available"] = configured_ok

        diag["available"] = True
        if configured_ok:
            diag["message"] = _t("service.lmstudio.connected_model_available", lang, model=Config.LMSTUDIO_MODEL)
        else:
            diag["message"] = _t("service.lmstudio.model_not_found", lang, model=Config.LMSTUDIO_MODEL)
            diag["setup_instructions"] = _t("setup.lmstudio.set_model", lang)

    except requests.ConnectionError:
        diag["message"] = _t("service.lmstudio.cannot_connect", lang)
        diag["setup_instructions"] = (
            f"1. {_t('setup.lmstudio.step1', lang)}\n"
            f"2. {_t('setup.lmstudio.step2', lang)}\n"
            f"3. {_t('setup.lmstudio.step4', lang)}"
        )
    except requests.Timeout:
        diag["message"] = _t("service.ollama.connection_timeout", lang)
        diag["setup_instructions"] = _t('setup.ollama.overloaded', lang)
    except Exception as e:
        diag["message"] = f"Error: {str(e)}"
        diag["setup_instructions"] = (
            f"1. {_t('setup.lmstudio.step1', lang)}\n"
            f"2. {_t('setup.lmstudio.step3', lang)}\n"
            f"3. {_t('setup.lmstudio.step2', lang)}\n"
            f"4. {_t('setup.lmstudio.step4', lang)}"
        )

    return diag


def _diagnose_whisper(lang="de") -> dict:
    """Get detailed Whisper diagnostics."""
    diag = {
        "name": "Whisper (Voice)",
        "icon": "microphone",
        "available": False,
        "message": "",
        "details": {},
        "setup_instructions": None,
        "config": {
            "model_size": Config.WHISPER_MODEL,
        },
    }

    try:
        import whisper
        diag["available"] = True
        diag["message"] = f"Installed - '{Config.WHISPER_MODEL}' model ready"
        diag["details"]["whisper_version"] = getattr(whisper, "__version__", "unknown")
        diag["details"]["model_loaded"] = _whisper_model is not None

        # Model size estimates
        model_sizes = {
            "tiny": "39 MB",
            "base": "74 MB",
            "small": "244 MB",
            "medium": "769 MB",
            "large": "1550 MB",
        }
        diag["details"]["estimated_model_size"] = model_sizes.get(Config.WHISPER_MODEL, "unknown")

        # Check if ffmpeg is available
        import shutil
        diag["details"]["ffmpeg_available"] = shutil.which("ffmpeg") is not None
        if not diag["details"]["ffmpeg_available"]:
            diag["message"] += " (ffmpeg not found - some formats may fail)"

    except ImportError:
        diag["message"] = "openai-whisper not installed"
        diag["setup_instructions"] = (
            "1. Install Whisper: pip install openai-whisper\n"
            "2. Install ffmpeg:\n"
            "   - Windows: winget install ffmpeg\n"
            "   - Mac: brew install ffmpeg\n"
            "   - Linux: apt install ffmpeg"
        )

    return diag


def _diagnose_chromadb(lang="de") -> dict:
    """Get detailed ChromaDB diagnostics."""
    diag = {
        "name": "ChromaDB (Vector Store)",
        "icon": "database",
        "available": False,
        "message": "",
        "details": {},
        "setup_instructions": None,
        "config": {
            "path": Config.CHROMA_PATH,
            "collection": Config.CHROMA_COLLECTION,
        },
    }

    try:
        import chromadb
        diag["details"]["chromadb_version"] = getattr(chromadb, "__version__", "unknown")

        os.makedirs(Config.CHROMA_PATH, exist_ok=True)
        client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
        collection = client.get_or_create_collection(
            name=Config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        diag["available"] = True
        count = collection.count()
        diag["details"]["document_count"] = count
        diag["details"]["collection_name"] = Config.CHROMA_COLLECTION

        # Get disk usage
        total_size = 0
        if os.path.exists(Config.CHROMA_PATH):
            for dirpath, dirnames, filenames in os.walk(Config.CHROMA_PATH):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
        diag["details"]["disk_usage_mb"] = round(total_size / (1024 * 1024), 2)

        diag["message"] = f"Operational - {count} documents indexed"

    except ImportError:
        diag["message"] = "chromadb not installed"
        diag["setup_instructions"] = "pip install chromadb"
    except Exception as e:
        diag["message"] = f"Error: {str(e)}"
        diag["setup_instructions"] = (
            "Try deleting the ChromaDB folder and restarting:\n"
            f"rm -rf {Config.CHROMA_PATH}"
        )

    return diag


def _diagnose_embeddings(lang="de") -> dict:
    """Get detailed embeddings diagnostics."""
    diag = {
        "name": "Sentence-Transformers (Embeddings)",
        "icon": "vector",
        "available": False,
        "message": "",
        "details": {},
        "setup_instructions": None,
        "config": {
            "model": Config.EMBEDDING_MODEL,
        },
    }

    try:
        import sentence_transformers
        diag["available"] = True
        diag["details"]["version"] = getattr(sentence_transformers, "__version__", "unknown")
        diag["message"] = f"Installed - model: {Config.EMBEDDING_MODEL}"

        # Check if model is downloaded
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "torch" / "sentence_transformers"
        model_name = Config.EMBEDDING_MODEL.replace("/", "_")
        model_path = cache_dir / model_name
        diag["details"]["model_cached"] = model_path.exists() if cache_dir.exists() else "unknown"

    except ImportError:
        diag["message"] = "sentence-transformers not installed"
        diag["setup_instructions"] = "pip install sentence-transformers"

    return diag


def _diagnose_stable_diffusion(lang="de") -> dict:
    """Get detailed Stable Diffusion diagnostics."""
    diag = {
        "name": "Stable Diffusion (Images)",
        "icon": "image",
        "available": False,
        "message": "",
        "details": {},
        "setup_instructions": None,
        "config": {
            "enabled": Config.SD_ENABLED,
            "endpoint": Config.SD_API_URL,
            "default_style": Config.SD_DEFAULT_STYLE,
        },
    }

    if not Config.SD_ENABLED:
        diag["message"] = "Disabled in configuration"
        diag["details"]["reason"] = "SD_ENABLED=false"
        diag["setup_instructions"] = (
            f"{_t('setup.sd.enable_header')}\n"
            f"1. {_t('setup.sd.install')}\n"
            f"2. {_t('setup.sd.start_api')}\n"
            f"3. {_t('setup.sd.enable_env')}"
        )
        return diag

    url = f"{Config.SD_API_URL}/sdapi/v1/sd-models"
    try:
        start = time.time()
        resp = requests.get(url, timeout=5)
        latency_ms = round((time.time() - start) * 1000, 1)
        resp.raise_for_status()

        models = resp.json()
        diag["available"] = True
        diag["details"]["latency_ms"] = latency_ms
        diag["details"]["models_count"] = len(models) if isinstance(models, list) else 0

        if isinstance(models, list) and models:
            diag["details"]["models"] = [m.get("model_name", m.get("title", "unknown"))[:50] for m in models[:5]]

        diag["message"] = f"Connected - {len(models)} model(s) available"

        # Try to get current model
        try:
            options_resp = requests.get(f"{Config.SD_API_URL}/sdapi/v1/options", timeout=3)
            if options_resp.ok:
                options = options_resp.json()
                diag["details"]["current_model"] = options.get("sd_model_checkpoint", "unknown")
        except:
            pass

    except requests.ConnectionError:
        diag["message"] = "Cannot connect to Stable Diffusion WebUI"
        diag["setup_instructions"] = (
            "1. Install AUTOMATIC1111 WebUI\n"
            "2. Start with: ./webui.sh --api\n"
            f"3. Ensure it's running at {Config.SD_API_URL}"
        )
    except requests.Timeout:
        diag["message"] = "Connection timed out"
    except Exception as e:
        diag["message"] = f"Error: {str(e)}"

    return diag
    """Get detailed FLUX diagnostics."""
    diag = {
        "name": "FLUX (AI Images)",
        "icon": "sparkles",
        "available": False,
        "message": "",
        "details": {},
        "setup_instructions": None,
        "config": {
            "enabled": Config.FLUX_ENABLED,
            "model": Config.FLUX_MODEL,
            "steps": Config.FLUX_STEPS,
            "width": Config.FLUX_WIDTH,
            "height": Config.FLUX_HEIGHT,
        },
    }

    if not Config.FLUX_ENABLED:
        diag["message"] = "Disabled in configuration"
        diag["details"]["reason"] = "FLUX_ENABLED=false"
        diag["setup_instructions"] = (
            "To enable FLUX image generation:\n"
            "1. Ensure Ollama is running\n"
            "2. Pull FLUX model: ollama pull flux-schnell\n"
            "3. Set FLUX_ENABLED=true in .env"
        )
        return diag

    url = f"{Config.OLLAMA_BASE_URL}/api/tags"
    try:
        start = time.time()
        resp = requests.get(url, timeout=5)
        latency_ms = round((time.time() - start) * 1000, 1)
        resp.raise_for_status()

        models = resp.json().get("models", [])
        model_names = [m["name"] for m in models]
        flux_models = [m for m in model_names if "flux" in m.lower()]

        diag["details"]["latency_ms"] = latency_ms
        diag["details"]["all_models"] = model_names[:10]
        diag["details"]["flux_models"] = flux_models

        if not flux_models:
            diag["message"] = "Ollama connected, but no FLUX models found"
            diag["setup_instructions"] = (
                f"Run: ollama pull {Config.FLUX_MODEL}\n"
                "Available models: https://ollama.com/library/flux"
            )
        elif not any(Config.FLUX_MODEL in m for m in model_names):
            diag["available"] = True
            diag["message"] = f"FLUX available via '{flux_models[0]}'"
            diag["details"]["using_model"] = flux_models[0]
            diag["setup_instructions"] = (
                f"To use {Config.FLUX_MODEL}: ollama pull {Config.FLUX_MODEL}"
            )
        else:
            diag["available"] = True
            diag["message"] = f"Connected - FLUX model '{Config.FLUX_MODEL}' ready"
            diag["details"]["using_model"] = Config.FLUX_MODEL

    except requests.ConnectionError:
        diag["message"] = "Cannot connect to Ollama"
        diag["setup_instructions"] = (
            "1. Install Ollama: https://ollama.ai\n"
            "2. Start the Ollama server: ollama serve\n"
            "3. Pull FLUX: ollama pull flux-schnell"
        )
    except requests.Timeout:
        diag["message"] = "Connection timed out"
    except Exception as e:
        diag["message"] = f"Error: {str(e)}"

    return diag


def _diagnose_database(lang="de") -> dict:
    """Get detailed database diagnostics."""
    import sqlite3

    diag = {
        "name": "SQLite Database",
        "icon": "storage",
        "available": False,
        "message": "",
        "details": {},
        "setup_instructions": None,
        "config": {
            "path": Config.DATABASE_PATH,
        },
    }

    try:
        if os.path.exists(Config.DATABASE_PATH):
            diag["details"]["file_size_mb"] = round(
                os.path.getsize(Config.DATABASE_PATH) / (1024 * 1024), 2
            )

            conn = sqlite3.connect(Config.DATABASE_PATH)
            cursor = conn.cursor()

            # Get SQLite version
            cursor.execute("SELECT sqlite_version()")
            diag["details"]["sqlite_version"] = cursor.fetchone()[0]

            # Count entries
            cursor.execute("SELECT COUNT(*) FROM entries")
            diag["details"]["entry_count"] = cursor.fetchone()[0]

            # Count by type
            cursor.execute("SELECT entry_type, COUNT(*) FROM entries GROUP BY entry_type")
            diag["details"]["entries_by_type"] = dict(cursor.fetchall())

            # Get table info
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            diag["details"]["tables"] = [row[0] for row in cursor.fetchall()]

            conn.close()

            diag["available"] = True
            diag["message"] = f"Operational - {diag['details']['entry_count']} entries"
        else:
            diag["message"] = "Database file not found (will be created on first use)"
            diag["available"] = True

    except Exception as e:
        diag["message"] = f"Error: {str(e)}"
        diag["setup_instructions"] = (
            "Database may be corrupted. To restore:\n"
            f"1. Backup: cp {Config.DATABASE_PATH} {Config.DATABASE_PATH}.bak\n"
            f"2. Delete: rm {Config.DATABASE_PATH}\n"
            "3. Restart the app to create a fresh database"
        )

    return diag


# Module-level cache for whisper model (loaded lazily by voice.py)
_whisper_model = None
