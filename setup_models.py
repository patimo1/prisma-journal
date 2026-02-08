#!/usr/bin/env python3
"""
Model Setup Script for PrismA - Secure Journal App

This script checks for and downloads the required models:
1. Ollama LLM model
2. Whisper speech-to-text model
3. Sentence-transformers embedding model

Run this after installing requirements.txt to ensure all models are ready.
"""

import os
import sys
import subprocess
import shutil

# Add project root to path for config access
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import Config
except ImportError:
    print("Warning: Could not load config.py, using defaults")
    class Config:
        OLLAMA_MODEL = "llama3.2"
        WHISPER_MODEL = "base"
        EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_status(name, status, message=""):
    symbol = "✓" if status else "✗"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    print(f"  {color}{symbol}{reset} {name}: {message}")


def check_ollama():
    """Check if Ollama is installed and has the required model."""
    print_header("Checking Ollama")

    # Check if ollama command exists
    if not shutil.which("ollama"):
        print_status("Ollama CLI", False, "Not found in PATH")
        print("\n  To install Ollama:")
        print("    1. Visit https://ollama.ai/")
        print("    2. Download and install for your platform")
        print("    3. Run 'ollama serve' in a terminal")
        return False

    print_status("Ollama CLI", True, "Found")

    # Check if ollama is running
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            raise Exception("Not responding")
        print_status("Ollama Server", True, "Running at localhost:11434")
        models = resp.json().get("models", [])
    except Exception as e:
        print_status("Ollama Server", False, f"Not running ({e})")
        print("\n  Start Ollama with: ollama serve")
        return False

    # Check for required model
    model_name = Config.OLLAMA_MODEL
    model_found = any(m.get("name", "").startswith(model_name) for m in models)

    if model_found:
        print_status(f"Model '{model_name}'", True, "Available")
        return True
    else:
        print_status(f"Model '{model_name}'", False, "Not found")
        print(f"\n  Downloading model '{model_name}'...")

        try:
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=False,
                text=True
            )
            if result.returncode == 0:
                print_status(f"Model '{model_name}'", True, "Downloaded successfully")
                return True
            else:
                print_status(f"Model '{model_name}'", False, "Download failed")
                return False
        except Exception as e:
            print_status(f"Model '{model_name}'", False, f"Error: {e}")
            return False


def check_whisper():
    """Check if Whisper is installed and download model if needed."""
    print_header("Checking Whisper")

    try:
        import whisper
        print_status("Whisper package", True, "Installed")
    except ImportError:
        print_status("Whisper package", False, "Not installed")
        print("\n  Install with: pip install openai-whisper")
        return False

    # Check for FFmpeg
    if not shutil.which("ffmpeg"):
        print_status("FFmpeg", False, "Not found in PATH")
        print("\n  FFmpeg is required for audio processing.")
        print("  Install FFmpeg:")
        print("    - Windows: choco install ffmpeg")
        print("    - Mac: brew install ffmpeg")
        print("    - Linux: apt install ffmpeg")
        return False

    print_status("FFmpeg", True, "Found")

    # Download Whisper model
    model_name = Config.WHISPER_MODEL
    print(f"\n  Loading Whisper model '{model_name}'...")
    print("  (This downloads the model on first run, ~150MB for 'base')")

    try:
        model = whisper.load_model(model_name)
        print_status(f"Model '{model_name}'", True, "Loaded successfully")
        del model  # Free memory
        return True
    except Exception as e:
        print_status(f"Model '{model_name}'", False, f"Error: {e}")
        return False


def check_embeddings():
    """Check if sentence-transformers is installed and download model."""
    print_header("Checking Sentence-Transformers")

    try:
        from sentence_transformers import SentenceTransformer
        print_status("sentence-transformers", True, "Installed")
    except ImportError:
        print_status("sentence-transformers", False, "Not installed")
        print("\n  Install with: pip install sentence-transformers")
        return False

    # Download embedding model
    model_name = Config.EMBEDDING_MODEL
    # Extract just the model name if it's a full path
    short_name = model_name.split("/")[-1] if "/" in model_name else model_name

    print(f"\n  Loading embedding model '{short_name}'...")
    print("  (This downloads the model on first run, ~80MB)")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        print_status(f"Model '{short_name}'", True, "Loaded successfully")
        del model  # Free memory
        return True
    except Exception as e:
        print_status(f"Model '{short_name}'", False, f"Error: {e}")
        return False


def check_chromadb():
    """Check if ChromaDB is installed."""
    print_header("Checking ChromaDB")

    try:
        import chromadb
        print_status("chromadb", True, "Installed")
        return True
    except ImportError:
        print_status("chromadb", False, "Not installed")
        print("\n  Install with: pip install chromadb")
        return False


def check_stable_diffusion():
    """Check if Stable Diffusion WebUI is running (optional)."""
    print_header("Checking Stable Diffusion (Optional)")

    try:
        import requests
        resp = requests.get("http://localhost:7860/sdapi/v1/sd-models", timeout=5)
        if resp.status_code == 200:
            models = resp.json()
            print_status("SD WebUI", True, f"Running with {len(models)} model(s)")
            return True
        else:
            raise Exception("Not responding correctly")
    except Exception:
        print_status("SD WebUI", False, "Not running (optional feature)")
        print("\n  Stable Diffusion is optional. To enable:")
        print("    1. Install AUTOMATIC1111 WebUI")
        print("    2. Run with --api flag")
        print("    3. Set SD_ENABLED=true in .env")
        return False


def main():
    print("\n" + "="*60)
    print("  PrismA - Secure Journal App - Model Setup")
    print("="*60)

    results = {
        "Ollama": check_ollama(),
        "Whisper": check_whisper(),
        "Embeddings": check_embeddings(),
        "ChromaDB": check_chromadb(),
        "Stable Diffusion": check_stable_diffusion(),
    }

    print_header("Summary")

    required_ok = all([results["Ollama"], results["Whisper"],
                       results["Embeddings"], results["ChromaDB"]])

    for name, status in results.items():
        optional = " (optional)" if name == "Stable Diffusion" else ""
        print_status(name + optional, status, "Ready" if status else "Not ready")

    print()

    if required_ok:
        print("  All required components are ready!")
        print("  Run the app with: python app/app.py")
        return 0
    else:
        print("  Some required components are missing.")
        print("  Please install them and run this script again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
