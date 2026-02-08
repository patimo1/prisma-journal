# PrismA - Secure Journal App

A local-first journaling web application with AI-powered insights, semantic search, and voice-to-text input.

## Features

- **Rich Journal Entries** - Write and organize journal entries with mood tracking and tags
- **AI Analysis** - Get AI-powered insights, reframes, and prompts via local LLM (Ollama)
- **Semantic Search** - Find related entries using ChromaDB vector search
- **Voice Input** - Dictate entries using Whisper speech-to-text
- **Image Generation** - Optional Stable Diffusion integration for entry artwork
- **Fully Local** - All data stays on your machine

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running with a model pulled (e.g. `ollama pull llama3.2`)
- FFmpeg (required by Whisper for audio processing)

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app (development with hot reload)
python app/app.py
```

The app runs at `http://localhost:5000`.

## Configuration

Set environment variables or edit `config.py`:

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `dev-secret-key-change-in-production` | Flask session secret |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.2` | LLM model name |
| `WHISPER_MODEL` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `SD_API_URL` | `http://localhost:7860` | Stable Diffusion API |
| `SD_ENABLED` | `false` | Enable image generation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model for embeddings |
