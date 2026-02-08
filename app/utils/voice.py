"""Voice transcription utilities with comprehensive error handling.

Provides Whisper-based audio transcription with:
- Audio file validation (size, format, duration)
- Graceful degradation when Whisper unavailable
- Detailed error messages with user guidance
- Progress tracking support
"""

import logging
import os
import tempfile

from config import Config

log = logging.getLogger(__name__)

_whisper_model = None

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------

# Maximum audio file size (50 MB)
MAX_AUDIO_SIZE_MB = 50
MAX_AUDIO_SIZE_BYTES = MAX_AUDIO_SIZE_MB * 1024 * 1024

# Maximum recording duration (10 minutes)
MAX_DURATION_MINUTES = 10
MAX_DURATION_SECONDS = MAX_DURATION_MINUTES * 60

# Supported audio formats
SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4", ".aac"}

# Minimum audio size (to detect empty recordings)
MIN_AUDIO_SIZE_BYTES = 1000  # 1 KB


# ---------------------------------------------------------------------------
# Audio Validation
# ---------------------------------------------------------------------------


def validate_audio_file(audio_bytes, filename="recording.webm"):
    """Validate an audio file before transcription.

    Args:
        audio_bytes: The audio file content as bytes
        filename: Original filename for format detection

    Returns:
        dict with:
            - valid: bool
            - error: str (only if invalid)
            - metadata: dict (only if valid) containing size_mb, format, etc.
    """
    if audio_bytes is None:
        return {"valid": False, "error": "No audio data provided"}

    # Check if it's bytes
    if not isinstance(audio_bytes, (bytes, bytearray)):
        return {"valid": False, "error": "Invalid audio data format"}

    size_bytes = len(audio_bytes)

    # Check minimum size (empty or corrupted recording)
    if size_bytes < MIN_AUDIO_SIZE_BYTES:
        return {
            "valid": False,
            "error": "Recording is too short or empty. Please record at least a few seconds of audio.",
        }

    # Check maximum size
    size_mb = size_bytes / (1024 * 1024)
    if size_bytes > MAX_AUDIO_SIZE_BYTES:
        return {
            "valid": False,
            "error": f"Audio file too large ({size_mb:.1f} MB). Maximum allowed is {MAX_AUDIO_SIZE_MB} MB. Try a shorter recording or lower quality.",
        }

    # Check file format
    ext = os.path.splitext(filename)[1].lower() if filename else ""
    if not ext:
        ext = ".webm"  # Default for browser recordings

    if ext not in SUPPORTED_FORMATS:
        return {
            "valid": False,
            "error": f"Unsupported audio format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}",
        }

    # Estimate duration (very rough, based on typical compression)
    # WebM/Opus: ~6 KB/s, MP3: ~16 KB/s, WAV: ~176 KB/s
    bitrate_estimates = {
        ".webm": 6 * 1024,   # 6 KB/s
        ".mp3": 16 * 1024,   # 16 KB/s (128 kbps)
        ".wav": 176 * 1024,  # 176 KB/s (44.1 kHz 16-bit stereo)
        ".m4a": 16 * 1024,
        ".ogg": 12 * 1024,
        ".flac": 80 * 1024,
        ".mp4": 16 * 1024,
        ".aac": 16 * 1024,
    }
    estimated_bitrate = bitrate_estimates.get(ext, 16 * 1024)
    estimated_duration_seconds = size_bytes / estimated_bitrate

    metadata = {
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 2),
        "format": ext,
        "estimated_duration_seconds": round(estimated_duration_seconds, 1),
        "estimated_duration_minutes": round(estimated_duration_seconds / 60, 1),
    }

    # Warn about long recordings
    if estimated_duration_seconds > MAX_DURATION_SECONDS:
        return {
            "valid": False,
            "error": f"Recording appears to be ~{metadata['estimated_duration_minutes']:.0f} minutes long. "
                     f"Maximum recommended is {MAX_DURATION_MINUTES} minutes. "
                     "Long recordings may fail or take a very long time. Consider splitting into shorter segments.",
            "metadata": metadata,
        }

    return {"valid": True, "metadata": metadata}


# ---------------------------------------------------------------------------
# Whisper Model Management
# ---------------------------------------------------------------------------


def get_whisper_model():
    """Lazy-load the Whisper model. Returns None if Whisper is unavailable."""
    global _whisper_model

    if _whisper_model is not None:
        return _whisper_model

    from utils.services import status

    if not status.whisper:
        log.warning("Whisper unavailable: %s", status.whisper_message)
        return None

    try:
        import whisper

        log.info("Loading Whisper model '%s' ...", Config.WHISPER_MODEL)
        _whisper_model = whisper.load_model(Config.WHISPER_MODEL)
        log.info("Whisper model '%s' loaded successfully.", Config.WHISPER_MODEL)
        return _whisper_model

    except ImportError:
        log.error("Whisper package not installed")
        return None
    except Exception as e:
        log.error("Failed to load Whisper model: %s", e)
        return None


def get_whisper_status():
    """Get detailed status of Whisper availability.

    Returns:
        dict with:
            - available: bool
            - message: str (status description)
            - model_loaded: bool
            - model_name: str
    """
    from utils.services import status

    result = {
        "available": status.whisper,
        "message": status.whisper_message,
        "model_loaded": _whisper_model is not None,
        "model_name": Config.WHISPER_MODEL,
    }

    if not status.whisper:
        result["guidance"] = (
            "To enable voice transcription:\n"
            "1. Install Whisper: pip install openai-whisper\n"
            "2. Install ffmpeg:\n"
            "   - Windows: winget install ffmpeg\n"
            "   - Mac: brew install ffmpeg\n"
            "   - Linux: apt install ffmpeg"
        )

    return result


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def transcribe_audio(audio_bytes, filename="recording.webm", language=None):
    """Transcribe audio bytes to text using Whisper.

    Args:
        audio_bytes: Audio file content as bytes
        filename: Original filename (for format detection)
        language: Optional language code (e.g., 'en', 'es'). None for auto-detect.

    Returns:
        dict on success: {
            "text": str,
            "language": str,
            "confidence": float (0.0-1.0),
            "duration_seconds": float,
            "metadata": dict
        }
        dict on error: {
            "error": str,
            "error_type": str ("validation", "transcription", "service"),
            "guidance": str (optional user guidance)
        }
    """
    # Validate audio file first
    validation = validate_audio_file(audio_bytes, filename)
    if not validation.get("valid"):
        return {
            "error": validation.get("error", "Invalid audio file"),
            "error_type": "validation",
            "guidance": "Try recording again with a working microphone in a quiet environment.",
        }

    metadata = validation.get("metadata", {})

    # Check Whisper availability
    model = get_whisper_model()
    if model is None:
        whisper_status = get_whisper_status()
        return {
            "error": "Voice transcription is not available.",
            "error_type": "service",
            "guidance": whisper_status.get("guidance", "Install openai-whisper and ffmpeg."),
        }

    # Create temp file for Whisper
    suffix = os.path.splitext(filename)[1] or ".webm"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Transcribe with optional language hint
        log.info(
            "Transcribing audio: %s, %.1f MB, estimated %.0f seconds",
            suffix,
            metadata.get("size_mb", 0),
            metadata.get("estimated_duration_seconds", 0),
        )

        transcribe_options = {}
        if language and language != "auto":
            transcribe_options["language"] = language

        result = model.transcribe(tmp_path, **transcribe_options)
        text = result.get("text", "").strip()
        detected_language = result.get("language", "unknown")

        # Compute confidence from average log-probability across segments
        # avg_logprob is typically between -1.0 (low) and 0.0 (high)
        segments = result.get("segments", [])
        if segments:
            avg = sum(s.get("avg_logprob", -1.0) for s in segments) / len(segments)
            confidence = round(min(1.0, max(0.0, 1.0 + avg)), 2)
            # Calculate actual duration from segments
            duration = segments[-1].get("end", 0) if segments else 0
        else:
            confidence = 0.0
            duration = 0

        # Warn about low confidence
        warning = None
        if confidence < 0.3:
            warning = "Low confidence transcription. The audio may be unclear or have background noise."
        elif not text:
            warning = "No speech detected in the audio. Make sure you're speaking clearly."

        result_dict = {
            "text": text,
            "language": detected_language,
            "confidence": confidence,
            "duration_seconds": round(duration, 1),
            "metadata": metadata,
        }

        if warning:
            result_dict["warning"] = warning

        log.info(
            "Transcription complete: %d chars, %.0fs duration, %.0f%% confidence",
            len(text),
            duration,
            confidence * 100,
        )

        return result_dict

    except Exception as e:
        error_message = str(e)
        log.error("Whisper transcription failed: %s", error_message)

        # Provide specific guidance based on error type
        if "ffmpeg" in error_message.lower():
            return {
                "error": "Audio processing failed. ffmpeg is not installed.",
                "error_type": "transcription",
                "guidance": "Install ffmpeg: Windows: winget install ffmpeg, Mac: brew install ffmpeg",
            }
        elif "WinError 2" in error_message or "cannot find the file" in error_message.lower() or "No such file" in error_message:
            # Windows/Unix error when ffmpeg executable is not found
            return {
                "error": "ffmpeg is not installed. It's required for audio processing.",
                "error_type": "transcription",
                "guidance": "Install ffmpeg: On Windows run 'winget install ffmpeg' in a terminal, then restart the app.",
            }
        elif "memory" in error_message.lower() or "cuda" in error_message.lower():
            return {
                "error": "Not enough memory for transcription. Try a shorter recording.",
                "error_type": "transcription",
                "guidance": "Try using a smaller Whisper model in Settings (e.g., 'tiny' or 'base').",
            }
        else:
            return {
                "error": f"Transcription failed: {error_message}",
                "error_type": "transcription",
                "guidance": "Try recording again. If the problem persists, check the System Status page.",
            }

    finally:
        # Clean up temp file
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Browser Permission Guidance
# ---------------------------------------------------------------------------


def get_microphone_permission_guidance():
    """Return guidance for users who have microphone permission issues.

    Returns:
        dict with browser-specific instructions
    """
    return {
        "title": "Microphone Access Required",
        "general": (
            "To use voice recording, your browser needs permission to access your microphone. "
            "When prompted, click 'Allow' to grant access."
        ),
        "browsers": {
            "chrome": (
                "In Chrome:\n"
                "1. Click the lock/info icon in the address bar\n"
                "2. Find 'Microphone' and set it to 'Allow'\n"
                "3. Refresh the page"
            ),
            "firefox": (
                "In Firefox:\n"
                "1. Click the lock icon in the address bar\n"
                "2. Click the arrow next to the connection info\n"
                "3. Click 'More Information' > 'Permissions'\n"
                "4. Find 'Use the Microphone' and select 'Allow'"
            ),
            "safari": (
                "In Safari:\n"
                "1. Go to Safari > Settings > Websites\n"
                "2. Select 'Microphone' from the sidebar\n"
                "3. Find this website and set it to 'Allow'"
            ),
            "edge": (
                "In Edge:\n"
                "1. Click the lock icon in the address bar\n"
                "2. Find 'Microphone' and set it to 'Allow'\n"
                "3. Refresh the page"
            ),
        },
        "troubleshooting": [
            "Make sure no other app is using your microphone",
            "Check that your microphone is properly connected",
            "Try a different browser if issues persist",
            "On some systems, you may need to grant permission in system settings",
        ],
    }
