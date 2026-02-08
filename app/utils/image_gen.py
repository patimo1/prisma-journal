import base64
import logging
import os
import io
import math
import random
import time

import requests
from config import Config

log = logging.getLogger(__name__)


def save_image(base64_data, filename):
    """Decode base64 image data and save to the uploads folder."""
    upload_dir = Config.UPLOAD_FOLDER
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, filename)
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(base64_data))
    return f"/static/uploads/{filename}"


def _normalize_style(style):
    if not style:
        return "minimalist abstract"
    cleaned = str(style).strip().lower()
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    aliases = {
        "minimalist": "minimalist abstract",
        "abstract": "minimalist abstract",
        "nature inspired": "nature inspired",
        "nature": "nature inspired",
        "urban": "urban architectural",
        "architectural": "urban architectural",
        "cosmic": "cosmic space",
        "space": "cosmic space",
        "line": "line art",
        "lineart": "line art",
        "collage art": "collage",
    }
    return aliases.get(cleaned, cleaned)


_EMOTION_COLORS = {
    "joy": "#f6c453",
    "trust": "#7ccaa5",
    "fear": "#7b6bd6",
    "surprise": "#f28bb3",
    "sadness": "#6ba4d6",
    "disgust": "#5ca36c",
    "anger": "#ea6a5b",
    "anticipation": "#f2a654",
}

_EMOTION_COLOR_WORDS = {
    "joy": "golden yellow",
    "trust": "soft green",
    "fear": "deep violet",
    "surprise": "rose pink",
    "sadness": "cool blue",
    "disgust": "earthy green",
    "anger": "warm red",
    "anticipation": "amber",
}


def _hex_to_rgb(hex_value):
    hex_value = hex_value.lstrip("#")
    return tuple(int(hex_value[i:i + 2], 16) for i in (0, 2, 4))


def _blend(c1, c2, alpha=0.5):
    return tuple(int(c1[i] * (1 - alpha) + c2[i] * alpha) for i in range(3))


def _derive_palette(emotions):
    base = ["#f2f2f2", "#d6e4f0", "#f4e3d7"]
    if not emotions:
        return [ _hex_to_rgb(c) for c in base ]
    names = []
    for em in emotions:
        if isinstance(em, dict):
            name = em.get("emotion")
        else:
            name = em
        if name and name not in names:
            names.append(name)
    colors = []
    for name in names:
        color_hex = _EMOTION_COLORS.get(str(name).lower())
        if color_hex:
            colors.append(_hex_to_rgb(color_hex))
    if not colors:
        colors = [_hex_to_rgb(c) for c in base]
    while len(colors) < 3:
        colors.append(_blend(colors[-1], _hex_to_rgb(base[len(colors) % len(base)]), 0.4))
    return colors[:5]


def build_artwork_prompt(style, themes=None, emotions=None, sentiment=None):
    style = _normalize_style(style)
    themes = themes or []
    emotions = emotions or []
    emotion_names = []
    for em in emotions:
        name = em.get("emotion") if isinstance(em, dict) else str(em)
        if name and name not in emotion_names:
            emotion_names.append(name)
    color_words = [
        _EMOTION_COLOR_WORDS.get(name.lower())
        for name in emotion_names
        if _EMOTION_COLOR_WORDS.get(name.lower())
    ]
    if not color_words:
        color_words = ["soft neutrals"]
    theme_phrase = ", ".join(themes[:3]) if themes else "reflection"
    sentiment_phrase = sentiment or "neutral"
    return (
        f"abstract composition, {style} style, {', '.join(color_words)}, "
        f"theme of {theme_phrase}, {sentiment_phrase} mood"
    )


def generate_algorithmic_art(style, emotions=None, themes=None, seed=None, size=768):
    """Generate algorithmic abstract artwork (Pillow-based).

    Returns image bytes (PNG) or None if Pillow is unavailable.
    """
    try:
        from PIL import Image, ImageDraw, ImageFilter
    except Exception as exc:
        log.warning("Pillow not available for algorithmic art: %s", exc)
        return None

    style = _normalize_style(style)
    rng = random.Random(seed)
    palette = _derive_palette(emotions)
    base = _blend(palette[0], (245, 245, 245), 0.6)
    image = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(image, "RGBA")

    def rand_color(alpha=180):
        color = rng.choice(palette)
        return (color[0], color[1], color[2], alpha)

    if style == "minimalist abstract":
        for _ in range(rng.randint(4, 7)):
            x0 = rng.randint(-50, size - 100)
            y0 = rng.randint(-50, size - 100)
            x1 = x0 + rng.randint(size // 4, size // 2)
            y1 = y0 + rng.randint(size // 4, size // 2)
            draw.rectangle([x0, y0, x1, y1], fill=rand_color(110))
        for _ in range(8):
            cx = rng.randint(0, size)
            cy = rng.randint(0, size)
            r = rng.randint(20, 80)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=rand_color(140), width=2)

    elif style == "watercolor":
        layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer, "RGBA")
        for _ in range(rng.randint(8, 14)):
            cx = rng.randint(0, size)
            cy = rng.randint(0, size)
            r = rng.randint(size // 8, size // 3)
            layer_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rand_color(90))
        layer = layer.filter(ImageFilter.GaussianBlur(radius=24))
        image = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")

    elif style == "geometric":
        for _ in range(rng.randint(10, 18)):
            points = [(rng.randint(0, size), rng.randint(0, size)) for _ in range(3)]
            draw.polygon(points, fill=rand_color(160))
        for _ in range(6):
            x0 = rng.randint(0, size)
            y0 = rng.randint(0, size)
            x1 = x0 + rng.randint(60, 200)
            y1 = y0 + rng.randint(60, 200)
            draw.rectangle([x0, y0, x1, y1], outline=rand_color(200), width=3)

    elif style == "surreal":
        for y in range(size):
            ratio = y / size
            color = _blend(palette[0], palette[-1], ratio)
            draw.line([(0, y), (size, y)], fill=color)
        for _ in range(5):
            cx = rng.randint(0, size)
            cy = rng.randint(size // 4, size - 50)
            r = rng.randint(40, 120)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rand_color(150))
        draw.rectangle([0, int(size * 0.7), size, size], fill=(240, 240, 240, 120))

    elif style == "nature inspired":
        for _ in range(rng.randint(6, 10)):
            x0 = rng.randint(-50, size)
            y0 = rng.randint(-50, size)
            x1 = x0 + rng.randint(120, 260)
            y1 = y0 + rng.randint(120, 260)
            draw.ellipse([x0, y0, x1, y1], fill=rand_color(140))
        for _ in range(8):
            x = rng.randint(0, size)
            y = rng.randint(0, size)
            draw.arc([x - 80, y - 80, x + 80, y + 80], 0, 180, fill=rand_color(200), width=3)

    elif style == "urban architectural":
        skyline_base = int(size * 0.65)
        for _ in range(12):
            w = rng.randint(50, 140)
            h = rng.randint(120, 320)
            x = rng.randint(0, size - w)
            y = skyline_base - h
            draw.rectangle([x, y, x + w, skyline_base], fill=rand_color(180))
            for wx in range(x + 8, x + w - 8, 16):
                for wy in range(y + 10, skyline_base - 8, 20):
                    draw.rectangle([wx, wy, wx + 6, wy + 8], fill=(255, 255, 255, 90))

    elif style == "cosmic space":
        image = Image.new("RGB", (size, size), (15, 18, 35))
        draw = ImageDraw.Draw(image, "RGBA")
        for _ in range(120):
            x = rng.randint(0, size)
            y = rng.randint(0, size)
            r = rng.randint(1, 2)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, rng.randint(120, 200)))
        nebula = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        nebula_draw = ImageDraw.Draw(nebula, "RGBA")
        for _ in range(6):
            cx = rng.randint(0, size)
            cy = rng.randint(0, size)
            r = rng.randint(size // 6, size // 3)
            nebula_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rand_color(80))
        nebula = nebula.filter(ImageFilter.GaussianBlur(radius=30))
        image = Image.alpha_composite(image.convert("RGBA"), nebula).convert("RGB")

    elif style == "impressionist":
        for _ in range(rng.randint(200, 320)):
            x = rng.randint(0, size)
            y = rng.randint(0, size)
            length = rng.randint(8, 24)
            angle = rng.random() * math.pi
            x2 = x + int(math.cos(angle) * length)
            y2 = y + int(math.sin(angle) * length)
            draw.line([x, y, x2, y2], fill=rand_color(120), width=rng.randint(2, 4))

    elif style == "line art":
        image = Image.new("RGB", (size, size), (250, 248, 245))
        draw = ImageDraw.Draw(image, "RGBA")
        line_color = rand_color(160)
        for _ in range(10):
            points = [(rng.randint(0, size), rng.randint(0, size)) for _ in range(4)]
            draw.line(points, fill=line_color, width=2)
        for _ in range(6):
            cx = rng.randint(0, size)
            cy = rng.randint(0, size)
            r = rng.randint(40, 120)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=line_color, width=2)

    elif style == "collage":
        for _ in range(10):
            x0 = rng.randint(0, size - 120)
            y0 = rng.randint(0, size - 120)
            x1 = x0 + rng.randint(80, 220)
            y1 = y0 + rng.randint(80, 220)
            shadow = (0, 0, 0, 40)
            draw.rectangle([x0 + 6, y0 + 6, x1 + 6, y1 + 6], fill=shadow)
            draw.rectangle([x0, y0, x1, y1], fill=rand_color(200))

    else:
        for _ in range(12):
            cx = rng.randint(0, size)
            cy = rng.randint(0, size)
            r = rng.randint(40, 140)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rand_color(120))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def save_bytes_image(image_bytes, filename):
    upload_dir = Config.UPLOAD_FOLDER
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    return f"/static/uploads/{filename}"


def generate_svg_placeholder(style, emotions=None, themes=None, seed=None, size=768):
    rng = random.Random(seed)
    palette = _derive_palette(emotions)
    colors = ["#%02x%02x%02x" % c for c in palette]
    bg = colors[0] if colors else "#f2f2f2"
    blobs = []
    for _ in range(6):
        cx = rng.randint(60, size - 60)
        cy = rng.randint(60, size - 60)
        r = rng.randint(40, 120)
        color = rng.choice(colors)
        blobs.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" fill-opacity="0.35" />')
    title = (" ".join((themes or ["reflection"])[:3])).strip()
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 {size} {size}'>"
        f"<rect width='100%' height='100%' fill='{bg}'/>"
        + "".join(blobs) +
        f"<text x='50%' y='92%' fill='#444' font-size='20' font-family='sans-serif' text-anchor='middle'>{style}</text>"
        f"</svg>"
    )
    return svg.encode("utf-8")


def check_comfyui_status():
    """Check if ComfyUI is running at the configured URL.
    
    Returns:
        tuple: (is_running: bool, message: str)
    """
    if not Config.COMFY_ENABLED:
        return False, "ComfyUI is disabled in configuration"
    
    try:
        url = f"{Config.COMFY_API_URL}/system_stats"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return True, "ComfyUI is running"
        return False, f"ComfyUI returned status {resp.status_code}"
    except requests.ConnectionError:
        return False, f"Cannot connect to ComfyUI at {Config.COMFY_API_URL}"
    except requests.Timeout:
        return False, "ComfyUI connection timed out"
    except Exception as e:
        return False, f"ComfyUI check failed: {str(e)}"


def generate_image_comfy(prompt: str, seed: int = None, entry_id: str = None):
    """Generate an image using ComfyUI API.
    
    Args:
        prompt: The text prompt for image generation
        seed: Random seed (auto-generated if None)
        entry_id: Optional entry ID for logging
        
    Returns:
        bytes: PNG image data, or None on failure
    """
    from models.comfy_workflow import create_prism_workflow, create_prism_workflow_with_refiner
    
    if not Config.COMFY_ENABLED:
        log.debug("ComfyUI is disabled, skipping")
        return None
    
    is_running, message = check_comfyui_status()
    if not is_running:
        log.warning("ComfyUI unavailable: %s", message)
        return None
    
    if seed is None:
        seed = random.randint(1, 2**32 - 1)
    
    try:
        # Create workflow based on mode
        if Config.COMFY_WORKFLOW_MODE == "refiner":
            workflow = create_prism_workflow_with_refiner(prompt, seed=seed)
            log.debug("Using ComfyUI refiner workflow for entry %s", entry_id or "unknown")
        else:
            workflow = create_prism_workflow(prompt, seed=seed)
            log.debug("Using ComfyUI base workflow for entry %s", entry_id or "unknown")
        
        # Submit workflow to ComfyUI
        prompt_url = f"{Config.COMFY_API_URL}/prompt"
        payload = {"prompt": workflow}
        
        log.debug("Submitting workflow to ComfyUI at %s", prompt_url)
        resp = requests.post(prompt_url, json=payload, timeout=30)
        resp.raise_for_status()
        
        data = resp.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            log.error("ComfyUI did not return a prompt_id")
            return None
        
        log.debug("ComfyUI workflow submitted, prompt_id: %s", prompt_id)
        
        # Poll for completion (max 120 seconds)
        history_url = f"{Config.COMFY_API_URL}/history/{prompt_id}"
        max_wait = 120
        poll_interval = 2
        elapsed = 0
        
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            
            try:
                history_resp = requests.get(history_url, timeout=10)
                history_resp.raise_for_status()
                history_data = history_resp.json()
                
                if prompt_id in history_data:
                    entry = history_data[prompt_id]
                    status = entry.get("status", {})
                    status_str = status.get("status_str", "unknown")
                    
                    if status_str == "success":
                        # Get output images
                        outputs = entry.get("outputs", {})
                        for node_id, node_output in outputs.items():
                            images = node_output.get("images", [])
                            if images:
                                # Get the first image
                                image_info = images[0]
                                filename = image_info.get("filename")
                                subfolder = image_info.get("subfolder", "")
                                
                                # Retrieve the image
                                view_url = f"{Config.COMFY_API_URL}/view"
                                params = {
                                    "filename": filename,
                                    "subfolder": subfolder,
                                    "type": "output"
                                }
                                
                                image_resp = requests.get(view_url, params=params, timeout=30)
                                image_resp.raise_for_status()
                                
                                log.info("ComfyUI image generated successfully for entry %s", entry_id or "unknown")
                                return image_resp.content
                        
                        log.error("ComfyUI completed but no images found in output")
                        return None
                    elif status_str == "error":
                        log.error("ComfyUI workflow failed: %s", status.get("messages", "Unknown error"))
                        return None
                    # Still processing, continue polling
                    
            except requests.RequestException as e:
                log.debug("Polling error (will retry): %s", e)
                continue
        
        log.error("ComfyUI generation timed out after %s seconds", max_wait)
        return None
        
    except requests.ConnectionError as e:
        log.error("Cannot connect to ComfyUI at %s: %s", Config.COMFY_API_URL, e)
        return None
    except requests.Timeout:
        log.error("ComfyUI request timed out")
        return None
    except Exception as e:
        log.error("ComfyUI image generation failed: %s", e, exc_info=True)
        return None


def generate_image(prompt, style=None, emotions=None, themes=None, seed=None, entry_id=None):
    """Generate an image using available image generation services.
    
    Tries services in order:
    1. ComfyUI (if COMFY_ENABLED and available)
    2. Stable Diffusion WebUI (if SD_ENABLED and available)
    3. Algorithmic art (fallback)
    
    Args:
        prompt: The text prompt for image generation
        style: Artwork style (used for algorithmic fallback)
        emotions: List of emotions (used for algorithmic fallback)
        themes: List of themes (used for algorithmic fallback)
        seed: Random seed
        entry_id: Optional entry ID for logging
        
    Returns:
        dict: {"type": "bytes"|"base64", "data": bytes|str, "source": str}
              or None on complete failure
    """
    if seed is None:
        seed = random.randint(1, 2**32 - 1)
    
    # Try ComfyUI first
    if Config.COMFY_ENABLED:
        log.debug("Attempting ComfyUI generation for entry %s", entry_id or "unknown")
        image_bytes = generate_image_comfy(prompt, seed=seed, entry_id=entry_id)
        if image_bytes:
            return {
                "type": "bytes",
                "data": image_bytes,
                "source": "comfyui"
            }
        log.debug("ComfyUI failed, trying next option")
    
    # Try Stable Diffusion WebUI
    from utils.services import status
    if Config.SD_ENABLED and status.stable_diffusion:
        log.debug("Attempting SD WebUI generation for entry %s", entry_id or "unknown")
        base64_data = generate_image_sd(prompt, style)
        if base64_data:
            return {
                "type": "base64",
                "data": base64_data,
                "source": "stable_diffusion"
            }
        log.debug("SD WebUI failed, falling back to algorithmic art")
    
    # Fall back to algorithmic art
    log.debug("Using algorithmic art generation for entry %s", entry_id or "unknown")
    image_bytes = generate_algorithmic_art(style, emotions=emotions, themes=themes, seed=seed)
    if image_bytes:
        return {
            "type": "bytes",
            "data": image_bytes,
            "source": "algorithmic"
        }
    
    log.error("All image generation methods failed for entry %s", entry_id or "unknown")
    return None


def generate_image_sd(prompt, style=None):
    """Generate an image via the Stable Diffusion WebUI API.
    
    Returns the image as a base64-encoded string, or None on failure.
    Checks both Config.SD_ENABLED and live service status.
    """
    from utils.services import status
    
    if not Config.SD_ENABLED:
        return None
    if not status.stable_diffusion:
        log.warning("Stable Diffusion unavailable: %s", status.sd_message)
        return None
    
    style = style or Config.SD_DEFAULT_STYLE
    styled_prompt = f"{prompt}, {style} style" if style else prompt
    url = f"{Config.SD_API_URL}/sdapi/v1/txt2img"
    payload = {
        "prompt": styled_prompt,
        "negative_prompt": "text, watermark, blurry, low quality",
        "steps": 20,
        "width": 512,
        "height": 512,
        "cfg_scale": 7,
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=Config.OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        images = data.get("images", [])
        return images[0] if images else None
    except requests.ConnectionError:
        log.error(
            "Cannot connect to Stable Diffusion at %s. "
            "Start the WebUI with --api flag.",
            Config.SD_API_URL,
        )
        return None
    except requests.Timeout:
        log.error("Stable Diffusion request timed out.")
        return None
    except Exception as e:
        log.error("Stable Diffusion image generation failed: %s", e)
        return None
