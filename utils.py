import os
from PIL import Image, UnidentifiedImageError
import base64
from typing import Tuple, Optional

def get_image_mime_type(file_path: str) -> Optional[str]:
    """Determines the MIME type of an image file."""
    try:
        with Image.open(file_path) as img:
            img.load()  # Ensure image data is loaded
            format_map = {"JPEG": "image/jpeg", "PNG": "image/png", "GIF": "image/gif", "WEBP": "image/webp"}
            return format_map.get(img.format, "application/octet-stream")
    except (IOError, UnidentifiedImageError):
        return None

def get_audio_mime_type(file_name: str) -> str:
    """Determines the MIME type of an audio file based on its extension."""
    file_type = os.path.splitext(file_name)[1].lower()
    audio_format_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }
    return audio_format_map.get(file_type, "application/octet-stream")

def get_video_mime_type(file_name: str) -> str:
    """Determines the MIME type of a video file based on its extension."""
    file_type = os.path.splitext(file_name)[1].lower()
    video_format_map = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }
    return video_format_map.get(file_type, "application/octet-stream")

def encode_file_to_base64(file_path: str) -> Optional[str]:
    """Reads a file and returns its base64 encoded string."""
    try:
        with open(file_path, "rb") as file_binary:
            return base64.b64encode(file_binary.read()).decode("utf-8")
    except Exception:
        return None

def get_default_prompt_for_media(media_type: str) -> str:
    """Returns a default prompt based on the media type."""
    if media_type == "image":
        return "Describe the uploaded image."
    elif media_type == "audio":
        return "Transcribe or summarize the uploaded audio."
    elif media_type == "video":
        return "Describe or summarize the uploaded video."
    return "Describe the uploaded content."