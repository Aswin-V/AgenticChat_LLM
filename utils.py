# utils.py
# This file contains common utility functions used across the application.
# These functions provide support for tasks like file processing,
# MIME type detection, and base64 encoding, abstracting these details
# from the main application logic.

from PIL import Image, UnidentifiedImageError
import os # Import the os module
import base64 # Used for encoding file data to base64 strings.
from typing import Tuple, Optional

def get_image_mime_type(file_path: str) -> Optional[str]:
    """Determines the MIME type of an image file."""
    try:
        with Image.open(file_path) as img:
            img.load()  # Ensure image data is loaded
            format_map = {"JPEG": "image/jpeg", "PNG": "image/png", "GIF": "image/gif", "WEBP": "image/webp"}
            return format_map.get(img.format, "application/octet-stream")
    # Catch common errors during image processing (file not found, not an image).
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
    # Determines the MIME type of a video file based on its file extension.
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
        # Open the file in binary read mode ('rb') and encode its content.
        with open(file_path, "rb") as file_binary:
            return base64.b64encode(file_binary.read()).decode("utf-8")
    except Exception:
        return None # Return None if any error occurs during file reading or encoding.

def process_uploaded_file(file_path: str, file_name: str) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """
    Processes an uploaded file to determine its type, MIME type,
    and base64 encode it if it's a supported media type (image, audio, video).
    It attempts image processing first using PIL, then falls back to checking
    audio/video extensions if PIL fails or indicates a non-image.
    For other file types, it provides an info message but no base64 data.

    and base64 encode it if it's a supported media type.
    Returns: (base64_data, mime_type, media_category ("image", "audio", "video", "file"), info_message)
    """
    base64_data: Optional[str] = None
    mime_type: Optional[str] = None
    media_category: Optional[str] = None # Use Optional[str] for consistency
    info_message: str = ""

    # Try to process as an image first
    # get_image_mime_type uses PIL, which can identify many image formats.
    image_mime = get_image_mime_type(file_path)
    if image_mime and image_mime != "application/octet-stream":
        base64_data = encode_file_to_base64(file_path)
        if base64_data:
            mime_type = image_mime
            media_category = "image"
            info_message = f"[Image '{file_name}' uploaded]"
        else:
            info_message = f"[Error processing image file '{file_name}']"
    else:
        # If not a recognized image by PIL, check for audio/video by extension
        # This is a simpler check based on file extension, less robust than PIL.
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext in ['.wav', '.mp3', '.ogg', '.flac']:
            mime_type = get_audio_mime_type(file_name) # Already gets mime from extension
            base64_data = encode_file_to_base64(file_path)
            media_category = "audio" # Set category even if encoding fails, for info_message
            info_message = f"[Audio file '{file_name}' uploaded]" if base64_data else f"[Error processing audio file '{file_name}']"
        elif file_ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            mime_type = get_video_mime_type(file_name) # Already gets mime from extension
            base64_data = encode_file_to_base64(file_path)
            media_category = "video" # Set category even if encoding fails, for info_message
            info_message = f"[Video file '{file_name}' uploaded]" if base64_data else f"[Error processing video file '{file_name}']"
        else:
            # For any other file extension, categorize as a generic file.
            media_category = "file"
            info_message = f"[File '{file_name}' uploaded. Type not fully supported for direct processing.]"

    if not base64_data: # If encoding failed or not applicable for media
        # If we couldn't get base64 data (either encoding failed or it's a non-media file),
        mime_type = None # Ensure mime_type is None if no b64_data for media

    return base64_data, mime_type, media_category, info_message