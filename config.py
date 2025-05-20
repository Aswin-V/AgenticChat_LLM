# config.py
# This file stores application-wide configuration settings.

from typing import List

# --- LLM Configuration ---
DEFAULT_LLM_PROVIDER: str = "google"

# Default model name. Choose a model that supports multimodal input if needed.
# Available models depend on the provider and your API key/permissions.
DEFAULT_MODEL: str = "gemini-2.0-flash"
#https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/gemini-2.5-flash-preview-05-20?inv=1&invt=Abx8Bg
AVAILABLE_MODELS: List[str] = [
    DEFAULT_MODEL,
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    #"gemini-2.5-pro-preview-05-06", #Does not have free quota tier
    "gemini-2.5-flash-preview-05-20"  # Example: Assuming this is a valid model name
]

# --- UI Configuration ---
UI_THEME: str = "gradio/soft" # Gradio theme
UI_TITLE: str = "Agentic Chat Application"
UI_DESCRIPTION: str = f"Powered by Langchain and Google Generative AI ({DEFAULT_MODEL}). Type your message, or upload an image, audio, or video file."
UI_AVATAR_AI: str = "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png" # URL for AI avatar image

UI_MODEL_LABEL: str = "Select Model"
UI_CHATBOT_LABEL: str = "Chat Window"
UI_TEXT_INPUT_LABEL: str = "Your Message"
UI_TEXT_INPUT_PLACEHOLDER: str = "Type your message here, or upload a file and ask about it..."
UI_FILE_UPLOAD_LABEL: str = "Upload File (Image, Audio, Video, etc.)"
UI_FILE_UPLOAD_TYPES: List[str] = ["image", "audio", "video", ".pdf", ".txt", ".json", ".csv"]
UI_BUTTON_SEND_TEXT: str = "Send"
UI_BUTTON_CLEAR_TEXT: str = "Clear Chat"
UI_LLM_INIT_ERROR: str = "LLM Handler could not be initialized. This is likely due to a missing or invalid GOOGLE_API_KEY in your .env file. Please check the console for more details and set up the API key."

# --- Default Prompts for Media (used when no text message is provided with a file) ---
DEFAULT_PROMPT_IMAGE: str = "Describe the uploaded image."
DEFAULT_PROMPT_AUDIO: str = "Transcribe or summarize the uploaded audio."
DEFAULT_PROMPT_VIDEO: str = "Describe or summarize the uploaded video."
DEFAULT_PROMPT_FILE: str = "Describe the uploaded content." # Generic fallback