# data_models.py
# This file defines data classes used for structuring data within the application,
# particularly for inputs to and outputs from core processing functions.

from dataclasses import dataclass
from typing import List, Optional, Any
# import gradio as gr # Only needed if you want to use specific Gradio types like gr.File

@dataclass
class PreparedInput:
    """
    Holds all the processed information ready to be used by the LLM
    and for display in the chat UI, returned by _prepare_inputs_for_llm.
    """
    text_for_llm: str
    display_prompt: str
    base64_image_data: Optional[str]
    image_mime_type: Optional[str]
    base64_audio_data: Optional[str]
    audio_mime_type: Optional[str]
    base64_video_data: Optional[str]
    video_mime_type: Optional[str]
    has_processed_media: bool
    processed_media_category: Optional[str]
    original_file_path_for_ui: Optional[str]

@dataclass
class ChatOutputUpdate:
    """
    Represents the collective set of updates to be yielded by chat_fn
    for various Gradio UI components. Each field corresponds to an output component.
    """
    text_input: str  # Value for text_input component update
    file_upload: Optional[Any]  # Value for file_upload component update (e.g., None to clear)
    chatbot_display: List[dict[str, Optional[str]]]  # Value for chatbot_display component update
    chat_history_state: List[dict[str, Optional[str]]]  # Value for chat_history_state update
    image_display: Any  # gr.update object or value for image_display component update
    audio_display: Any  # gr.update object or value for audio_display component update
    video_display: Any  # gr.update object or value for video_display component update