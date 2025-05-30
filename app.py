# app.py
# This is the main application file for the Agentic Chat LLM interface.
# It handles the core logic for processing user input (text and files),
# interacting with the LLM handler, and managing the chat history.

import gradio as gr
import os
from typing import List, Tuple, Optional
from typing import AsyncGenerator, Any # Import AsyncGenerator
from config import (
    DEFAULT_LLM_PROVIDER, DEFAULT_MODEL, AVAILABLE_MODELS, # LLM Config
    DEFAULT_PROMPT_IMAGE, DEFAULT_PROMPT_AUDIO, DEFAULT_PROMPT_VIDEO, DEFAULT_PROMPT_FILE # Default Prompts
)

from llm_handler import LLMHandler
from utils import (
    process_uploaded_file, # Import the moved function
)

# Initialize LLM Handler
# This instance will be shared across the application.
llm_handler_instance: Optional[LLMHandler] = None

def initialize_llm_handler(model_name: str) -> None:
    """
    Initializes or re-initializes the global LLMHandler instance
    with the specified model name. This is called at startup and
    when the user changes the model via the UI dropdown.
    """
    global llm_handler_instance
    try:
        llm_handler_instance = LLMHandler(provider=DEFAULT_LLM_PROVIDER, model_name=model_name)
        # No return needed as it modifies a global instance and Gradio output is None
    except ValueError as e:
        print(f"Error initializing LLMHandler: {e}")
        # No return needed

# --- Input Processing Helper ---
def _prepare_inputs_for_llm(
    user_message: str,
    # Gradio File component output. It's a temporary file object
    # created by Gradio when a file is uploaded. Its 'name' attribute
    # holds the path to the temporary file.
    # It's Optional because the user might not upload a file.
    file_obj: Optional[gr.File]
) -> Tuple[
    str,  # text_for_llm
    str,  # display_prompt (what user sees in history)
    Optional[str], Optional[str],  # image_data, image_mime
    Optional[str], Optional[str],  # audio_data, audio_mime
    Optional[str], Optional[str],  # video_data, video_mime
    bool,  # has_processed_media (true if image/audio/video successfully processed for LLM)
    Optional[str], # media_category ("image", "audio", "video", "file", None)
    Optional[str]  # original_file_path (for UI display)
]:
    """
    Processes user message and an optional file upload to prepare inputs for the LLM
    and the chat display.
    """
    # Initialize media data variables
    # These will hold base64 encoded data and MIME types for supported media.
    base64_image_data: Optional[str] = None
    # MIME type (e.g., "image/jpeg", "audio/wav")
    image_mime_type: Optional[str] = None
    base64_audio_data: Optional[str] = None
    audio_mime_type: Optional[str] = None
    base64_video_data: Optional[str] = None
    video_mime_type: Optional[str] = None
    file_processing_info_message: str = ""
    has_processed_media: bool = False
    processed_media_category: Optional[str] = None
    original_file_path_for_ui: Optional[str] = None

    # The `process_uploaded_file` function (from utils.py) handles reading
    # the file, base64 encoding, and determining its type and info message.

    # Process uploaded file if any
    if file_obj is not None:
        temp_file_path = file_obj.name
        original_file_path_for_ui = temp_file_path # Store for UI display
        file_name = os.path.basename(temp_file_path)
        # process_uploaded_file returns: b64_data, mime, category, info_msg
        b64_data, mime, category_from_util, info_msg = process_uploaded_file(temp_file_path, file_name)
        file_processing_info_message = info_msg
        processed_media_category = category_from_util # Store the category

        if category_from_util == "image" and b64_data and mime:
            base64_image_data, image_mime_type = b64_data, mime
            has_processed_media = True
        elif category_from_util == "audio" and b64_data and mime:
            base64_audio_data, audio_mime_type = b64_data, mime
            has_processed_media = True
        elif category_from_util == "video" and b64_data and mime:
            base64_video_data, video_mime_type = b64_data, mime
            has_processed_media = True

    # Determine the text for LLM and the prompt to display to the user
    # text_for_llm is the actual text query sent to the LLM.
    # display_prompt is what the user sees as their message in the chat history.
    text_for_llm = user_message
    display_prompt = user_message

    if file_processing_info_message: # If any file was processed (even if not media for LLM)
        # Prepend file processing info to the user's message in the display history.
        # This shows the user that their file was received and processed.
        display_prompt = f"{file_processing_info_message}\n\n{user_message}" if user_message else file_processing_info_message

    if has_processed_media and not user_message:
        # If there's media but no specific user text, use a default prompt for that media type
        if base64_image_data: text_for_llm = DEFAULT_PROMPT_IMAGE
        elif base64_audio_data: text_for_llm = DEFAULT_PROMPT_AUDIO
        elif base64_video_data: text_for_llm = DEFAULT_PROMPT_VIDEO

    return (
        text_for_llm, display_prompt,
        base64_image_data, image_mime_type,
        base64_audio_data, audio_mime_type,
        base64_video_data, video_mime_type,
        has_processed_media,
        processed_media_category,
        original_file_path_for_ui
    )


# --- UI Helper for Immediate File Preview ---
def handle_file_upload_for_preview(file_obj: Optional[gr.File]) -> Tuple[Any, Any, Any]:
    """
    Handles a file upload or clear event specifically for updating UI previews.
    Returns update objects for image, audio, and video display components.
    """
    image_update = gr.update(value=None, visible=False)
    audio_update = gr.update(value=None, visible=False)
    video_update = gr.update(value=None, visible=False)

    if file_obj is None:
        # File was cleared by the user
        return image_update, audio_update, video_update

    temp_file_path = file_obj.name
    file_name = os.path.basename(temp_file_path)
    
    # Process the file to determine its category and if it's valid media
    # We use b64_data's presence to confirm if the media was successfully processed for preview.
    b64_data, _, category, _ = process_uploaded_file(temp_file_path, file_name)

    if category == "image" and b64_data:
        image_update = gr.update(value=temp_file_path, visible=True)
    elif category == "audio" and b64_data:
        audio_update = gr.update(value=temp_file_path, visible=True)
    elif category == "video" and b64_data:
        video_update = gr.update(value=temp_file_path, visible=True)
    # If category is 'file' or b64_data is None (processing failed for media),
    # all previews remain hidden as per their initial update values.
    
    return image_update, audio_update, video_update

# --- Gradio Interface Logic ---
async def chat_fn(
    user_message: str,
    file_obj: Optional[gr.File], # Gradio's File component output
    chat_history: List[dict[str, Optional[str]]],
    model_name: str  # Add model_name as input
) -> AsyncGenerator[
    Tuple[
        str,  # text_input update
        Optional[gr.File],  # file_upload update
        List[dict[str, Optional[str]]],  # chatbot_display update
        List[dict[str, Optional[str]]],  # chat_history_state update
        Any,  # image_display update (gr.update object)
        Any,  # audio_display update (gr.update object)
        Any   # video_display update (gr.update object)
    ], None
]:
    """
    Handles the chat interaction.
    Processes user input (text and/or file), calls the LLM, and updates history.
    """
    # Ensure LLM handler is initialized
    if llm_handler_instance is None:
        error_message = "LLM Handler not initialized. Please check API key and configurations."
        # Use a generic message if user_message is also empty (e.g. on file upload attempt with no text)
        user_turn_content = user_message if user_message else "[File Upload Attempt]"
        updated_history = chat_history + [{"role": "user", "content": user_turn_content}, {"role": "assistant", "content": error_message}]
        # Yield initial (empty/hidden) states for media displays
        yield "", None, updated_history, updated_history, gr.update(value=None, visible=False), gr.update(value=None, visible=False), gr.update(value=None, visible=False)
        return

    # Initialize media display updates to hidden
    image_update = gr.update(value=None, visible=False)
    audio_update = gr.update(value=None, visible=False)
    video_update = gr.update(value=None, visible=False)

    # Prepare inputs from user message and file
    (
        text_for_llm, display_prompt,
        base64_image_data, image_mime_type,
        base64_audio_data, audio_mime_type,
        base64_video_data, video_mime_type,
        has_processed_media,
        media_category, # New from _prepare_inputs_for_llm
        original_file_path # New from _prepare_inputs_for_llm
    ) = _prepare_inputs_for_llm(user_message, file_obj)

    # Update media display components based on successfully processed uploaded file
    if original_file_path: # A file was provided
        if media_category == "image" and base64_image_data: # Successfully processed as image
            image_update = gr.update(value=original_file_path, visible=True)
            # Ensure other media types are hidden if an image is shown
            audio_update = gr.update(value=None, visible=False)
            video_update = gr.update(value=None, visible=False)
        elif media_category == "audio" and base64_audio_data: # Successfully processed as audio
            audio_update = gr.update(value=original_file_path, visible=True)
            image_update = gr.update(value=None, visible=False)
            video_update = gr.update(value=None, visible=False)
        elif media_category == "video" and base64_video_data: # Successfully processed as video
            video_update = gr.update(value=original_file_path, visible=True)
            image_update = gr.update(value=None, visible=False)
            audio_update = gr.update(value=None, visible=False)
        # If not a displayable/processed media type, updates remain to hide/clear them.
    # Handle cases of no actual input for the LLM or display
    # display_prompt will be empty if user_message is empty AND no file was uploaded/processed
    if not display_prompt:
        # If there's no text message and no file was uploaded/processed
        # (or the file processing yielded an empty info message, which shouldn't happen
        # for any file), then there's no input to process or display.
        # Provide a message asking for input.
        if not chat_history:
            error_message = "Please type a message or upload a file."
        else:
            error_message = "Please provide new input or ask a question."
        # Do not add user message to history if it's empty
        chat_history.append({"role": "assistant", "content": error_message})
        yield "", None, chat_history, chat_history, image_update, audio_update, video_update
        return

    # Append user's turn to chat history for display
    # display_prompt contains the file info message (if any) and the user's text.
    chat_history.append({"role": "user", "content": display_prompt})
    yield "", None, chat_history, chat_history, image_update, audio_update, video_update # Update UI with user's message & media

    # The actual call to the LLM handler happens here.
    # It's only called if there's user text or successfully processed media.
    # Call LLM if there's something to process
    if text_for_llm or has_processed_media:
        ai_response = await llm_handler_instance.generate_response(
            user_text=text_for_llm,
            base64_image_data=base64_image_data,
            image_mime_type=image_mime_type,
            base64_audio_data=base64_audio_data,
            audio_mime_type=audio_mime_type,
            base64_video_data=base64_video_data,
            video_mime_type=video_mime_type,
            chat_history=chat_history[:-1] # Pass history excluding the current user prompt
        )
        chat_history.append({"role": "assistant", "content": ai_response})
    else:
        # This else block should ideally not be reached if the initial `if not display_prompt:`
        # check works correctly. However, it serves as a fallback.
        # It might be reached if a file was uploaded, `file_processing_info_message` was set,
        # but `has_processed_media` is False (e.g., PDF) AND `user_message` is empty.
        chat_history.append({"role": "assistant", "content": "I received the file information. How can I help you with it?"})

    yield "", None, chat_history, chat_history, image_update, audio_update, video_update # Update UI with AI's response


# --- Main Application Execution ---
if __name__ == "__main__":
    # Import UI creation function here to avoid circular imports at module load time
    # and to ensure app.py's globals are defined before ui.py tries to import them.
    # This ensures that `llm_handler_instance` is potentially initialized
    # before the UI is built, allowing the UI to display an error if initialization fails.
    from ui import create_gradio_ui

    # Ensure LLM is initialized with the default model before UI creation.
    # This is crucial because ui.py will check llm_handler_instance status
    # to display an error message if initialization failed.
    if llm_handler_instance is None:
        # Attempt to initialize the LLM handler with the default model.
        initialize_llm_handler(DEFAULT_MODEL) # Attempt to initialize with the default model
        if llm_handler_instance is None:
            # This print is for the console; the UI will show a gr.Error based on
            # the llm_handler_instance state when create_gradio_ui() is called.
            # This indicates a critical failure, likely due to a missing API key.
            print("CRITICAL: LLM Handler failed to initialize at startup. Check API key and configurations.")

    # Create and launch the Gradio UI
    demo_instance = create_gradio_ui()
    demo_instance.queue() 
    demo_instance.launch(debug=True)