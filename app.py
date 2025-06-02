# app.py
# This is the main application logic file for the Agentic Chat LLM interface.
# It acts as the "Service Layer" or "Application Controller" in a layered architecture.
#
# Responsibilities:
# - Orchestrating the overall chat flow.
# - Processing user input (text and files) received from the UI layer (`ui.py`).
# - Preparing data for the LLM via `llm_handler.py`.
# - Managing chat history state.
# - Initializing core components like the LLM handler.
# - Launching the Gradio UI.

import gradio as gr
import os
# from dataclasses import dataclass # No longer needed here if PreparedInput is moved
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
from data_models import PreparedInput, ChatOutputUpdate # Import the data classes

# Global LLM Handler instance.
# This instance is initialized at startup and can be re-initialized if the model changes.
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
        print(f"LLMHandler initialized successfully with model: {model_name}")
        # This function is often called as a Gradio event handler, which expects no specific output
        # if it's just modifying global state or performing an action.
    except ValueError as e:
        print(f"Error initializing LLMHandler: {e}")
        llm_handler_instance = None # Ensure it's None if initialization fails

# --- Input Processing Helper ---
def _prepare_inputs_for_llm(
    user_message: str,
    file_obj: Optional[gr.File]
) -> PreparedInput:
    """
    Processes user message and an optional file upload to prepare inputs for the LLM
    and the chat display. This function acts as a core part of the application logic,
    transforming raw UI inputs into structured data for the LLM and for display.

    Args:
        user_message (str): The text message from the user.
        file_obj (Optional[gr.File]): The Gradio temporary file object if a file was uploaded.

    Returns:
        PreparedInput: An object containing all processed data.
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

    return PreparedInput(
        text_for_llm=text_for_llm,
        display_prompt=display_prompt,
        base64_image_data=base64_image_data,
        image_mime_type=image_mime_type,
        base64_audio_data=base64_audio_data,
        audio_mime_type=audio_mime_type,
        base64_video_data=base64_video_data,
        video_mime_type=video_mime_type,
        has_processed_media=has_processed_media,
        processed_media_category=processed_media_category,
        original_file_path_for_ui=original_file_path_for_ui
    )

# --- UI Helper for Immediate File Preview ---
def handle_file_upload_for_preview(file_obj: Optional[gr.File]) -> Tuple[Any, Any, Any]:
    """
    Handles a file upload or clear event specifically for updating UI previews.
    Returns update objects for image, audio, and video display components.
    """
    # Default updates: hide all media previews.
    image_update = gr.update(value=None, visible=False)
    audio_update = gr.update(value=None, visible=False)
    video_update = gr.update(value=None, visible=False)

    if file_obj is None:
        # If file_obj is None, it means the file was cleared in the UI.
        # Return updates to hide all previews.
        return image_update, audio_update, video_update

    temp_file_path = file_obj.name
    file_name = os.path.basename(temp_file_path)
    
    # Process the file to determine its category and if it's valid media
    # The `process_uploaded_file` utility handles the underlying file reading and type detection.
    # We use `b64_data`'s presence to confirm if the media was successfully processed
    # and is suitable for preview.
    b64_data, _, category, _ = process_uploaded_file(temp_file_path, file_name)

    if category == "image" and b64_data:
        image_update = gr.update(value=temp_file_path, visible=True)
    elif category == "audio" and b64_data:
        audio_update = gr.update(value=temp_file_path, visible=True)
    elif category == "video" and b64_data:
        video_update = gr.update(value=temp_file_path, visible=True)
    # If category is 'file' (e.g., PDF, TXT) or if b64_data is None (meaning
    # the file wasn't successfully processed as displayable media),
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
    Core asynchronous function to handle a chat interaction turn.
    This function is triggered by user submission (text and/or file) from the UI.

    It orchestrates:
    1. Validation of LLM handler initialization.
    2. Preparation of user inputs (text, files) using `_prepare_inputs_for_llm`.
    3. Updating the chat history with the user's turn and media previews.
    4. Invoking the LLM handler (`llm_handler_instance`) to get a response.
    5. Updating the chat history with the AI's response.
    6. Yielding updates back to the Gradio UI components.

    Args:
        user_message (str): The text message from the user.
        file_obj (Optional[gr.File]): Gradio temporary file object if a file was uploaded.
        chat_history (List[dict[str, Optional[str]]]): Current state of the chat history.
        model_name (str): The name of the LLM model selected by the user.

    Yields:
        Tuple: A tuple of updates for various Gradio UI components (textbox, file upload,
               chatbot display, history state, media previews).
    """
    # Step 1: Ensure LLM handler is initialized. If not, show an error.
    if llm_handler_instance is None:
        error_message = "LLM Handler not initialized. Please check API key and configurations."
        user_turn_content = user_message if user_message else "[File Upload Attempt]"
        updated_history = chat_history + [
            {"role": "user", "content": user_turn_content},
            {"role": "assistant", "content": error_message}
        ]
        output_updates = ChatOutputUpdate(
            text_input="",
            file_upload=None,
            chatbot_display=updated_history,
            chat_history_state=updated_history,
            image_display=gr.update(value=None, visible=False),
            audio_display=gr.update(value=None, visible=False),
            video_display=gr.update(value=None, visible=False)
        )
        yield (
            output_updates.text_input, output_updates.file_upload, output_updates.chatbot_display,
            output_updates.chat_history_state, output_updates.image_display,
            output_updates.audio_display, output_updates.video_display
        )
        return

    # Initialize media display updates for the user's current submission
    submitted_image_preview_update = gr.update(value=None, visible=False)
    submitted_audio_preview_update = gr.update(value=None, visible=False)
    submitted_video_preview_update = gr.update(value=None, visible=False)

    # Step 2: Prepare inputs from user message and file using the helper function.
    prepared_input = _prepare_inputs_for_llm(user_message, file_obj)

    # Step 3a: Update UI media display components based on the processed uploaded file.
    # This happens before the LLM call, providing immediate feedback to the user.
    if prepared_input.original_file_path_for_ui: # A file was provided
        if prepared_input.processed_media_category == "image" and prepared_input.base64_image_data:
            submitted_image_preview_update = gr.update(value=prepared_input.original_file_path_for_ui, visible=True)
            # Ensure other media types are hidden if an image is shown
            submitted_audio_preview_update = gr.update(value=None, visible=False)
            submitted_video_preview_update = gr.update(value=None, visible=False)
        elif prepared_input.processed_media_category == "audio" and prepared_input.base64_audio_data:
            submitted_audio_preview_update = gr.update(value=prepared_input.original_file_path_for_ui, visible=True)
            submitted_image_preview_update = gr.update(value=None, visible=False)
            submitted_video_preview_update = gr.update(value=None, visible=False)
        elif prepared_input.processed_media_category == "video" and prepared_input.base64_video_data:
            submitted_video_preview_update = gr.update(value=prepared_input.original_file_path_for_ui, visible=True)
            submitted_image_preview_update = gr.update(value=None, visible=False)
            submitted_audio_preview_update = gr.update(value=None, visible=False)
        # If not a displayable/processed media type, updates remain to hide/clear them.

    if not prepared_input.display_prompt:
        # If display_prompt is empty, it means no user text and no file info to show.
        if not chat_history:
            error_message = "Please type a message or upload a file."
        else:
            error_message = "Please provide new input or ask a question."
        # Do not add user message to history if it's empty
        chat_history.append({"role": "assistant", "content": error_message})
        output_updates = ChatOutputUpdate(
            text_input="",
            file_upload=None,
            chatbot_display=chat_history,
            chat_history_state=chat_history,
            image_display=submitted_image_preview_update, # Use the potentially updated preview
            audio_display=submitted_audio_preview_update,
            video_display=submitted_video_preview_update
        )
        yield (
            output_updates.text_input, output_updates.file_upload, output_updates.chatbot_display,
            output_updates.chat_history_state, output_updates.image_display,
            output_updates.audio_display, output_updates.video_display
        )
        return

    # Step 3b: Append user's turn (text and file info) to chat history for display.
    # display_prompt contains the file info message (if any) and the user's text.
    chat_history.append({"role": "user", "content": prepared_input.display_prompt})
    # Yield updates to show the user's message and any media preview immediately.
    user_turn_output_updates = ChatOutputUpdate(
        text_input="", file_upload=None, chatbot_display=chat_history, chat_history_state=chat_history,
        image_display=submitted_image_preview_update, audio_display=submitted_audio_preview_update, video_display=submitted_video_preview_update
    )
    yield (
        user_turn_output_updates.text_input, user_turn_output_updates.file_upload,
        user_turn_output_updates.chatbot_display, user_turn_output_updates.chat_history_state,
        user_turn_output_updates.image_display, user_turn_output_updates.audio_display,
        user_turn_output_updates.video_display
    )

    # Step 4: Call the LLM if there's meaningful content (text_for_llm or processed media).
    if prepared_input.text_for_llm or prepared_input.has_processed_media:
        # The llm_handler_instance handles the actual API call and response generation.
        ai_response = await llm_handler_instance.generate_response(
            user_text=prepared_input.text_for_llm,
            base64_image_data=prepared_input.base64_image_data,
            image_mime_type=prepared_input.image_mime_type,
            base64_audio_data=prepared_input.base64_audio_data,
            audio_mime_type=prepared_input.audio_mime_type,
            base64_video_data=prepared_input.base64_video_data,
            video_mime_type=prepared_input.video_mime_type,
            chat_history=chat_history[:-1] # Pass history excluding the current user prompt
        )
        chat_history.append({"role": "assistant", "content": ai_response})
    else:        
        # This fallback might be reached if a file was uploaded (e.g., a PDF),
        # resulting in a `display_prompt` (e.g., "[File 'doc.pdf' uploaded]..."),
        # but `has_processed_media` is False (it's not an image/audio/video for direct LLM processing)
        # AND `text_for_llm` is empty (no user text query about the file).
        # In this scenario, the LLM wasn't called, so provide a generic assistant message.
        chat_history.append({"role": "assistant", "content": "I received the file information. How can I help you with it?"})

    # Step 5: Yield final updates to the UI, including the AI's response.
    # The text input and file upload fields are cleared. Media previews should also be cleared.
    final_output_updates = ChatOutputUpdate(
        text_input="", file_upload=None, chatbot_display=chat_history, chat_history_state=chat_history,
        image_display=gr.update(value=None, visible=False), # Clear preview
        audio_display=gr.update(value=None, visible=False), # Clear preview
        video_display=gr.update(value=None, visible=False)  # Clear preview
    )
    yield (
        final_output_updates.text_input, final_output_updates.file_upload,
        final_output_updates.chatbot_display, final_output_updates.chat_history_state,
        final_output_updates.image_display, final_output_updates.audio_display,
        final_output_updates.video_display
    )


# --- Main Application Execution ---
if __name__ == "__main__":
    # This block executes when app.py is run directly.
    # It's responsible for initializing the application and launching the UI.

    # Dynamically import UI creation function here.
    # This helps avoid potential circular import issues if ui.py also imports from app.py
    # (though in this layered setup, ui.py primarily calls app.py functions).
    # It also ensures that `llm_handler_instance` is initialized (or attempted)
    # before the UI is built, allowing the UI to reflect initialization status.
    from ui import create_gradio_ui

    if llm_handler_instance is None:
        # Attempt to initialize the LLM handler with the default model.
        initialize_llm_handler(DEFAULT_MODEL) # Attempt to initialize with the default model
        if llm_handler_instance is None:
            # This print is for the console; the UI will show a gr.Error based on
            # the llm_handler_instance state when create_gradio_ui() is called.
            # This indicates a critical failure, likely due to a missing API key.
            print("CRITICAL: LLM Handler failed to initialize at startup. Check API key and configurations.")

    # Create and launch the Gradio UI
    # The `create_gradio_ui` function (from ui.py) defines the UI layout and event handlers.
    demo_instance: gr.Blocks = create_gradio_ui()
    demo_instance.queue() 
    demo_instance.launch(debug=True) # Launch in debug mode for development