# ui.py
# This module is responsible for creating and managing the Gradio-based user interface.
# It defines all UI components, their layout, and their event handlers.
# This layer is concerned with presentation and user interaction.
#
# Key principles for this layer:
# - Define UI elements and structure.
# - Handle direct UI events (e.g., button clicks, file uploads for preview).
# - Delegate core application logic (like processing chat messages, LLM interaction)
#   to functions in the application layer (`app.py`).

import gradio as gr

from app import (
    chat_fn, initialize_llm_handler, handle_file_upload_for_preview,
    llm_handler_instance,
    DEFAULT_MODEL,
    AVAILABLE_MODELS
)
from config import ( # Import UI config variables
    UI_THEME, UI_TITLE, UI_DESCRIPTION, UI_AVATAR_AI,
    UI_MODEL_LABEL, UI_CHATBOT_LABEL, UI_TEXT_INPUT_LABEL,
    UI_TEXT_INPUT_PLACEHOLDER, UI_FILE_UPLOAD_LABEL, UI_FILE_UPLOAD_TYPES,
    UI_BUTTON_SEND_TEXT, UI_BUTTON_CLEAR_TEXT, UI_LLM_INIT_ERROR
)

# --- UI-Specific Helper Functions ---

def clear_chat_history_fn_ui():
    """
    Clears all relevant chat interface components and media displays in the UI.
    This function is called when the "Clear Chat" button is pressed.

    Returns:
        Tuple: A tuple of values to update/clear various Gradio components:
               text input, file upload, chatbot display, chat history state, and media previews.
    """
    return (
        "",  # text_input
        None,  # file_upload
        [],  # chatbot_display
        [],  # chat_history_state
        gr.update(value=None, visible=False),  # image_display
        gr.update(value=None, visible=False),  # audio_display
        gr.update(value=None, visible=False)   # video_display
    )

def create_gradio_ui() -> gr.Blocks:
    """
    Creates and returns the Gradio Blocks UI for the chat application.
    This function defines the entire layout, components, and event bindings.

    Returns:
        gr.Blocks: The main Gradio Blocks instance representing the UI.
    """
    with gr.Blocks(theme=UI_THEME) as demo:
        # Application Title and Description
        gr.Markdown(f"""
        # {UI_TITLE}
        {UI_DESCRIPTION}
        """)

        with gr.Row():
            # --- Left Column: Inputs and Media Preview ---
            with gr.Column(scale=1): # Left column for inputs and media display
                model_selection = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value=DEFAULT_MODEL,
                    label=UI_MODEL_LABEL,
                )
                model_selection.change(fn=initialize_llm_handler, inputs=[model_selection], outputs=None)
                # When the model selection changes, call `initialize_llm_handler` from app.py
                # to re-initialize the LLM with the new model.

                text_input = gr.Textbox(
                    label=UI_TEXT_INPUT_LABEL,
                    placeholder=UI_TEXT_INPUT_PLACEHOLDER,
                    lines=3,
                    autofocus=True,
                )
                file_upload = gr.File(
                    label=UI_FILE_UPLOAD_LABEL,
                    file_types=UI_FILE_UPLOAD_TYPES,
                )
                
                # Accordion for Uploaded Media Preview
                with gr.Accordion("Uploaded Media Preview", open=True): # Open by default
                    image_display = gr.Image(label="Image Preview", visible=False, interactive=False, height=200)
                    audio_display = gr.Audio(label="Audio Preview", visible=False, interactive=False) # type: ignore
                    video_display = gr.Video(label="Video Preview", visible=False, interactive=False, height=200) # type: ignore

                # Event handler for immediate media preview on file upload/clear
                # `handle_file_upload_for_preview` (from app.py, but could be in ui.py if purely UI)
                # is called to update the preview components.
                file_upload.change(
                    fn=handle_file_upload_for_preview,
                    inputs=[file_upload],
                    outputs=[image_display, audio_display, video_display]
                )

                # Action Buttons: Send and Clear
                with gr.Row():
                    submit_button = gr.Button(UI_BUTTON_SEND_TEXT, variant="primary", scale=2)
                    clear_button = gr.Button(UI_BUTTON_CLEAR_TEXT, scale=1)

            # --- Right Column: Chat Display ---
            with gr.Column(scale=2): # Right column for chat display
                # The main chatbot display area.
                # `avatar_images` provides custom avatars for user and AI.
                chatbot_display = gr.Chatbot(
                    label=UI_CHATBOT_LABEL,
                    height=700, # Adjusted height
                    avatar_images=(None, UI_AVATAR_AI),
                    type="messages" # Explicitly set the type
                )
        
        # Hidden state variable to maintain the chat history across interactions.
        # This is crucial for providing context to the LLM.
        chat_history_state = gr.State([])

        # --- Event Handling for Chat Submission ---
        # The `chat_fn` (from app.py) is the core logic handler for chat interactions.
        # It expects specific inputs and yields updates for multiple UI components.
        submit_outputs = [
            text_input, file_upload, chatbot_display, chat_history_state,
            image_display, audio_display, video_display
        ]

        submit_event_params = {
            "fn": chat_fn,
            "inputs": [text_input, file_upload, chat_history_state, model_selection],
            "outputs": submit_outputs,
            "show_progress": "full"
        }
        # Bind the submit action to both the button click and pressing Enter in the textbox.
        submit_button.click(**submit_event_params)
        text_input.submit(**submit_event_params)

        # --- Event Handling for Clearing Chat ---
        # The `clear_chat_history_fn_ui` (defined in this file) handles resetting the UI.
        # It returns updates for the same set of components as a chat submission.
        clear_outputs = [
            text_input, file_upload, chatbot_display, chat_history_state,
            image_display, audio_display, video_display
        ]
        clear_button.click(
            fn=clear_chat_history_fn_ui,
            inputs=[],
            outputs=clear_outputs,
            queue=False 
        )

        # --- LLM Initialization Check ---
        # If the LLM handler (from app.py) failed to initialize (e.g., missing API key),
        # display an error message prominently in the UI.
        if llm_handler_instance is None:
            gr.Error(UI_LLM_INIT_ERROR)
    
    return demo # Return the fully constructed Gradio Blocks instance.