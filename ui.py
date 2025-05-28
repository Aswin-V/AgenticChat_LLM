import gradio as gr

from app import (
    chat_fn,
    initialize_llm_handler,
    handle_file_upload_for_preview, # Import the new handler
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

# UI-specific helper functions
def clear_chat_history_fn_ui():
    """Clears the chat interface components and media displays."""
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
    """
    with gr.Blocks(theme=UI_THEME) as demo:
        gr.Markdown(f"""
        # {UI_TITLE}
        {UI_DESCRIPTION}
        """)

        with gr.Row():
            with gr.Column(scale=1): # Left column for inputs and media display
                model_selection = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value=DEFAULT_MODEL,
                    label=UI_MODEL_LABEL,
                )
                model_selection.change(fn=initialize_llm_handler, inputs=[model_selection], outputs=None)

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
                
                # Media display components are defined inside the Accordion
                with gr.Accordion("Uploaded Media Preview", open=True): # Open by default
                    image_display = gr.Image(label="Image Preview", visible=False, interactive=False, height=200)
                    audio_display = gr.Audio(label="Audio Preview", visible=False, interactive=False) # type: ignore
                    video_display = gr.Video(label="Video Preview", visible=False, interactive=False, height=200) # type: ignore

                # Event handler for immediate media preview on file upload/clear
                # This is placed AFTER the display components are defined within the Accordion.
                file_upload.change(
                    fn=handle_file_upload_for_preview,
                    inputs=[file_upload],
                    outputs=[image_display, audio_display, video_display]
                )

                with gr.Row():
                    submit_button = gr.Button(UI_BUTTON_SEND_TEXT, variant="primary", scale=2)
                    clear_button = gr.Button(UI_BUTTON_CLEAR_TEXT, scale=1)

            with gr.Column(scale=2): # Right column for chat display
                chatbot_display = gr.Chatbot(
                    label=UI_CHATBOT_LABEL,
                    height=700, # Adjusted height
                    avatar_images=(None, UI_AVATAR_AI),
                    type="messages" # Explicitly set the type
                )
        
        chat_history_state = gr.State([])

        # Event handling for chat submission
        # chat_fn yields a 7-tuple for these outputs
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
        submit_button.click(**submit_event_params)
        text_input.submit(**submit_event_params)

        # Clear chat button
        # clear_chat_history_fn_ui also returns a 7-tuple
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

        if llm_handler_instance is None:
            gr.Error(UI_LLM_INIT_ERROR)
    
    return demo