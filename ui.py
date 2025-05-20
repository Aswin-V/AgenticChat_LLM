import gradio as gr

# Import necessary functions and variables from the app module
# Assuming app.py is in the same directory
from app import (
    chat_fn,
    initialize_llm_handler,
    llm_handler_instance, # This is the global instance from app.py
    DEFAULT_MODEL,
    AVAILABLE_MODELS
)

# UI-specific helper functions
def clear_chat_history_fn_ui():
    """Clears the chat interface components."""
    # Expected outputs: text_input, file_upload, chatbot_display, chat_history_state
    return "", None, [], []

def create_gradio_ui() -> gr.Blocks:
    """
    Creates and returns the Gradio Blocks UI for the chat application.
    """
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"""
        # Agentic Chat Application
        Powered by Langchain and Google Generative AI ({DEFAULT_MODEL}).
        Type your message, or upload an image, audio, or video file.
        """)

        # Model selection dropdown
        model_selection = gr.Dropdown(
            choices=AVAILABLE_MODELS,
            value=DEFAULT_MODEL, # Default model for the dropdown
            label="Select Model",
        )

        # Update LLM handler when the model selection changes
        # This calls initialize_llm_handler from app.py, which updates app.llm_handler_instance
        model_selection.change(fn=initialize_llm_handler, inputs=[model_selection], outputs=None)

        # Chatbot display
        chatbot_display = gr.Chatbot(
            label="Chat Window",
            height=600,
            type="messages",
            avatar_images=(None, "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png")
        )
        
        # State to store chat history in Gradio's format
        chat_history_state = gr.State([]) 

        with gr.Row():
            text_input = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here, or upload a file and ask about it...",
                scale=3,
                autofocus=True,
            )
            file_upload = gr.File(
                label="Upload File (Image, Audio, Video, etc.)",
                file_types=["image", "audio", "video", ".pdf", ".txt", ".json", ".csv"],
                scale=1,
            )

        submit_button = gr.Button("Send", variant="primary")

        # Event handling for chat submission
        submit_event_params = {
            "fn": chat_fn, # chat_fn from app.py
            "inputs": [text_input, file_upload, chat_history_state, model_selection],
            "outputs": [text_input, file_upload, chatbot_display, chat_history_state],
            "show_progress": "full"
        }
        submit_button.click(**submit_event_params)
        text_input.submit(**submit_event_params)

        # Clear chat button
        clear_button = gr.Button("Clear Chat")
        clear_button.click(
            fn=clear_chat_history_fn_ui, # Use the local UI clear function
            inputs=[],
            outputs=[text_input, file_upload, chatbot_display, chat_history_state],
            queue=False 
        )

        # Display error if LLM handler (from app.py) is None
        # This check refers to the llm_handler_instance imported from app.py
        if llm_handler_instance is None:
            gr.Error("LLM Handler could not be initialized. This is likely due to a missing or invalid GOOGLE_API_KEY in your .env file. Please check the console for more details and set up the API key.")
    
    return demo