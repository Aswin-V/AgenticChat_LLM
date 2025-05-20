import gradio as gr
from PIL import Image, UnidentifiedImageError # Keep UnidentifiedImageError
import os
from typing import List, Tuple, Optional
from typing import AsyncGenerator, Any # Import AsyncGenerator

from llm_handler import LLMHandler
from utils import (
    get_image_mime_type, get_audio_mime_type, get_video_mime_type,
    encode_file_to_base64, get_default_prompt_for_media
)

# --- Configuration ---
# You can change the default LLM provider and model here
DEFAULT_LLM_PROVIDER = "google" 
# Use "gemini-pro" for text-only, "gemini-pro-vision" for text+image
DEFAULT_MODEL = "gemini-1.5-flash" 
AVAILABLE_MODELS = [
    DEFAULT_MODEL,
    "gemini-1.5-pro",
    "gemini-2.5-flash-preview-04-17"  # Assuming this is a valid model name
]

# Initialize LLM Handler
llm_handler_instance: Optional[LLMHandler] = None

# Make the LLM choice dynamic via UI elements
def initialize_llm_handler(model_name: str) -> Optional[LLMHandler]:
    global llm_handler_instance
    try:
        llm_handler_instance = LLMHandler(provider=DEFAULT_LLM_PROVIDER, model_name=model_name)
        return llm_handler_instance
    except ValueError as e:
        print(f"Error initializing LLMHandler: {e}")
        return None

# --- File Processing Logic ---
def process_uploaded_file(file_path: str, file_name: str) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """
    Processes an uploaded file to determine its type, MIME type,
    and base64 encode it if it's a supported media type.
    Returns: (base64_data, mime_type, media_category ("image", "audio", "video", "file"), info_message)
    """
    base64_data: Optional[str] = None
    mime_type: Optional[str] = None
    media_category: Optional[str] = None
    info_message: str = ""

    # Try to process as an image first
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
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext in ['.wav', '.mp3', '.ogg', '.flac']:
            mime_type = get_audio_mime_type(file_name)
            base64_data = encode_file_to_base64(file_path)
            if base64_data:
                media_category = "audio"
                info_message = f"[Audio file '{file_name}' uploaded]"
            else:
                info_message = f"[Error processing audio file '{file_name}']"
        elif file_ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            mime_type = get_video_mime_type(file_name)
            base64_data = encode_file_to_base64(file_path)
            if base64_data:
                media_category = "video"
                info_message = f"[Video file '{file_name}' uploaded]"
            else:
                info_message = f"[Error processing video file '{file_name}']"
        else:
            media_category = "file"
            info_message = f"[File '{file_name}' uploaded. Type not fully supported for direct processing.]"

    # If base64_data is None after attempting to process as media, ensure mime_type is also None
    if not base64_data:
        mime_type = None
        # media_category might still be "audio" or "video" if encoding failed,
        # but it won't be passed to LLM if base64_data is None.

    return base64_data, mime_type, media_category, info_message


# --- Gradio Interface Logic ---
async def chat_fn(
    user_message: str,
    file_obj: Optional[gr.File], # Gradio's File component output
    chat_history: List[dict[str, Optional[str]]],
    model_name: str  # Add model_name as input
) -> AsyncGenerator[Tuple[str, Optional[gr.File], List[dict[str, Optional[str]]], List[dict[str, Optional[str]]]], None]:
    """
    Handles the chat interaction.
    Processes user input (text and/or file), calls the LLM, and updates history.
    """
    # Ensure LLM handler is initialized
    if llm_handler_instance is None:
        error_message = "LLM Handler not initialized. Please check API key and configurations."
        updated_history = chat_history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": error_message}]
        yield "", None, updated_history, updated_history
        return

    # Initialize media data variables
    file_info_message: str = ""
    base64_image_data: Optional[str] = None
    image_mime_type: Optional[str] = None
    base64_audio_data: Optional[str] = None
    audio_mime_type: Optional[str] = None
    base64_video_data: Optional[str] = None
    video_mime_type: Optional[str] = None

    # Process uploaded file if any
    if file_obj is not None:
        temp_file_path = file_obj.name
        file_name = os.path.basename(temp_file_path)
        b64_data, mime, category, info_msg = process_uploaded_file(temp_file_path, file_name)
        file_info_message = info_msg

        if category == "image" and b64_data and mime:
            base64_image_data, image_mime_type = b64_data, mime
        elif category == "audio" and b64_data and mime:
            base64_audio_data, audio_mime_type = b64_data, mime
        elif category == "video" and b64_data and mime:
            base64_video_data, video_mime_type = b64_data, mime

    # Determine the full user prompt and text for LLM
    full_user_prompt = user_message
    text_for_llm = user_message
    media_processed = False

    if base64_image_data and image_mime_type:
        media_processed = True
        full_user_prompt = f"{file_info_message}\n\n{user_message}" if user_message else file_info_message
        if not user_message: text_for_llm = get_default_prompt_for_media("image")
    elif base64_audio_data and audio_mime_type:
        media_processed = True
        full_user_prompt = f"{file_info_message}\n\n{user_message}" if user_message else file_info_message
        if not user_message: text_for_llm = get_default_prompt_for_media("audio")
    elif base64_video_data and video_mime_type:
        media_processed = True
        full_user_prompt = f"{file_info_message}\n\n{user_message}" if user_message else file_info_message
        if not user_message: text_for_llm = get_default_prompt_for_media("video")
    elif file_info_message: # Other file types or processing errors
        full_user_prompt = f"{file_info_message}\n\n{user_message}" if user_message else file_info_message

    # Handle case with no user message and no processable file
    if not user_message and not media_processed and not file_info_message: # If truly no input at all
        if not chat_history:
            updated_history = chat_history + [{"role": "assistant", "content": "Please type a message or upload a file."}]
            yield "", None, updated_history, updated_history
        else:
            updated_history = chat_history + [{"role": "assistant", "content": "Please provide new input or ask a question."}]
            yield "", None, updated_history, updated_history
        return
    elif not user_message and not media_processed and file_info_message: # File uploaded but not processable media, and no text
        # Let the file_info_message be the prompt
        text_for_llm = "" # No specific user text, LLM will see file_info_message in history
        # full_user_prompt is already set to file_info_message

    # Append user's turn to chat history for display
    chat_history.append({"role": "user", "content": full_user_prompt})
    yield "", None, chat_history, chat_history # Update UI with user's message

    # Call LLM if there's something to process
    if text_for_llm or base64_image_data or base64_audio_data or base64_video_data:
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
        # This case should ideally be caught by earlier checks,
        # but as a fallback if only a non-processable file_info_message was shown.
        if not chat_history[-1].get("content"): # If user prompt was empty
             chat_history[-1]["content"] = "..." # Placeholder if user prompt was truly empty
        chat_history.append({"role": "assistant", "content": "I received the file information. How can I help you with it?"})

    yield "", None, chat_history, chat_history # Update UI with AI's response

# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # Agentic Chat Application
    Powered by Langchain and Google Generative AI ({DEFAULT_MODEL}).
    Type your message, or upload an image, audio, or video file.
    """)

    # Model selection dropdown
    model_selection = gr.Dropdown(
        choices=AVAILABLE_MODELS,
        value=DEFAULT_MODEL,
        label="Select Model",
    )

    # Initialize LLM handler with the default model
    # This will be called once when the script starts
    if llm_handler_instance is None: # Ensure it's only initialized once at startup if not already
        llm_handler_instance = initialize_llm_handler(model_selection.value)

    # Update LLM handler when the model selection changes
    model_selection.change(initialize_llm_handler, inputs=[model_selection], outputs=None) # No direct output to a component

    # Chatbot display
    chatbot_display = gr.Chatbot(
        label="Chat Window",
        height=600,
        type="messages", # Use new message format
        avatar_images=(None, "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png") # (user, AI)
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
            # More generic file types, specific handling is done in process_uploaded_file
            file_types=["image", "audio", "video", ".pdf", ".txt", ".json", ".csv"],
            scale=1,
        )

    submit_button = gr.Button("Send", variant="primary")

    # Event handling for chat submission
    submit_event_params = {
        "fn": chat_fn,
        "inputs": [text_input, file_upload, chat_history_state, model_selection],
        "outputs": [text_input, file_upload, chatbot_display, chat_history_state],
        "show_progress": "full"
    }
    submit_button.click(**submit_event_params)
    text_input.submit(**submit_event_params)

    # Clear chat button
    def clear_chat_history_fn():
        return [], [], None, [] # text_input, file_upload, chatbot_display, chat_history_state

    clear_button = gr.Button("Clear Chat")
    clear_button.click(
        fn=clear_chat_history_fn,
        inputs=[],
        outputs=[text_input, file_upload, chatbot_display, chat_history_state],
        queue=False 
    )

    # Display error if LLM handler failed to initialize
    if llm_handler_instance is None:
        gr.Error("LLM Handler could not be initialized. This is likely due to a missing or invalid GOOGLE_API_KEY in your .env file. Please check the console for more details and set up the API key.")


if __name__ == "__main__":
    demo.queue() 
    demo.launch(debug=True)