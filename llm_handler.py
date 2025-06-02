# llm_handler.py
# This module is responsible for all interactions with the Large Language Model (LLM).
# It encapsulates the logic for initializing the LLM client (e.g., Google Generative AI),
# formatting chat history appropriately for the LLM, and generating responses,
# including handling multimodal inputs like text, images, audio, and video.

import os
from dotenv import load_dotenv
from typing import List, Any, Optional, Dict # Use lowercase 'dict' for type hints

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
# It's crucial for this to be called early, so os.getenv can access these variables.
load_dotenv()

class LLMHandler:
    """
    Handles interactions with the Large Language Model (LLM), including initialization,
    formatting chat history, and generating responses with multimodal inputs.
    """
    def __init__(self, provider: str = "google", model_name: str = "gemini-pro-vision", google_api_key: Optional[str] = None):
        """
        Initializes the LLMHandler.

        Args:
            provider (str): The LLM provider (e.g., "google", "ollama", "openai").
                            Currently, only "google" is fully implemented.
            model_name (str): The specific model name to use from the provider.
            google_api_key (Optional[str]): The Google API key. If None, it's fetched
                                            from the GOOGLE_API_KEY environment variable.
        """
        self.provider = provider
        self.model_name = model_name
        self.google_api_key: Optional[str] = google_api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.google_api_key and self.provider == "google":
            raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file or pass it directly.")

        self.llm = self._init_llm()

    def _init_llm(self):
        """
        Initializes the LLM client based on the specified provider and model.
        
        Returns:
            An instance of the LLM client (e.g., ChatGoogleGenerativeAI).
        Raises:
            ValueError: If an unsupported LLM provider is specified.
        """
        if self.provider == "google":
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.google_api_key
            )
        elif self.provider == "ollama":
            print("Ollama provider not yet implemented. Using Google as fallback.")
            return ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.google_api_key)
        elif self.provider == "openai":
            print("OpenAI provider not yet implemented. Using Google as fallback.")
            return ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.google_api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _format_chat_history_for_llm(self, chat_history: List[dict[str, Any]]) -> List[Any]:
        """
        Converts Gradio's dictionary-based chat history format to Langchain's message objects.
        Gradio chat history is a list of dictionaries, where each dictionary has "role"
        (user or assistant) and "content". The content from user messages might be
        a tuple (text, filepath) if it included a file upload for UI display purposes,
        or just a string. This function extracts the textual part for the LLM.

        Args:
            chat_history (List[dict[str, Any]]): The chat history in Gradio's format.
        Returns:
            List[Any]: A list of Langchain message objects (HumanMessage, AIMessage).
        """
        messages = []
        for msg_dict in chat_history:
            role = msg_dict.get("role")
            content = msg_dict.get("content")
            
            actual_text_content: Optional[str] = None
            # Gradio's chatbot component can store user messages with files as (text, filepath_or_url_tuple)
            # We only need the text part for the LLM history.
            if isinstance(content, tuple):
                actual_text_content = content[0] if content[0] else "" # Use text part
            elif isinstance(content, str): # Regular text message or AI message
                actual_text_content = content

            # Ensure actual_text_content is not None before creating a message.
            # An empty string is valid content for an LLM message.
            if actual_text_content is not None:
                if role == "user":
                    messages.append(HumanMessage(content=actual_text_content))
                elif role == "assistant":
                    messages.append(AIMessage(content=actual_text_content))
                # System messages could be handled here too if needed
        return messages

    async def generate_response(self, user_text: str,
                                base64_image_data: Optional[str],
                                image_mime_type: Optional[str],
                                base64_audio_data: Optional[str],
                                audio_mime_type: Optional[str],
                                base64_video_data: Optional[str],
                                video_mime_type: Optional[str],
                                chat_history: List[dict[str, Optional[str]]]) -> str:
        """
        Generates a response from the LLM based on text, optional media (image, audio, video),
        and chat history.

        Args:
            user_text (str): The text input from the user.
            base64_image_data (Optional[str]): Base64 encoded image data.
            image_mime_type (Optional[str]): MIME type of the image.
            base64_audio_data (Optional[str]): Base64 encoded audio data.
            audio_mime_type (Optional[str]): MIME type of the audio.
            base64_video_data (Optional[str]): Base64 encoded video data.
            video_mime_type (Optional[str]): MIME type of the video.
            chat_history (List[dict[str, Optional[str]]]): The conversation history.

        Returns:
            str: The LLM's response as a string.
        """
        llm_messages: List[Any] = self._format_chat_history_for_llm(chat_history)
        
        current_input_content = []
        # Add text part if present
        if user_text:
            current_input_content.append({"type": "text", "text": user_text})

        # Add image part if present
        if base64_image_data and image_mime_type and self.provider == "google":
            data_uri = f"data:{image_mime_type};base64,{base64_image_data}"
            # Format for multimodal input (image) for Google Generative AI models via Langchain
            current_input_content.append({"type": "image_url", "image_url": data_uri})
        elif base64_image_data: 
            # Fallback message if MIME type is missing or for providers not supporting direct image data
            current_input_content.append({"type": "text", "text": "[Image uploaded, but current LLM may not process it directly]"})

        # Add audio part if present
        if base64_audio_data and audio_mime_type and self.provider == "google":
            # Correct format for audio as per Langchain documentation for Google GenAI
            current_input_content.append({
                "type": "media",
                "data": base64_audio_data,
                "mime_type": audio_mime_type
            })
        elif base64_audio_data: # Fallback for audio
            current_input_content.append({"type": "text", "text": "[Audio uploaded, but current LLM may not process it directly]"})

        # Add video part if present
        if base64_video_data and video_mime_type and self.provider == "google":
            # Correct format for video as per Langchain documentation for Google GenAI
            current_input_content.append({
                "type": "media",
                "data": base64_video_data,
                "mime_type": video_mime_type
            })
        elif base64_video_data: # Fallback for video
            current_input_content.append({"type": "text", "text": "[Video uploaded, but current LLM may not process it directly]"})

        # Ensure there's some content to send.
        # If current_input_content is empty (e.g., only history was passed and no new user input),
        # an error might occur with the LLM.
        if not current_input_content:
             if not llm_messages: 
                return "Please provide some input."
             else: 
                # This case (only history, no new input) should ideally be handled by the UI
                # to prevent calling the LLM without new content.
                # However, as a safeguard, send a minimal text.
                current_input_content.append({"type": "text", "text": "..."})
        
        llm_messages.append(HumanMessage(content=current_input_content))
        
        try:
            # Asynchronously invoke the LLM with the prepared messages.
            ai_response_obj = await self.llm.ainvoke(llm_messages)
            return str(ai_response_obj.content) # Ensure content is string
        except Exception as e:
            # Catch potential errors during LLM interaction and return a user-friendly message.
            print(f"Error invoking LLM: {e}")
            if "API key not valid" in str(e):
                return "Error: The Google API key is not valid. Please check your .env file or API key settings."
            if "content" in str(e).lower() and "empty" in str(e).lower():
                 return "Error: The message content sent to the LLM was empty or invalid. Please try again."
            return f"Sorry, I encountered an error processing your request: {str(e)[:100]}..."

if __name__ == "__main__":
    import asyncio
    from PIL import Image # For dummy image creation in test
    import base64
    
    async def test_handler():
        try:
            # Test with a model known to support text and potentially images
            handler = LLMHandler(model_name="gemini-1.5-flash-latest")
            history = []

            # Test text-only interaction
            response = await handler.generate_response("Hello, Gemini!", None, None, None, None, None, None, history) # type: ignore
            print(f"Text-only response: {response}")
            history.append({"role": "user", "content": "Hello, Gemini!"})
            history.append({"role": "assistant", "content": response})

            response_with_history = await handler.generate_response("What was my first question?", None, None, None, None, None, None, history)
            print(f"Response with history: {response_with_history}")

            # Image Test
            image_path_test = "dummy_test_image.jpg"
            if not os.path.exists(image_path_test):
                try:
                    img_test = Image.new('RGB', (60, 30), color='red')
                    img_test.save(image_path_test, "JPEG")
                    print(f"Created {image_path_test} for testing.")
                except Exception as e:
                    print(f"Could not create dummy image: {e}")
            if os.path.exists(image_path_test):
                with open(image_path_test, "rb") as img_file:
                    b64_img = base64.b64encode(img_file.read()).decode("utf-8") # type: ignore
                img_response = await handler.generate_response("Describe this image.", b64_img, "image/jpeg", None, None, None, None, [])
                print(f"Image response: {img_response}")
            else:
                print(f"Skipping vision test as {image_path_test} was not found/created.")

            # Audio Test (Conceptual - requires a dummy audio file and model support for 'audio_url')
            # audio_path_test = "dummy_test_audio.wav"
            # Create a dummy wav file if you want to test this part thoroughly
            # if os.path.exists(audio_path_test):
            #     with open(audio_path_test, "rb") as audio_file:
            #         b64_audio = base64.b64encode(audio_file.read()).decode("utf-8") # type: ignore
            #     audio_response = await handler.generate_response("Transcribe this audio.", None, None, b64_audio, "audio/wav", None, None, [])
            #     print(f"Audio response: {audio_response}")

            # Video Test (Conceptual - requires a dummy video file and model support for 'video_url')
            # video_path_test = "dummy_test_video.mp4"
            # Create a dummy mp4 file if you want to test this part thoroughly
            # if os.path.exists(video_path_test):
            #     with open(video_path_test, "rb") as video_file:
            #         b64_video = base64.b64encode(video_file.read()).decode("utf-8") # type: ignore
            #     video_response = await handler.generate_response("Describe this video.", None, None, None, None, b64_video, "video/mp4", [])
            #     print(f"Video response: {video_response}")
        except ValueError as ve:
            print(ve)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    asyncio.run(test_handler())