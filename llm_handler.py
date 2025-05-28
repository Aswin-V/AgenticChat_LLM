import os
from dotenv import load_dotenv
from PIL import Image
from typing import List, Any, Optional, Tuple, Dict # Corrected 'dict' to 'Dict' for consistency if used, or use lowercase 'dict'

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

class LLMHandler:
    """
    Handles interactions with the Large Language Model (LLM), including initialization,
    formatting chat history, and generating responses with multimodal inputs.
    """
    def __init__(self, provider="google", model_name="gemini-pro-vision", google_api_key: Optional[str] = None):
        self.provider = provider
        self.model_name = model_name
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.google_api_key and self.provider == "google":
            raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file or pass it directly.")

        self.llm = self._init_llm()

    def _init_llm(self):
        """Initializes the LLM client based on the specified provider."""
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

    def _format_chat_history_for_llm(self, chat_history: List[Dict[str, Any]]) -> List[Any]:
        """
        Converts Gradio's dictionary-based chat history format to Langchain's message objects.
        Handles cases where content might be a string or a tuple (text, filepath) for UI display.
        """
        messages = []
        for msg_dict in chat_history:
            role = msg_dict.get("role")
            content = msg_dict.get("content")
            
            actual_text_content: Optional[str] = None
            if isinstance(content, tuple): # User message with image: (text, filepath)
                actual_text_content = content[0] if content[0] else "" # Use text part
            elif isinstance(content, str): # Regular text message or assistant message
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
                                chat_history: List[Dict[str, Optional[str]]]) -> str:
        """
        Generates a response from the LLM based on text, optional media (image, audio, video),
        and chat history.
        """
        llm_messages = self._format_chat_history_for_llm(chat_history)
        
        current_input_content = []
        # Add text part if present
        if user_text:
            current_input_content.append({"type": "text", "text": user_text})

        # Add image part if present
        if base64_image_data and image_mime_type and self.provider == "google":
            data_uri = f"data:{image_mime_type};base64,{base64_image_data}"
            # Correct format for images as per Langchain documentation for Google GenAI
            current_input_content.append({"type": "image_url", "image_url": data_uri})
        elif base64_image_data: # Fallback if MIME type is missing or for other providers
            current_input_content.append({"type": "text", "text": "[Image uploaded, but current LLM may not process it directly]"})

        # Add audio part if present
        if base64_audio_data and audio_mime_type and self.provider == "google":
            # Correct format for audio as per Langchain documentation for Google GenAI
            current_input_content.append({
                "type": "media",
                "data": base64_audio_data,
                "mime_type": audio_mime_type
            })
        elif base64_audio_data:
            current_input_content.append({"type": "text", "text": "[Audio uploaded, but current LLM may not process it directly]"})

        # Add video part if present
        if base64_video_data and video_mime_type and self.provider == "google":
            # Correct format for video as per Langchain documentation for Google GenAI
            current_input_content.append({
                "type": "media",
                "data": base64_video_data,
                "mime_type": video_mime_type
            })
        elif base64_video_data:
            current_input_content.append({"type": "text", "text": "[Video uploaded, but current LLM may not process it directly]"})

        # Ensure there's some content to send if no new user input but history exists
        if not current_input_content:
             if not llm_messages: 
                return "Please provide some input."
             else: # Only history, no new input (should ideally be handled by UI not to call if no new content)
                current_input_content.append({"type": "text", "text": "..."}) # Placeholder if only history

        llm_messages.append(HumanMessage(content=current_input_content))
        
        try:
            ai_response_obj = await self.llm.ainvoke(llm_messages)
            return ai_response_obj.content
        except Exception as e:
            print(f"Error invoking LLM: {e}")
            if "API key not valid" in str(e):
                return "Error: The Google API key is not valid. Please check your .env file or API key settings."
            if "content" in str(e).lower() and "empty" in str(e).lower():
                 return "Error: The message content sent to the LLM was empty or invalid. Please try again."
            return f"Sorry, I encountered an error processing your request: {str(e)[:100]}..."

if __name__ == "__main__":
    import asyncio
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