# minimal_test.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
import base64 # Import the base64 module
from dotenv import load_dotenv

load_dotenv() # Make sure GOOGLE_API_KEY is in your .env

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("GOOGLE_API_KEY not found in .env")
    exit()

try:
    # Replace with your desired model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", google_api_key=GOOGLE_API_KEY)

    # Create a dummy image or use a real one
    image_path = "/home/avc/MEAA/llm/llmwithSearch/dummy_image.jpg"

    # Read the image file and encode it in base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Determine the image type (e.g., png, jpeg) for the data URI
    # For simplicity, we'll assume jpeg for the dummy image.
    # For a more robust solution, you might use a library like 'imghdr' or Pillow's format attribute.
    image_type = "jpeg" if image_path.endswith(".jpg") or image_path.endswith(".jpeg") else "png"

    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image in detail."},
            {"type": "image_url", "image_url": f"data:image/{image_type};base64,{encoded_image}"}
        ]
    )

    print("Sending request to LLM...")
    response = llm.invoke([message]) # Use invoke for synchronous test
    print("\nLLM Response:")
    print(response.content)

except Exception as e:
    print(f"An error occurred: {e}")
