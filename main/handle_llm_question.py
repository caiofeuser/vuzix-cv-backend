
import logging
import base64
import io
from PIL import Image

async def handle_llm_question(image_base64: str, question: str, llm_vision_model):
    """
    Send the image and the question to the LLM and return the answer.
    """
    if not llm_vision_model:
        return "Error: The Gemini API was not configured correctly."
    try:
        logging.info("Sending request to the LLM Gemini...")
        image_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_bytes))
        
        prompt_parts = [question, img]
        
        response = await llm_vision_model.generate_content_async(prompt_parts)
        
        logging.info("Response received from the LLM.")
        return response.text
    except Exception as e:
        logging.error(f"Error communicating with the Gemini API: {e}")
        return "Sorry, I couldn't process your question."