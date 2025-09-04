import uvicorn
import json
import logging
from run_object_detection import run_object_detection
from handle_llm_question import handle_llm_question

import google.generativeai as genai

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from tensorflow.lite.python.interpreter import Interpreter


try:
    GOOGLE_API_KEY = "AIzaSyAQ3YnmEvkD712RSKT7Se4fiMZqnnfH4_I"
    genai.configure(api_key=GOOGLE_API_KEY)
    print(GOOGLE_API_KEY)
    llm_vision_model = genai.GenerativeModel('gemini-1.5-flash')
    logging.info("Google Gemini API configured successfully.")
except Exception as e:
    logging.error(
        f"Error configuring the Gemini API. Check your API Key. Error: {e}")
    llm_vision_model = None

logging.basicConfig(level=logging.INFO)

try:
    interpreter = Interpreter(model_path="../cv/model.tflite")
    interpreter.allocate_tensors()
    with open("../cv/labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    logging.info("Generic TFLite model and labels loaded successfully.")
except Exception as e:
    logging.error(
        f"Error loading generic_model.tflite or generic_labels.txt: {e}")
    exit()

# get the interpreter from the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

app = FastAPI()

# WebSocket endpoint


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("Client connected.")
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            msg_type = message.get("type")
            image = message.get("image")

            # Route the message based on the type
            if msg_type == "detection":
                detection_results = run_object_detection(
                    image, input_width, input_height, interpreter, output_details, input_details)
                response = {
                    "type": "detection_results",
                    "detections": detection_results
                }
                await websocket.send_text(json.dumps(response))

            elif msg_type == "question":
                llm_answer = await handle_llm_question(image, message.get("text"), llm_vision_model)
                response = {
                    "type": "llm_answer",
                    "text": llm_answer
                }
                await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        logging.info("Client disconnected.")
    except Exception as e:
        logging.error(f"Error in WebSocket connection: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
