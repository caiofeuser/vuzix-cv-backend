import cv2
import numpy as np
import logging
import base64

def run_object_detection(image_base64,input_width, input_height, interpreter, output_details, input_details):
    """
    Run the object detection model on the image
    """
    try:
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return {}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (input_width, input_height))
        input_data = np.expand_dims(img_resized, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # This model returns 4 tensors of output: boxes, classes, scores and number of detections.
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]      # Coordinates [y_min, x_min, y_max, x_max]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]    # IDs of classes
        scores = interpreter.get_tensor(output_details[2]['index'])[0]     # Confidence scores
        
        detection_results = {"boxes": [], "classes": [], "scores": []}
        
        for i in range(len(scores)):
            if scores[i] > 0.5: # Confidence threshold
                box = boxes[i].tolist()
                
                y_min, x_min, y_max, x_max = box
                
                detection_results["boxes"].append([x_min, y_min, x_max, y_max])
                detection_results["classes"].append(int(classes[i]))
                detection_results["scores"].append(float(scores[i]))
        
        return detection_results

    except Exception as e:
        logging.error(f"Error during object detection: {e}")
        return {}
