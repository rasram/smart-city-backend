import google.generativeai as genai
from google.generativeai import GenerativeModel
import numpy as np
import io
import base64
import PIL.Image as Image
from fastapi.responses import JSONResponse
from fastapi import UploadFile
from dotenv import load_dotenv
import os

l1 = 100
l2 = 100
x_bin, y_bin = 150, 150

def config():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dotenv_path = os.path.join(project_root, '.env')

    load_dotenv(dotenv_path)

    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

    genai.configure(api_key=GOOGLE_API_KEY)
    global model
    model = GenerativeModel('gemini-1.5-pro')    

def inverse_kinematics(x, y):
    """Compute the joint angles for the SCARA robot."""
    D = (x**2 + y**2 - l1**2 - l2**2)
    
    if D < 0:
        raise ValueError("The point is unreachable.")

    theta2 = np.arccos(D / (2 * l1 * l2))
    theta1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    
    return theta1, theta2

async def process_image(file: UploadFile):
    if not file:
        return JSONResponse({'error': 'No selected file'}), 400

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    buffered = io.BytesIO()
    image.save(buffered, format=image.format or 'JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str, file

def run_chat_model(img_str, file):
    response = model.generate_content([
        {
            "inline_data": {
                "data": img_str,
                "mime_type": file.content_type or "image/jpeg"
            }
        },
        "Classify the waste shown in this image into one of the following categories: "
        "1. Plastic (e.g., bottles, bags, containers, wrappers), "
        "2. Paper (e.g., newspapers, cardboard, office paper, magazines), "
        "3. Organic (e.g., food waste, garden waste, leaves, fruit peels), "
        "4. Metal (e.g., cans, aluminum foil, metal scraps, wires), "
        "5. Glass (e.g., bottles, jars, broken glass), "
        "6. Electronic (e.g., old phones, chargers, computers, circuit boards), "
        "7. Textiles (e.g., clothes, fabric scraps, rugs), "
        "8. Hazardous (e.g., batteries, chemicals, medical waste, paint cans), "
        "9. Rubber (e.g., tires, rubber bands, hoses), "
        "10. Construction debris (e.g., bricks, cement, tiles, wood planks), "
        "11. Mixed waste (e.g., unsegregated trash, multiple categories in one item), "
        "or any other specific category not listed here. "
        "Provide a concise explanation for your classification, highlighting why the item fits into the category based on its visible properties, material, or context."
    ])

    classification_result = response.text
    return classification_result