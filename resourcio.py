
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import cv2
import numpy as np
import os
import requests
import json

app = FastAPI()


class LLMClient:
    def __init__(self, api_key: str = "AIzaSyCDoGcbBYTTHpWdC1qNptOoErYtIF0M4w4", model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def analyze_fit(self, avatar_img_path: str, dress_img_path: str, user_stats: dict, prompt_template: str):
        """
        Sends request to Gemini API with prompt and user stats.
        """
        headers = {"Content-Type": "application/json"}
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"

      
        prompt = prompt_template.format(**user_stats)

        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ],
            "generationConfig": {"temperature": 0.4}
        }

        resp = requests.post(url, headers=headers, json=payload)

        if resp.status_code != 200:
            return {"fit_category": "Error", "comment": f"LLM API failed: {resp.text}"}

        data = resp.json()
        try:
            text_output = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return {"fit_category": "Error", "comment": f"Invalid LLM response: {str(e)}"}

       
        try:
            structured_output = json.loads(text_output)
            return structured_output
        except:
            return {"fit_category": "Unstructured", "comment": text_output}


def simple_fit_analysis(avatar_img, dress_img):
    avatar_gray = cv2.cvtColor(np.array(avatar_img), cv2.COLOR_RGB2GRAY)
    dress_gray = cv2.cvtColor(np.array(dress_img), cv2.COLOR_RGB2GRAY)

    _, avatar_thresh = cv2.threshold(avatar_gray, 250, 255, cv2.THRESH_BINARY_INV)
    avatar_contours, _ = cv2.findContours(avatar_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    avatar_box = cv2.boundingRect(max(avatar_contours, key=cv2.contourArea))

    _, dress_thresh = cv2.threshold(dress_gray, 250, 255, cv2.THRESH_BINARY_INV)
    dress_contours, _ = cv2.findContours(dress_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dress_box = cv2.boundingRect(max(dress_contours, key=cv2.contourArea))

    avatar_w, avatar_h = avatar_box[2], avatar_box[3]
    dress_w, dress_h = dress_box[2], dress_box[3]

    width_ratio = dress_w / avatar_w
    height_ratio = dress_h / avatar_h

    if width_ratio > 1.2:
        return "Dress may be too wide"
    elif width_ratio < 0.7:
        return "Dress may be too tight"
    elif height_ratio > 1.2:
        return "Dress may be too long"
    elif height_ratio < 0.7:
        return "Dress may be too short"
    else:
        return "Dress looks like a good fit!"


@app.get("/")
def root():
    return {"message": "Hello from Resourcio!"}



@app.post("/virtual-tryon/")
async def virtual_tryon(dress: UploadFile = File(...), avatar: UploadFile = File(...)):
    dress_img = Image.open(dress.file).convert("RGBA")
    avatar_img = Image.open(avatar.file).convert("RGB")

    dress_resized = dress_img.resize(
        (avatar_img.width, int(dress_img.height * avatar_img.width / dress_img.width))
    )

    avatar_img.paste(
        dress_resized,
        (0, avatar_img.height - dress_resized.height),
        dress_resized
    )

    fit_status = simple_fit_analysis(avatar_img, dress_img)

    # output_path = "output_avatar.png"
    # avatar_img.save(output_path)

    return JSONResponse(content={"fit_status": fit_status})



@app.post("/fit-analysis/")
async def fit_analysis(
    dress: UploadFile = File(...),
    avatar: UploadFile = File(...),
    height: str = Form(...),
    weight: str = Form(...),
    age: str = Form(...)
):
    avatar_path = f"temp_avatar_{avatar.filename}.png"
    dress_path = f"temp_dress_{dress.filename}.png"
    Image.open(avatar.file).convert("RGB").save(avatar_path, format="PNG")
    Image.open(dress.file).convert("RGBA").save(dress_path, format="PNG")

    user_stats = {"height": height, "weight": weight, "age": age}

    prompt_template = """
    You are a fashion fit assistant.
    Based on the avatar image and dress image provided, along with user stats:
    Height: {height}, Weight: {weight}, Age: {age}

    Please return a JSON response with two fields:
    - "fit_category": One of ["Good Fit", "Too Tight", "Too Loose", "Too Long", "Too Short"]
    - "comment": A short explanation (max 30 words).
    """

    client = LLMClient()
    structured_output = client.analyze_fit(avatar_path, dress_path, user_stats, prompt_template)

    if os.path.exists(avatar_path): os.remove(avatar_path)
    if os.path.exists(dress_path): os.remove(dress_path)

    return JSONResponse(content=structured_output)
