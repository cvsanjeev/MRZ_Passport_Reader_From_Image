import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from mrz_reader.reader import MRZReader
import os
import base64

app = FastAPI()

# CORS (adjust the domain for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # safer: set to your vercel app URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models ONCE (reuse for every request)
reader = MRZReader(
    facedetection_protxt="./weights/face_detector/deploy.prototxt",
    facedetection_caffemodel="./weights/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
    segmentation_model="./weights/mrz_detector/mrz_seg.tflite",
    easy_ocr_params={"lang_list": ["en"], "gpu": False}
)

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        text_results, segmented_image, detected_face = reader.predict(
            img,
            do_facedetect=True,
            preprocess_config={}
        )

        # Convert processed images to base64 for frontend
        def encode_image(image):
            if image is not None:
                _, buffer = cv2.imencode('.jpg', image)
                return base64.b64encode(buffer).decode("utf-8")
            return None

        seg_img_b64 = encode_image(segmented_image)
        face_img_b64 = encode_image(detected_face)

        # Example: send text results and images (base64)
        return JSONResponse({
            "text_results": [
                {
                    "bbox": bbox,
                    "text": text,
                    "confidence": float(confidence)
                } for bbox, text, confidence in text_results
            ],
            "segmented_image": seg_img_b64,
            "detected_face": face_img_b64,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)})

