from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from app.utils.image_utils import preprocess_image
import tensorflow as tf
import numpy as np
import pickle

router = APIRouter(prefix="/api", tags=["Prediction"])

# Load model and class names
model = tf.keras.models.load_model("models/dog_emotion_custom_cnn_final.h5")

with open("models/evaluation_results.pkl", "rb") as f:
    eval_data = pickle.load(f)
class_names = eval_data["class_names"]

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = preprocess_image(contents)

        predictions = model.predict(img)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
