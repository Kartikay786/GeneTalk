from PIL import Image
import numpy as np
import io

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")
