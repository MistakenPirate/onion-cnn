import io
import os

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from onion.predict import load_model, predict_image

WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS", "model_weights.pth")

app = FastAPI(title="Onion Quality Classifier")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
model, device = None, None


@app.on_event("startup")
def startup():
    global model, device
    model, device = load_model(WEIGHTS_PATH)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    result = predict_image(image, model, device)
    return result
