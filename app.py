from fastapi import FastAPI, File, UploadFile
import pandas as pd
import joblib
from io import BytesIO

# Инициализация FastAPI-приложения
app = FastAPI()

# Загрузка ранее обученной ML-модели (В моем случаи плюс подпапка /Colab Notebooks)
model_path = "/content/drive/MyDrive/Colab Notebooks/laptop_price_model.pkl"
model = joblib.load(model_path)

# Эндпоинт для предсказания
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(BytesIO(content))
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}
