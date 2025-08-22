from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import requests
import tempfile
import pandas as pd
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Previsão de Cargos")

# CORS
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo esperado
class InputData(BaseModel):
    idade: int
    genero: str
    etnia: str
    pcd: str
    vive_no_brasil: str
    estado_moradia: str
    nivel_ensino: str
    formacao: str
    tempo_experiencia_dados: str
    linguagens_preferidas: str
    bancos_de_dados: str
    cloud_preferida: str

# Carregar modelo + encoder do cargo
MODEL_URL = os.getenv("MODEL_URL")
resp = requests.get(MODEL_URL)
resp.raise_for_status()
tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
with open(tmp_file.name, "wb") as f:
    f.write(resp.content)

pipeline, le_cargo = joblib.load(tmp_file.name)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    # Converter para DataFrame
    df = pd.DataFrame([data.dict()])

    # Rodar pipeline
    pred_encoded = pipeline.predict(df)

    # Decodificar cargo
    cargo_previsto = le_cargo.inverse_transform(pred_encoded)

    return {
        "input": data.dict(),
        "cargo_previsto": cargo_previsto[0]
    }
