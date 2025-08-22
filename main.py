from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import requests
import tempfile
import pandas as pd
from pydantic import BaseModel
from dotenv import load_dotenv

# Carregar variáveis do .env
load_dotenv()

app = FastAPI(title="Previsão de Cargos")

# Configuração de CORS
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Classe para os parâmetros recebidos
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
    linguagens_Preferidas: str
    bancos_de_dados: str
    cloud_preferida: str

# Variáveis do .env
MODEL_URL = os.getenv("MODEL_URL")
EXPECTED_COLUMNS = os.getenv("EXPECTED_COLUMNS", "").split(",")

def load_model():
    resp = requests.get(MODEL_URL)
    resp.raise_for_status()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    with open(tmp_file.name, "wb") as f:
        f.write(resp.content)
    return joblib.load(tmp_file.name)

modelo = load_model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])

    # Reordenar colunas se definido no .env
    if EXPECTED_COLUMNS and EXPECTED_COLUMNS[0] != "":
        df = df[EXPECTED_COLUMNS]

    pred = modelo.predict(df)
    return {
        "input": data.dict(),
        "cargo_previsto": str(pred[0])
    }
