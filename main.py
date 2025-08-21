from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import requests
import tempfile
import pandas as pd
from pydantic import BaseModel

# Definir app FastAPI
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

# Carregar modelo do GitHub (link RAW no .env)
MODEL_URL = os.getenv("MODEL_URL")

def load_model():
    if not MODEL_URL:
        raise ValueError("Variável de ambiente MODEL_URL não definida")
    resp = requests.get(MODEL_URL)
    resp.raise_for_status()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    with open(tmp_file.name, "wb") as f:
        f.write(resp.content)
    return joblib.load(tmp_file.name)

modelo = load_model()

# Rota de saúde
@app.get("/health")
def health():
    return {"status": "ok"}

# Rota de previsão
@app.post("/predict")
def predict(data: InputData):
    # Converter input para DataFrame
    df = pd.DataFrame([data.dict()])

    # Aqui você pode ajustar a ordem das colunas se necessário
    # exemplo: df = df[["idade","experiencia", ...]]
    # Ou usar variáveis de ambiente EXPECTED_COLUMNS

    pred = modelo.predict(df)
    return {"cargo_previsto": str(pred[0])}
