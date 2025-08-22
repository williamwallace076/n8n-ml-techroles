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

# Carregar modelo
MODEL_URL = os.getenv("MODEL_URL")
resp = requests.get(MODEL_URL)
resp.raise_for_status()
tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
with open(tmp_file.name, "wb") as f:
    f.write(resp.content)

saved = joblib.load(tmp_file.name)
model = saved["model"]
le_dict = saved["le_dict"]
feature_columns = saved["feature_columns"]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    d = data.model_dump()

    # Ignorar 'vive_no_brasil' e 'pcd'
    for k in ["vive_no_brasil", "pcd"]:
        if k in d:
            d.pop(k)

    # Aplicar LabelEncoder em todas as colunas disponíveis
    for col in le_dict:
        if col in d:
            d[col] = le_dict[col].transform([d[col]])[0]

    # Criar DataFrame na ordem correta
    df_input = pd.DataFrame([{k: d.get(k, 0) for k in feature_columns}])

    # Predição
    pred_encoded = model.predict(df_input)
    cargo_previsto = le_dict["cargo"].inverse_transform(pred_encoded)

    return {
        "input": d,
        "cargo_previsto": cargo_previsto[0]
    }

# Exemplo de teste rápido
if __name__ == "__main__":
    teste = InputData(
        idade=25,
        genero="Masculino",
        etnia="Branco",
        pcd="Não",
        vive_no_brasil="Sim",
        estado_moradia="Pará (PA)",
        nivel_ensino="Pós-graduação",
        formacao="Computação / Engenharia de Software / Sistemas de Informação/ TI",
        tempo_experiencia_dados="5 anos",
        linguagens_preferidas="Python, JavaScript",
        bancos_de_dados="PostgreSQL, MongoDB",
        cloud_preferida="AWS"
    )
    resultado = predict(teste)
    print(resultado)
