from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import requests
import tempfile
import pandas as pd
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import unidecode

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

# ==========================
# Funções utilitárias
# ==========================
def normalize_text(v):
    if v is None:
        return "desconhecido"
    v = str(v)
    v = unidecode.unidecode(v).lower().strip().replace(" ", "")
    return v if v != "" else "desconhecido"

bancos_sql = {'sqlserver','mysql','postgresql','oracle','googlebigquery',
              'sqlite','saphana','snowflake','amazonaurora','mariadb',
              'db2','firebird','amazonredshift','microsoftaccess'}

bancos_nosql = {'s3','databricks','amazonathena','mongodb','hive','dynamodb',
                'presto','elasticsearch','redis','firebase','splunk','nenhum',
                'cassandra','hbase','googlefirestore','neo4j','excel'}

def map_bancos_user(input_str):
    x = normalize_text(input_str)
    sql_count = sum(1 for s in bancos_sql if s in x)
    nosql_count = sum(1 for s in bancos_nosql if s in x)
    return sql_count, nosql_count

# ==========================
# Carregar modelo
# ==========================
MODEL_URL = os.getenv("MODEL_URL")
# print("URL do modelo:", MODEL_URL)

resp = requests.get(MODEL_URL)
resp.raise_for_status()
tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
with open(tmp_file.name, "wb") as f:
    f.write(resp.content)

saved = joblib.load(tmp_file.name)
model = saved["model"]
le_dict = saved["le_dict"]
feature_columns = saved["feature_columns"]

# ==========================
# Rotas
# ==========================
@app.get("/health")
def health():
    return {"status": "ok"}

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

@app.post("/predict")
def predict(data: InputData):
    d = data.model_dump()

    # Ignorar campos que não entram no modelo
    for k in ["vive_no_brasil", "pcd"]:
        d.pop(k, None)

    # Mapear bancos
    sql_count, nosql_count = map_bancos_user(d.pop("bancos_de_dados", ""))
    d["sql_count"] = sql_count
    d["nosql_count"] = nosql_count

    # Normalizar e aplicar LabelEncoder
    for col, le in le_dict.items():
        if col in d:
            d[col] = normalize_text(d[col])
            try:
                d[col] = int(le.transform([d[col]])[0])
            except ValueError:
                d[col] = 0  # fallback para desconhecido

    # Criar DataFrame na ordem correta
    df_input = pd.DataFrame([{k: d.get(k, 0) for k in feature_columns}])

    # Predição
    pred_encoded = model.predict(df_input)
    cargo_previsto = le_dict["cargo"].inverse_transform(pred_encoded)
    cargo_previsto_str = str(cargo_previsto[0])

    return {
        "input": d,
        "cargo_previsto": cargo_previsto_str
    }

# ==========================
# Teste rápido
# ==========================
if __name__ == "__main__":
    teste = InputData(
        idade=25,
        genero="masculino",
        etnia="branco",
        pcd="nao",
        vive_no_brasil="sim",
        estado_moradia="para(pa)",
        nivel_ensino="posgraduacao",
        formacao="computacao/engenhariadesoftware/sistemasdeinformacao/ti",
        tempo_experiencia_dados="de3a4anos",
        linguagens_preferidas="python,javascript",
        bancos_de_dados="postgresql,mongodb",
        cloud_preferida="amazonwebservices(aws)"
    )
    resultado = predict(teste)
    print(resultado)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
