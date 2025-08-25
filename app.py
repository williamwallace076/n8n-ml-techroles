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

# Bancos de dados para API
bancos_sql = {'sqlserver','mysql','postgresql','oracle','googlebigquery',
              'sqlite','saphana','snowflake','amazonauroraourds','mariadb',
              'db2','firebird','amazonredshift','microsoftaccess'}

bancos_nosql = {'s3','databricks','amazonathena','mongodb','hive','dynamodb',
                'presto','elaticsearch','redis','firebase','splunk','nenhum',
                'cassandra','hbase','googlefirestore','neo4j','excel'}

def map_bancos_user(input_str):
    x = str(input_str).lower()
    sql_count = sum(1 for s in bancos_sql if s in x)
    nosql_count = sum(1 for s in bancos_nosql if s in x)
    return sql_count, nosql_count

# Carregar modelo
MODEL_URL = os.getenv("MODEL_URL")
resp = requests.get(MODEL_URL)
resp.raise_for_status()
tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
print("Modelo temporário criado:", tmp_file.name)
with open(tmp_file.name, "wb") as f:
    f.write(resp.content)

saved = joblib.load(tmp_file.name)
model = saved["model"]
print("modelo:", model)
le_dict = saved["le_dict"]
feature_columns = saved["feature_columns"]

@app.get("/health")
def health():
    return {"status": "ok"}

# Input esperado
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

    # Ignorar 'vive_no_brasil' e 'pcd'
    for k in ["vive_no_brasil", "pcd", "nosql_count", "sql_count"]:
        d.pop(k, None)

    # Mapear bancos de dados
    sql_count, nosql_count = map_bancos_user(d.pop("bancos_de_dados", ""))
    d["sql_count"] = sql_count
    d["nosql_count"] = nosql_count

    # Aplicar LabelEncoder nas colunas restantes
    for col in le_dict:
        if col in d:
            d[col] = int(le_dict[col].transform([d[col]])[0])

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


if __name__ == "__main__":
    # teste = InputData(
    #     idade=25,
    #     genero="Masculino",
    #     etnia="Branco",
    #     pcd="Não",
    #     vive_no_brasil="Sim",
    #     estado_moradia="Pará (PA)",
    #     nivel_ensino="Pós-graduação",
    #     formacao="Computação / Engenharia de Software / Sistemas de Informação/ TI",
    #     tempo_experiencia_dados="de 3 a 4 anos",
    #     linguagens_preferidas="Python, JavaScript",
    #     bancos_de_dados="PostgreSQL, MongoDB",
    #     cloud_preferida="Amazon Web Services (AWS)"
    #     )
    # resultado = predict(teste)
    # print(resultado)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)


