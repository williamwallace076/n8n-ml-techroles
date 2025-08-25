import pandas as pd
import unidecode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ==========================
# 1. Carregar e padronizar
# ==========================
url = "https://raw.githubusercontent.com/Weverton-Cristian/Processing-Dataset_-Intelligent_Agent/master/dataset.csv"
df = pd.read_csv(url)

# Normalizar strings: minúsculo, sem acento, sem espaços
def normalize_text(v):
    if v is None:
        return "desconhecido"
    v = str(v)
    v = unidecode.unidecode(v).lower().strip().replace(" ", "")
    return v if v != "" else "desconhecido"

df = df.applymap(lambda x: normalize_text(x) if isinstance(x, str) else x)

# ==========================
# 2. Limpeza básica
# ==========================
df = df.drop_duplicates()
df = df.dropna(subset=["cargo"])
df = df[df["cargo"].astype(str).str.strip() != ""]

# ==========================
# 3. Mapear cargos em grupos
# ==========================
mapeamento_cargos = {
    'analistadebi/bianalyst': 'analiseeestrategia',
    'analistadenegocios/businessanalyst': 'analiseeestrategia',
    'analistadeinteligenciademercado/marketintelligence': 'analiseeestrategia',
    'analistademarketing': 'analiseeestrategia',
    'economista': 'analiseeestrategia',
    'productmanager/productowner(pm/apm/dpm/gpm/po)': 'analiseeestrategia',
    'cientistadedados/datascientist': 'cienciadedadoseestatistica',
    'engenheirodemachinelearning/mlengineer': 'cienciadedadoseestatistica',
    'estatistico': 'cienciadedadoseestatistica',
    'analistadedados/dataanalyst': 'cienciadedadoseestatistica',
    'engenheirodedados/arquitetodedados/dataengineer/dataarchitect': 'engenhariadedados',
    'analyticsengineer': 'engenhariadedados',
    'dba/administradordebancodedados': 'engenhariadedados',
    'desenvolvedor/engenheirodesoftware/analistadesistemas': 'engenhariadesoftware',
    'analistadesuporte/analistatecnico': 'engenhariadesoftware',
    'outrasengenharias(naoincluidev)': 'engenhariadesoftware',
    'professor': 'educacaoesuportetecnico',
    'outraopcao': 'educacaoesuportetecnico',
}

df["cargo"] = df["cargo"].map(lambda x: mapeamento_cargos.get(x, "outraopcao"))

# ==========================
# 4. Mapear bancos SQL/NoSQL
# ==========================
bancos_sql = {
    'sqlserver','mysql','postgresql','oracle','googlebigquery',
    'sqlite','saphana','snowflake','amazonauroraourds','mariadb',
    'db2','firebird','amazonredshift','microsoftaccess'
}
bancos_nosql = {
    's3','databricks','amazonathena','mongodb','hive','dynamodb',
    'presto','elasticsearch','elaticsearch','redis','firebase','splunk','nenhum',
    'cassandra','hbase','googlefirestore','neo4j','excel'
}

def map_bancos(col):
    col = col.fillna("").astype(str).str.lower()
    sql_count = col.apply(lambda x: sum(1 for s in bancos_sql if s in x))
    nosql_count = col.apply(lambda x: sum(1 for s in bancos_nosql if s in x))
    return pd.DataFrame({"sql_count": sql_count, "nosql_count": nosql_count})

df = pd.concat([df, map_bancos(df["bancos_de_dados"])], axis=1)
df = df.drop(columns=["bancos_de_dados"])

# ==========================
# 5. Tratar idade
# ==========================
df['idade'] = pd.to_numeric(df['idade'], errors='coerce').fillna(0).astype(int)

# ==========================
# 6. Label Encoding
# ==========================
categorical_cols = [
    "genero", "etnia", "estado_moradia", "nivel_ensino", "formacao",
    "tempo_experiencia_dados", "linguagens_preferidas", "cloud_preferida",
    "vive_no_brasil", "pcd", "cargo"
]

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].fillna("desconhecido")
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# ==========================
# 7. Separar X e y
# ==========================
X = df.drop(["cargo", "vive_no_brasil", "pcd"], axis=1)
y = df["cargo"]

# ==========================
# 8. Treino/Teste
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==========================
# 9. Treinar modelo
# ==========================
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# ==========================
# 10. Avaliação
# ==========================
y_pred = rf_model.predict(X_test)
print(f"\n✅ Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print("\n📊 Relatório de classificação:\n", classification_report(y_test, y_pred))

# ==========================
# 11. Salvar modelo
# ==========================
joblib.dump({
    "model": rf_model,
    "le_dict": le_dict,
    "feature_columns": X.columns.tolist()
}, "modelo_cargos.pkl")

print("🚀 Modelo salvo em 'modelo_cargos.pkl'")
