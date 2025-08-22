import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Carregar dados
url = "https://raw.githubusercontent.com/Weverton-Cristian/Processing-Dataset_-Intelligent_Agent/master/dataset.csv"
df = pd.read_csv(url)

# 2. Limpeza básica
df = df.drop_duplicates()
df = df.dropna(subset=["cargo"])
df = df[df["cargo"].astype(str).str.strip() != ""]

# 3. One-hot para algumas colunas
df = pd.get_dummies(df, columns=["estado_moradia", "etnia", "formacao", "linguagens_preferidas"])

# 4. Remover categorias raras (<30 ocorrências)
for col in df.columns:
    raros = df[col].value_counts()[df[col].value_counts() < 30].index
    df.loc[df[col].isin(raros), col] = 'Outra Opção'
    df = df[df[col] != 'Outra Opção']

# 5. Tratar idade
df['idade'] = pd.to_numeric(df['idade'], errors='coerce').fillna(0).astype(int)

# 6. Mapear bancos de dados
bancos_sql = {'sqlserver','mysql','postgresql','oracle','googlebigquery',
              'sqlite','saphana','snowflake','amazonauroraourds','mariadb',
              'db2','firebird','amazonredshift','microsoftaccess'}

bancos_nosql = {'s3','databricks','amazonathena','mongodb','hive','dynamodb',
                'presto','elaticsearch','redis','firebase','splunk','nenhum',
                'cassandra','hbase','googlefirestore','neo4j','excel'}

def map_bancos(col):
    col = col.str.lower()
    sql_count = col.apply(lambda x: sum(1 for s in bancos_sql if s in x))
    nosql_count = col.apply(lambda x: sum(1 for s in bancos_nosql if s in x))
    return pd.DataFrame({"sql_count": sql_count, "nosql_count": nosql_count})

bancos_mapped = map_bancos(df["bancos_de_dados"])
df = pd.concat([df, bancos_mapped], axis=1)
df = df.drop(columns=["bancos_de_dados"])

# 7. LabelEncoding colunas categóricas
le_dict = {}
categorical_cols = ["genero", "tempo_experiencia_dados", "pcd",
                    "vive_no_brasil", "cargo", "cloud_preferida", "nivel_ensino"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# 8. Separar X e y
X = df.drop(["cargo", "vive_no_brasil", "pcd", "nosql_count", "sql_count"], axis=1)
y = df["cargo"]

# 9. Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 10. Treinar modelo
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 11. Avaliar
y_pred = rf_model.predict(X_test)
print(f"\n✅ Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print("\n📊 Relatório de classificação:\n", classification_report(y_test, y_pred))
print("\n📉 Matriz de confusão:\n", confusion_matrix(y_test, y_pred))

# 12. Salvar modelo + encoders + colunas
joblib.dump({
    "model": rf_model,
    "le_dict": le_dict,
    "feature_columns": X.columns.tolist()
}, "modelo_cargos.pkl")
