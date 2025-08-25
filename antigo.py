import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Carregar dados
url = "https://raw.githubusercontent.com/Weverton-Cristian/Processing-Dataset_-Intelligent_Agent/master/dataset.csv"
df = pd.read_csv(url)

# 2. Limpeza básica
df = df.drop_duplicates()
df = df.dropna(subset=["cargo"])
df = df[df["cargo"].astype(str).str.strip() != ""]

# 3. Definição das categorias para o One-Hot Encoder
# Adicionando 'Outra Opção' para garantir que categorias desconhecidas sejam tratadas
# Você não precisa mais remover linhas raras, o pré-processador irá lidar com isso
categorical_features = ['genero', 'etnia', 'estado_moradia', 'pcd', 'vive_no_brasil', 
                        'nivel_ensino', 'formacao', 'tempo_experiencia_dados', 
                        'linguagens_preferidas', 'cloud_preferida']
one_hot_features = ['genero', 'etnia', 'estado_moradia', 'formacao', 'cloud_preferida']
label_encode_features = ['pcd', 'vive_no_brasil', 'nivel_ensino', 'linguagens_preferidas',
                         'tempo_experiencia_dados']

# 4. Criar um transformador customizado para os bancos de dados
class BancosDeDadosTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bancos_sql = {'sqlserver', 'mysql', 'postgresql', 'oracle', 'googlebigquery',
                           'sqlite', 'saphana', 'snowflake', 'amazonauroraourds', 'mariadb',
                           'db2', 'firebird', 'amazonredshift', 'microsoftaccess'}
        self.bancos_nosql = {'s3', 'databricks', 'amazonathena', 'mongodb', 'hive', 'dynamodb',
                             'presto', 'elaticsearch', 'redis', 'firebase', 'splunk', 'nenhum',
                             'cassandra', 'hbase', 'googlefirestore', 'neo4j', 'excel'}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.astype(str).str.lower()
        sql_count = X.apply(lambda x: sum(1 for s in self.bancos_sql if s in x))
        nosql_count = X.apply(lambda x: sum(1 for s in self.bancos_nosql if s in x))
        return np.c_[sql_count, nosql_count]

# 5. Criar o pré-processador do pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), one_hot_features),
        ('label', LabelEncoder(), label_encode_features),
        ('bancos', BancosDeDadosTransformer(), ['bancos_de_dados']),
        # Note: 'idade' não foi incluída pois não há uma transformação clara para ela.
        # Se precisar de colunas numéricas, adicione-as aqui.
    ],
    remainder='drop'
)

# 6. Preparar os dados para o treinamento
y = df["cargo"]
X = df.drop("cargo", axis=1)

# A sua InputData tem 13 colunas, a sua entrada para a pipeline precisa ter as 13 colunas.
# Aqui vamos alinhar X com as colunas que a sua API irá enviar.
X_api_columns = ['idade', 'genero', 'etnia', 'pcd', 'vive_no_brasil', 'estado_moradia',
                 'nivel_ensino', 'formacao', 'tempo_experiencia_dados', 
                 'linguagens_preferidas', 'bancos_de_dados', 'cloud_preferida']
X = X[X_api_columns]

# 7. Treinar o modelo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1))
])

# 8. Codificar a variável target e treinar a pipeline
le_cargo = LabelEncoder()
y_encoded = le_cargo.fit_transform(y)
pipeline.fit(X, y_encoded)

# 9. Salvar a pipeline e o mapeamento de cargos
joblib.dump(pipeline, "modelo_pipeline.pkl")
joblib.dump(dict(zip(le_cargo.classes_, le_cargo.transform(le_cargo.classes_))), "mapeamento_cargos.pkl")

print("\n✅ Pipelines de pré-processamento e modelo treinados com sucesso!")
print("✅ Arquivos 'modelo_pipeline.pkl' e 'mapeamento_cargos.pkl' salvos.")

# Opcional: Teste a acurácia
y_pred = pipeline.predict(X)
print(f"\n📊 Acurácia no conjunto de treino: {accuracy_score(y_encoded, y_pred):.4f}")
joblib.dump({
    "model": pipeline,
    "le_dict": le_cargo,
    "feature_columns": X.columns.tolist()
}, "modelo_cargos.pkl")
