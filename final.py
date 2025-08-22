import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Carregar dados
url = "https://raw.githubusercontent.com/Weverton-Cristian/Processing-Dataset_-Intelligent_Agent/master/dataset.csv"
df = pd.read_csv(url)
df = df.drop_duplicates()
df = df.dropna(subset=["cargo"])
df = df[df["cargo"].astype(str).str.strip() != ""]

# Substituir valores raros (menos de 35 ocorrências)
for col in df.columns:
    raros = df[col].value_counts()[df[col].value_counts() < 35].index
    df.loc[df[col].isin(raros), col] = 'Outra Opção'
    df.drop(df[df[col] == 'Outra Opção'].index, inplace=True)

# Definir colunas
categoricas = ["estado_moradia", "etnia", "linguagens_preferidas", "vive_no_brasil", 
               "nivel_ensino", "formacao", "pcd", "cloud_preferida", "genero"]
numericas = ["idade", "tempo_experiencia_dados"]

# LabelEncoder para variáveis textuais que não vão virar dummies
le = LabelEncoder()
df['cargo'] = le.fit_transform(df['cargo'])

# Pré-processamento
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ("num", "passthrough", numericas)
    ]
)

# Pipeline completo
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1))
])

# Separar features e alvo
X = df.drop("cargo", axis=1)
y = df["cargo"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Treinar modelo
pipeline.fit(X_train, y_train)

# Testar modelo
y_pred = pipeline.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Salvar pipeline completo
joblib.dump((pipeline, le), "modelo_cargos_pipeline.pkl")
