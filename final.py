import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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

# 4. Remover categorias raras (menos de 35 ocorrências)
for col in df.columns:
    raros = df[col].value_counts()[df[col].value_counts() < 35].index
    df.loc[df[col].isin(raros), col] = 'Outra Opção'
    df = df[df[col] != 'Outra Opção']

# 5. Tratar idade
df['idade'] = pd.to_numeric(df['idade'], errors='coerce').fillna(0).astype(int)

# 6. LabelEncoding para colunas categóricas restantes
le_dict = {}
for col in ["genero", "tempo_experiencia_dados", "bancos_de_dados", "pcd",
            "vive_no_brasil", "cargo", "cloud_preferida", "nivel_ensino"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# 7. Separar X e y
X = df.drop(["cargo", "vive_no_brasil", "pcd"], axis=1)
y = df["cargo"]

# 8. Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 9. Treinar modelo
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 10. Avaliar
y_pred = rf_model.predict(X_test)
print(f"\n✅ Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print("\n📊 Relatório de classificação:\n", classification_report(y_test, y_pred))

# 11. Salvar tudo
joblib.dump({
    "model": rf_model,
    "le_dict": le_dict,
    "feature_columns": X.columns.tolist()
}, "modelo_cargos.pkl")
