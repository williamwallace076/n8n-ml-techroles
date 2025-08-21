import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.model_selection import train_test_split

url = "https://raw.githubusercontent.com/Weverton-Cristian/Processing-Dataset_-Intelligent_Agent/master/dataset.csv"
df = pd.read_csv(url)
df = df.drop_duplicates()
df = df.dropna(subset=["cargo"])
df = df[df["cargo"].astype(str).str.strip() != ""]

df = pd.get_dummies(df, columns=["estado_moradia"])
df = pd.get_dummies(df, columns=["etnia"])
df['cloud_preferida'].value_counts()
df["genero"].value_counts()
df["cargo"].value_counts()

for i in df.columns:
  raros = df[i].value_counts()[df[i].value_counts() < 35].index
  df.loc[df[i].isin(raros), i] = 'Outra Opção'
  df.drop(df[df[i] == 'Outra Opção'].index, inplace=True)

le = LabelEncoder()
df['genero'] = le.fit_transform(df['genero'])
df['formacao'] = le.fit_transform(df['formacao'])
df['tempo_experiencia_dados'] = le.fit_transform(df['tempo_experiencia_dados'])
df['linguagens_preferidas'] = le.fit_transform(df['linguagens_preferidas'])
df['bancos_de_dados'] = le.fit_transform(df['bancos_de_dados']) 
df['pcd'] = le.fit_transform(df['pcd'])
df['vive_no_brasil'] = le.fit_transform(df['vive_no_brasil'])
df['cargo'] = le.fit_transform(df['cargo'])
df['cloud_preferida'] = le.fit_transform(df['cloud_preferida'])
df['nivel_ensino'] = le.fit_transform(df['nivel_ensino'])

y = df["cargo"]
X = df.drop(["cargo", "vive_no_brasil"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y # Importante para manter a proporção do alvo nos dois conjuntos
)

rf_model = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do modelo: {accuracy:.4f}")

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

joblib.dump(rf_model, "modelo_cargos.pkl")

