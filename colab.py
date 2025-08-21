import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("dataset.csv")
df = df.drop_duplicates()
df = df.dropna(subset=["cargo"])
df = df[df["cargo"].astype(str).str.strip() != ""]

df = pd.get_dummies(df, columns=["estado_moradia"])
df = pd.get_dummies(df, columns=["etnia"])
# df = pd.get_dummies(df, columns=["cloud_preferida"])
# df.dropna(subset=['cargo'], inplace=True)
df['cloud_preferida'].value_counts()

# df.dropna(inplace=True)
# df[df["cargo"] == "Professor"]

# df = df.drop(columns=["etnia","pcd", "vive_no_brasil", "estado_moradia", "nivel_ensino", "cloud_preferida"])
df["genero"].value_counts()
# df.drop(df[df["genero"] == "Prefiro não informar"].index, inplace=True)

df["cargo"].value_counts()


for i in df.columns:
  raros = df[i].value_counts()[df[i].value_counts() < 35].index
  df.loc[df[i].isin(raros), i] = 'Outra Opção'
  df.drop(df[df[i] == 'Outra Opção'].index, inplace=True)


# contagem_cargos = df['cargo'].value_counts()
# limiar = 60
# cargos_raros = contagem_cargos[contagem_cargos < limiar].index

# df.loc[df['cargo'].isin(cargos_raros), 'cargo'] = 'Outra Opção'

# print("Nova distribuição após agrupar cargos raros:")
# print(df['cargo'].value_counts())

# df.drop(df[df["cargo"] =="Outra Opção"].index, inplace=True)
# df.drop(df[df["cargo"] =="Outras Engenharias (não inclui dev)"].index, inplace=True)

le = LabelEncoder()
df['genero'] = le.fit_transform(df['genero'])
df['formacao'] = le.fit_transform(df['formacao'])
df['tempo_experiencia_dados'] = le.fit_transform(df['tempo_experiencia_dados'])
df['linguagens_preferidas'] = le.fit_transform(df['linguagens_preferidas'])
df['bancos_de_dados'] = le.fit_transform(df['bancos_de_dados']) # TODO: COMEÇAR POR AQUI
# df['etnia'] = le.fit_transform(df['etnia']) #
df['pcd'] = le.fit_transform(df['pcd'])
df['vive_no_brasil'] = le.fit_transform(df['vive_no_brasil'])
df['cargo'] = le.fit_transform(df['cargo'])
# df['estado_moradia'] = le.fit_transform(df['estado_moradia']) #
df['cloud_preferida'] = le.fit_transform(df['cloud_preferida']) 
df['nivel_ensino'] = le.fit_transform(df['nivel_ensino'])


y = df["cargo"]

# O 'X' são todas as outras colunas (as features)
# Usamos o método .drop() para remover a coluna alvo
X = df.drop(["cargo", "vive_no_brasil"], axis=1)

from sklearn.model_selection import train_test_split

# Dividir os dados: 70% para treino, 30% para teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42, # Para reprodutibilidade
    stratify=y # Importante para manter a proporção do alvo nos dois conjuntos
)

rf_model = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)

rf_model.fit(X_train, y_train)

# Fazer previsões nos dados de teste
y_pred = rf_model.predict(X_test)

# Avaliar o desempenho
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do modelo: {accuracy:.4f}")

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))