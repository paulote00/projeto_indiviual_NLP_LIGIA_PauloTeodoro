import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

df_train = pd.read_csv('/kaggle/input/ligia-nlp/train.csv')
df_test = pd.read_csv('/kaggle/input/ligia-nlp/test.csv')

df_train['full_text'] = df_train['title'].fillna('') + " " + df_train['text'].fillna('')
df_test['full_text'] = df_test['title'].fillna('') + " " + df_test['text'].fillna('')

X = df_train['full_text']
y = df_train['label']
X_test_final = df_test['full_text']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = make_pipeline(
    TfidfVectorizer(max_features=10000, stop_words='english'),
    LogisticRegression(class_weight='balanced', random_state=42, max_iter=500)
)

pipeline.fit(X_train, y_train)

y_pred_val = pipeline.predict(X_val)
f1 = f1_score(y_val, y_pred_val, average='macro')
conf_matrix = confusion_matrix(y_val, y_pred_val)

print(f"F1-Score (Macro): {f1:.4f}")
print("Matriz de Confusão:\n", conf_matrix)

explainer = LimeTextExplainer(class_names=['Fake (0)', 'True (1)'])

idx_fake = np.where(y_val == 0)[0][0]
idx_true = np.where(y_val == 1)[0][0]

exp_fake = explainer.explain_instance(X_val.iloc[idx_fake], pipeline.predict_proba, num_features=10)
exp_fake.save_to_file('lime_fake_explanation.html')

exp_true = explainer.explain_instance(X_val.iloc[idx_true], pipeline.predict_proba, num_features=10)
exp_true.save_to_file('lime_true_explanation.html')

y_test_pred = pipeline.predict(X_test_final)
submission = pd.DataFrame({'id': df_test['id'], 'target': y_test_pred})
submission.to_csv('submission.csv', index=False)

# --- COMENTÁRIOS DA EXECUÇÃO ---
# pd.read_csv: Carrega os arquivos usando os caminhos exatos mapeados na nuvem do Kaggle.
# df_train['full_text']: Mescla o título e o corpo da notícia para garantir o máximo de contexto ao algoritmo, tratando vazios com fillna('').
# train_test_split: Separa 20% do volume para validação interna, mantendo a proporção exata de classes reais e falsas (stratify=y).
# make_pipeline: Empacota a conversão de texto para matemática (TfidfVectorizer) e o treinamento do modelo (LogisticRegression) em um fluxo único.
# f1_score / confusion_matrix: Extrai as métricas de performance exigidas para a etapa de Resultados do artigo.
# LimeTextExplainer: Executa a análise de interpretabilidade (XAI), isolando quais palavras específicas ativaram a decisão do modelo.
# exp_fake.save_to_file: Salva a análise visual do LIME em formato de página web (.html) para você capturar a tela.
# submission.to_csv: Gera a tabela final padronizada para o Leaderboard da competição.
