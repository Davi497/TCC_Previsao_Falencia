# %% [markdown]
# # Análise de Dados de Falência de Empresas (Versão 2 - com SMOTE)

# %% [markdown]
# ## 1. Configuração Inicial e Carregamento de Dados
# Importa as bibliotecas necessárias e carrega os datasets ARFF, concatenando-os em um único DataFrame.

# %% 
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Carregar os datasets ARFF
data1, meta1 = arff.loadarff("1year.arff")
data2, meta2 = arff.loadarff("2year.arff")
data3, meta3 = arff.loadarff("3year.arff")
data4, meta4 = arff.loadarff("4year.arff")
data5, meta5 = arff.loadarff("5year.arff")

# Converter para DataFrames e concatenar
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df3 = pd.DataFrame(data3)
df4 = pd.DataFrame(data4)
df5 = pd.DataFrame(data5)
df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

print(f"Shape inicial do DataFrame: {df.shape}")

# %% [markdown]
# ## 2. Exploração Inicial dos Dados
# Realiza uma exploração básica para entender a estrutura e o conteúdo do DataFrame.

# %% 
print("\nPrimeiras 5 linhas do DataFrame:")
print(df.head())

print("\nEstatísticas descritivas do DataFrame:")
print(df.describe())

print("\nInformações do DataFrame (tipos de dados e valores não nulos):")
print(df.info())

print("\nContagem de valores da coluna 'class' (antes do renomeio):")
print(df["class"].value_counts())

# %% [markdown]
# ## 3. Ajuste e Renomeação de Colunas
# Renomeia as colunas para nomes mais descritivos, facilitando a interpretação.

# %% 
colunas = {
    'Attr1': 'ROA',
    'Attr2': 'Passivo_Total_Ativo_Total',
    'Attr3': 'Capital_de_Giro_Ativo_Total',
    'Attr4': 'Ativo_Circulante_Passivo_Circulante',
    'Attr5': 'Fluxo_de_Caixa_Divida',
    'Attr6': 'Lucros_Retidos_Ativo_Total',
    'Attr7': 'EBIT_Ativo_Total',
    'Attr8': 'Patrimonio_Liquido_Passivo',
    'Attr9': 'Vendas_Ativo_Total',
    'Attr10': 'Indice_de_Capital_Proprio',
    'Attr11': 'EBITDA_Ativo_Total',
    'Attr12': 'EBIT_Despesas_Financeiras',
    'Attr13': 'Margem_Bruta',
    'Attr14': 'Margem_Operacional',
    'Attr15': 'Giro_da_Divida',
    'Attr16': 'Passivo_Total_Ativo_Total_2',
    'Attr17': 'Ativo_Total_Passivo_Total',
    'Attr18': 'Lucro_Bruto_Ativo_Total',
    'Attr19': 'Margem_Liquida',
    'Attr20': 'Vendas_Ativo_Total_2',
    'Attr21': 'Crescimento_das_Vendas',
    'Attr22': 'Resultado_Operacional_Ativo_Total',
    'Attr23': 'Lucro_Liquido_Vendas',
    'Attr24': 'Lucro_Bruto_Vendas',
    'Attr25': 'Patrimonio_Liquido_Ativo_Total',
    'Attr26': 'Lucro_Liquido_Patrimonio',
    'Attr27': 'Indice_de_Endividamento',
    'Attr28': 'Capital_de_Giro_Vendas',
    'Attr29': 'Caixa_Ativo_Total',
    'Attr30': 'Caixa_Vendas',
    'Attr31': 'Lucro_Operacional_Vendas',
    'Attr32': 'Ativo_Circulante_Ativo_Total',
    'Attr33': 'Estoque_Vendas',
    'Attr34': 'Estoque_Ativo_Circulante',
    'Attr35': 'Ativo_Total',
    'Attr36': 'Vendas',
    'Attr37': 'EBITDA',
    'Attr38': 'Lucro_Liquido',
    'Attr39': 'Lucro_Operacional',
    'Attr40': 'Capital_de_Giro',
    'Attr41': 'Ativo_Circulante',
    'Attr42': 'Passivo_Circulante',
    'Attr43': 'Passivo_Total',
    'Attr44': 'Patrimonio_Liquido',
    'Attr45': 'Lucro_Bruto',
    'Attr46': 'Lucros_Retidos',
    'Attr47': 'Fluxo_de_Caixa',
    'Attr48': 'Despesas_Operacionais',
    'Attr49': 'Despesas_Financeiras',
    'Attr50': 'Estoque',
    'Attr51': 'Contas_a_Receber',
    'Attr52': 'Contas_a_Pagar',
    'Attr53': 'Taxa_Crescimento_Vendas',
    'Attr54': 'Giro_do_Ativo',
    'Attr55': 'Indice_Capital_de_Giro',
    'Attr56': 'Margem_Operacional_2',
    'Attr57': 'Retorno_sobre_Vendas',
    'Attr58': 'Custo_Receita',
    'Attr59': 'Cobertura_de_Juros',
    'Attr60': 'Divida_Patrimonio_Liquido',
    'Attr61': 'Indice_de_Liquidez',
    'Attr62': 'Ciclo_de_Conversao_de_Caixa',
    'Attr63': 'Giro_de_Contas_a_Receber',
    'Attr64': 'Giro_de_Estoque',
    'class': 'Alvo'
}
df = df.rename(columns=colunas)

print("\nPrimeiras 5 linhas do DataFrame após renomear colunas:")
print(df.head())
print(f"Número de colunas após renomeio: {len(df.columns)}")

# %% [markdown]
# ## 4. Pré-processamento da Variável Alvo
# Decodifica e converte a coluna 'Alvo' para o tipo inteiro.

# %% 
print("\nTipo de dado da coluna 'Alvo' antes da conversão:")
print(df['Alvo'].dtype)
print("Valores únicos da coluna 'Alvo' antes da conversão:")
print(df['Alvo'].unique())

df['Alvo'] = df['Alvo'].str.decode('utf-8').astype(int)

print("\nContagem de valores da coluna 'Alvo' após a conversão:")
print(df['Alvo'].value_counts())
print("Tipo de dado da coluna 'Alvo' após a conversão:")
print(df['Alvo'].dtype)

# %% [markdown]
# ## 5. Tratamento de Valores Ausentes e Infinitos
# Identifica e trata valores ausentes (NaN) e infinitos, substituindo-os pela mediana das respectivas colunas.

# %% 
# Separar variáveis explicativas (X) e variável alvo (y)
X = df.drop('Alvo', axis=1)
y = df['Alvo']

print(f"\nTotal de valores nulos em X antes do tratamento: {X.isnull().sum().sum()}")
print(f"Total de valores infinitos em X antes do tratamento: {np.isinf(X).sum().sum()}")

# Troca infinitos por NaN para que o imputer possa tratá-los
X = X.replace([np.inf, -np.inf], np.nan)

# Imputa valores NaN pela mediana
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

print("\nInformações do DataFrame (tipos de dados e valores não nulos):")
print(df.info())


print(f"Total de valores nulos em X após o tratamento: {X.isnull().sum().sum()}")
print(f"Total de valores infinitos em X após o tratamento: {np.isinf(X).sum().sum()}")

# %% [markdown]
# ## 6. Divisão dos Dados em Conjuntos de Treino e Teste
# Divide o dataset em conjuntos de treino e teste para avaliação do modelo, garantindo estratificação da variável alvo.

# %% 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"\nShape de X_train: {X_train.shape}")
print(f"Shape de X_test: {X_test.shape}")
print(f"Shape de y_train: {y_train.shape}")
print(f"Shape de y_test: {y_test.shape}")
print("Valores únicos em y_train (após split):")
print(y_train.value_counts())

# %% [markdown]
# ## 7. Padronização dos Dados
# Aplica a padronização (StandardScaler) aos conjuntos de treino e teste para normalizar as features.

# %% 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Dados de treino e teste padronizados.")

# %% [markdown]
# ## 8. Aplicação da Técnica SMOTE (Apenas no Treino)
# Aplica o SMOTE nos dados de treino padronizados para balancear a classe minoritária (empresas falidas).

# %% 
print("\n--- Distribuição das Classes ANTES do SMOTE (Treino) ---")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("\n--- Distribuição das Classes APÓS o SMOTE (Treino) ---")
print(pd.Series(y_train_smote).value_counts())
print(f"Novo shape de X_train_smote: {X_train_smote.shape}")

# %% [markdown]
# ## 9. Modelo de Regressão Logística
# Treina um modelo de Regressão Logística com os dados balanceados pelo SMOTE e avalia no teste original.

# %% 
modelo_lr = LogisticRegression(
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)
modelo_lr.fit(X_train_smote, y_train_smote)

# Previsões no conjunto de teste original
y_pred_lr = modelo_lr.predict(X_test_scaled)

print("\nRelatório de Classificação (Regressão Logística com SMOTE):")
print(classification_report(y_test, y_pred_lr))

# Matriz de Confusão
cm_lr = confusion_matrix(y_test, y_pred_lr)
print("\nMatriz de Confusão (Regressão Logística):\n", cm_lr)

# Visualização da Matriz de Confusão
plt.figure(figsize=(6,4))
ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['Não Falida', 'Falida']).plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão (Regressão Logística - SMOTE)')
plt.show()

# %% [markdown]
# ## 10. Modelo de Random Forest
# Treina um modelo de Random Forest com os dados balanceados pelo SMOTE e avalia no teste original.

# %% 
modelo_rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
modelo_rf.fit(X_train_smote, y_train_smote)

# Previsões no conjunto de teste original
y_pred_rf = modelo_rf.predict(X_test_scaled)

print("\nRelatório de Classificação (Random Forest com SMOTE):")
print(classification_report(y_test, y_pred_rf))

# Matriz de Confusão
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nMatriz de Confusão (Random Forest):\n", cm_rf)

# Visualização da Matriz de Confusão
plt.figure(figsize=(6,4))
ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Não Falida', 'Falida']).plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão (Random Forest - SMOTE)')
plt.show()

# %% [markdown]
# ## 11. Modelo Gradient Boosting
# Treina um modelo HistGradientBoostingClassifier com os dados balanceados pelo SMOTE e avalia no teste original.

# %% 
modelo_hgb = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

modelo_hgb.fit(X_train_smote, y_train_smote)

# Previsões no conjunto de teste original
y_pred_hgb = modelo_hgb.predict(X_test_scaled)

print("\nRelatório de Classificação (Gradient Boosting com SMOTE):")
print(classification_report(y_test, y_pred_hgb))

# Matriz de Confusão
cm_hgb = confusion_matrix(y_test, y_pred_hgb)
print("\nMatriz de Confusão (Gradient Boosting):\n", cm_hgb)

# Visualização da Matriz de Confusão
plt.figure(figsize=(6,4))
ConfusionMatrixDisplay(confusion_matrix=cm_hgb, display_labels=['Não Falida', 'Falida']).plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão (Gradient Boosting - SMOTE)')
plt.show()

# %% [markdown]
# ## 12. Curva ROC Comparativa dos Modelos
# Plota as curvas ROC para os modelos treinados com SMOTE para comparar seus desempenhos.

# %% 
# Probabilidades previstas
prob_lr = modelo_lr.predict_proba(X_test_scaled)[:, 1]
prob_rf = modelo_rf.predict_proba(X_test_scaled)[:, 1]
prob_hgb = modelo_hgb.predict_proba(X_test_scaled)[:, 1]

# Curvas ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, prob_rf)
fpr_hgb, tpr_hgb, _ = roc_curve(y_test, prob_hgb)

# AUC
auc_lr = roc_auc_score(y_test, prob_lr)
auc_rf = roc_auc_score(y_test, prob_rf)
auc_hgb = roc_auc_score(y_test, prob_hgb)

# Plot
plt.figure(figsize=(9, 7))

plt.plot(
    fpr_lr,
    tpr_lr,
    linewidth=2,
    label=f'Regressão Logística (AUC = {auc_lr:.3f})'
)

plt.plot(
    fpr_rf,
    tpr_rf,
    linewidth=2,
    label=f'Random Forest (AUC = {auc_rf:.3f})'
)

plt.plot(
    fpr_hgb,
    tpr_hgb,
    linewidth=2,
    label=f'HistGradientBoosting (AUC = {auc_hgb:.3f})'
)

# Linha aleatória
plt.plot([0, 1], [0, 1], '--', label='Classificador Aleatório')

plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC – Comparação dos Modelos (com SMOTE)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)

plt.show()

# %% [markdown]
# ## 13. Tabela Comparativa dos Modelos
# Apresenta uma tabela comparativa com métricas de desempenho para cada modelo treinado com SMOTE.

# %% 
tabela_resultados = pd.DataFrame({
    'Modelo': [
        'Regressão Logística (SMOTE)',
        'Random Forest (SMOTE)',
        'Gradient Boosting (SMOTE)'
    ],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_hgb)
    ],
    'Precision (Classe Falida)': [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_hgb)
    ],
    'Recall (Classe Falida)': [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_hgb)
    ],
    'F1-Score (Classe Falida)': [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_hgb)
    ],
    'ROC-AUC': [
        auc_lr,
        auc_rf,
        auc_hgb
    ]
})

# Arredondar para 3 casas decimais
tabela_resultados = tabela_resultados.round(3)

print("\nTabela Comparativa dos Modelos:")
print(tabela_resultados)

# Estilização
tabela_estilizada = (
    tabela_resultados.style
    .hide(axis='index')
    .format({
        'Accuracy': '{:.3f}',
        'Precision (Classe Falida)': '{:.3f}',
        'Recall (Classe Falida)': '{:.3f}',
        'F1-Score (Classe Falida)': '{:.3f}',
        'ROC-AUC': '{:.3f}'
    })
    .highlight_max(
        subset=[
            'Accuracy',
            'Precision (Classe Falida)',
            'Recall (Classe Falida)',
            'F1-Score (Classe Falida)',
            'ROC-AUC'
        ],
        color='#C6EFCE'   # verde claro
    )
    .set_caption('Tabela 1 – Comparação do desempenho dos modelos preditivos com SMOTE')
    .set_table_styles([
        {
            'selector': 'caption',
            'props': [
                ('font-size', '14px'),
                ('font-weight', 'bold'),
                ('text-align', 'center')
            ]
        },
        {
            'selector': 'th',
            'props': [
                ('background-color', '#4472C4'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center')
            ]
        },
        {
            'selector': 'td',
            'props': [
                ('text-align', 'center')
            ]
        }
    ])
)

tabela_estilizada
# %%
